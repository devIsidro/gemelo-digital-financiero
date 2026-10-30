"""
Alertas del Gemelo Digital Financiero (Fase 6).

Cuando un paso del DAG falla o un SLA no se cumple:

1. La alerta se guarda en data/observabilidad/alertas.jsonl, y el Panel
   Operativo la muestra (funciona siempre, sin servicios externos).
2. Si hay un correo configurado, se manda un aviso. La configuración va en
   variables de entorno (archivo .env, que no se sube a GitHub):

     ALERTAS_SMTP_USUARIO     cuenta que envía (p. ej. una cuenta de Gmail)
     ALERTAS_SMTP_CONTRASENA  contraseña de aplicación de esa cuenta
     ALERTAS_CORREO_PARA      quién recibe (si falta, el mismo usuario)
     ALERTAS_SMTP_HOST        por defecto smtp.gmail.com
     ALERTAS_SMTP_PUERTO      por defecto 587 (STARTTLS)

   Sin esas variables el correo se omite y todo lo demás sigue igual.
"""

import json
import os
import smtplib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

import pandas as pd

RUTA_ALERTAS = Path("data/observabilidad/alertas.jsonl")
URL_PANEL = os.environ.get("ALERTAS_PANEL_URL", "http://localhost:8501")


@dataclass
class Alerta:
    nivel: str  # "crítica" o "advertencia"
    titulo: str
    detalle: str
    origen: str  # p. ej. "paso clean_silver" o "SLA frescura"
    fecha: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


# SLA que, si no se cumplen, son críticos (los datos podrían estar mal); el
# resto son advertencias (la plataforma funciona, pero fuera del objetivo).
SLA_CRITICOS = {"calidad", "consistencia", "frescura"}


def de_slas(resultados) -> list:
    """Una alerta por cada SLA que no se cumple (los «sin datos» no alertan)."""
    return [
        Alerta(
            "crítica" if r.clave in SLA_CRITICOS else "advertencia",
            f"No se cumple el SLA «{r.nombre}»",
            f"Objetivo: {r.objetivo}. Real: {r.real}. {r.detalle}".strip(),
            f"SLA {r.clave}",
        )
        for r in resultados
        if r.cumple is False
    ]


# ---------------------------------------------------------------------------
# Registro (lo que lee el Panel Operativo)
# ---------------------------------------------------------------------------


def registrar(alertas, estado_correo: str, ruta: Path = RUTA_ALERTAS) -> None:
    if not alertas:
        return
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("a", encoding="utf-8") as f:
        for a in alertas:
            f.write(
                json.dumps({**asdict(a), "correo": estado_correo}, ensure_ascii=False)
            )
            f.write("\n")


def historial(ruta: Path = RUTA_ALERTAS, n: int = 20) -> pd.DataFrame:
    """Las últimas n alertas, de la más reciente a la más vieja."""
    ruta = Path(ruta)
    columnas = ["fecha", "nivel", "titulo", "detalle", "origen", "correo"]
    if not ruta.exists():
        return pd.DataFrame(columns=columnas)
    filas = []
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        try:
            filas.append(json.loads(linea))
        except json.JSONDecodeError:
            continue  # una línea dañada no debe tumbar el panel
    df = pd.DataFrame(filas, columns=columnas)
    if df.empty:
        return df
    df["fecha"] = pd.to_datetime(df["fecha"], utc=True, errors="coerce").dt.tz_convert(
        "America/Mexico_City"
    )
    return df.sort_values("fecha", ascending=False).head(n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Correo
# ---------------------------------------------------------------------------


def config_correo(entorno=None):
    """La configuración del correo, o None si no está completa."""
    entorno = os.environ if entorno is None else entorno
    usuario = entorno.get("ALERTAS_SMTP_USUARIO", "").strip()
    contrasena = entorno.get("ALERTAS_SMTP_CONTRASENA", "").strip()
    if not usuario or not contrasena:
        return None
    return {
        "host": entorno.get("ALERTAS_SMTP_HOST", "").strip() or "smtp.gmail.com",
        "puerto": int(entorno.get("ALERTAS_SMTP_PUERTO", "").strip() or 587),
        "usuario": usuario,
        "contrasena": contrasena,
        "para": entorno.get("ALERTAS_CORREO_PARA", "").strip() or usuario,
    }


def armar_correo(alertas, remitente: str, para: str) -> EmailMessage:
    n = len(alertas)
    criticas = sum(1 for a in alertas if a.nivel == "crítica")
    asunto = f"[Gemelo Digital] {n} alerta{'s' if n > 1 else ''}"
    if criticas:
        asunto += f" ({criticas} crítica{'s' if criticas > 1 else ''})"
    asunto += f": {alertas[0].titulo}"
    lineas = ["Alertas del pipeline del Gemelo Digital Financiero:", ""]
    for a in alertas:
        lineas += [
            f"[{a.nivel.upper()}] {a.titulo}",
            f"  {a.detalle}",
            f"  Origen: {a.origen}",
            "",
        ]
    lineas += [f"Panel Operativo: {URL_PANEL}", "Airflow: http://localhost:8080"]
    msg = EmailMessage()
    msg["Subject"] = asunto
    msg["From"] = remitente
    msg["To"] = para
    msg.set_content("\n".join(lineas))
    return msg


def enviar_correo(alertas, config=None, smtp=smtplib.SMTP) -> str:
    """Manda un solo correo con todas las alertas. Regresa cómo le fue:
    «enviado», «no configurado» o «error: …» (nunca truena: una alerta que
    no se pudo mandar no debe tumbar el DAG)."""
    if not alertas:
        return "sin alertas"
    config = config_correo() if config is None else config
    if config is None:
        return "no configurado"
    try:
        msg = armar_correo(alertas, config["usuario"], config["para"])
        with smtp(config["host"], config["puerto"], timeout=15) as servidor:
            servidor.starttls()
            servidor.login(config["usuario"], config["contrasena"])
            servidor.send_message(msg)
        return "enviado"
    except Exception as err:  # noqa: BLE001
        return f"error: {str(err)[:120]}"


def notificar(
    alertas, ruta: Path = RUTA_ALERTAS, config=None, smtp=smtplib.SMTP
) -> str:
    """Correo (si está configurado) + registro para el panel."""
    if not alertas:
        return "sin alertas"
    estado = enviar_correo(alertas, config, smtp)
    registrar(alertas, estado, ruta)
    return estado
