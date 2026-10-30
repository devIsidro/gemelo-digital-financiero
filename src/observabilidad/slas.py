"""
Acuerdos de nivel de servicio (SLA) del Gemelo Digital Financiero (Fase 6).

Cada SLA es una promesa medible sobre la plataforma, con su objetivo y cómo
se mide. Se evalúan igual en el Panel Operativo y en el último paso del DAG
(revisar_slas), que además manda las alertas.

- Duración del pipeline: la corrida completa termina en 15 min o menos
  (lo típico es ~5).
- Frescura de los datos: hubo una carga exitosa en las últimas 26 h (el DAG
  es diario, con 2 h de margen).
- Tasa de éxito: 90% o más de las últimas 30 corridas terminan bien.
- Calidad de datos: 100% de las reglas de negocio de Silver se cumplen.
- Consistencia entre capas: 100% de las revisiones Bronze → Silver → Gold.
- Asistente de IA: 95% o más de respuestas correctas en la evaluación con el
  modelo real y 10 s o menos por respuesta.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

OBJETIVOS = {
    "duracion_min": 15.0,
    "frescura_horas": 26.0,
    "tasa_exito_pct": 90.0,
    "calidad_pct": 100.0,
    "ia_correctas_pct": 95.0,
    "ia_segundos": 10.0,
}


@dataclass
class ResultadoSLA:
    clave: str
    nombre: str
    objetivo: str
    real: str
    cumple: object  # True, False o None (sin datos para medirlo)
    detalle: str = ""

    def __post_init__(self):
        if self.cumple is not None:
            self.cumple = bool(self.cumple)


def _sin_datos(clave, nombre, objetivo, detalle) -> ResultadoSLA:
    return ResultadoSLA(clave, nombre, objetivo, "—", None, detalle)


# ---------------------------------------------------------------------------
# Cada SLA
# ---------------------------------------------------------------------------


def sla_duracion(corridas, corrida_actual=None) -> ResultadoSLA:
    clave, nombre = "duracion", "Duración del pipeline"
    objetivo = f"≤ {OBJETIVOS['duracion_min']:.0f} min"
    if corrida_actual is not None:
        minutos, de_donde = corrida_actual["minutos"], "esta corrida"
    elif corridas is not None and (corridas["state"] == "success").any():
        exitosa = corridas[corridas["state"] == "success"].iloc[-1]
        minutos, de_donde = float(exitosa["minutos"]), "la última corrida exitosa"
    else:
        return _sin_datos(clave, nombre, objetivo, "Todavía no hay corridas exitosas.")
    cumple = minutos <= OBJETIVOS["duracion_min"]
    return ResultadoSLA(
        clave, nombre, objetivo, f"{minutos:.1f} min", cumple, f"Medido en {de_donde}."
    )


def sla_frescura(corridas, ahora, corrida_actual=None) -> ResultadoSLA:
    clave, nombre = "frescura", "Frescura de los datos"
    objetivo = f"≤ {OBJETIVOS['frescura_horas']:.0f} h desde la última carga"
    if corrida_actual is not None and not corrida_actual["fallidos"]:
        return ResultadoSLA(
            clave, nombre, objetivo, "0.0 h", True, "Los datos se acaban de cargar."
        )
    if corridas is None or not (corridas["state"] == "success").any():
        return _sin_datos(clave, nombre, objetivo, "Todavía no hay corridas exitosas.")
    fin = corridas[corridas["state"] == "success"]["fin"].dropna().max()
    horas = (ahora - fin).total_seconds() / 3600
    cumple = horas <= OBJETIVOS["frescura_horas"]
    return ResultadoSLA(
        clave,
        nombre,
        objetivo,
        f"{horas:.1f} h",
        cumple,
        f"Última carga exitosa: {fin:%d/%m/%Y %H:%M} h.",
    )


def sla_tasa_exito(corridas) -> ResultadoSLA:
    clave, nombre = "tasa_exito", "Tasa de éxito de corridas"
    objetivo = f"≥ {OBJETIVOS['tasa_exito_pct']:.0f}% (últimas 30)"
    if corridas is None:
        return _sin_datos(clave, nombre, objetivo, "Sin conexión a Airflow.")
    terminadas = corridas[corridas["state"].isin(["success", "failed"])]
    if terminadas.empty:
        return _sin_datos(
            clave, nombre, objetivo, "Todavía no hay corridas terminadas."
        )
    exitosas = int((terminadas["state"] == "success").sum())
    pct = exitosas / len(terminadas) * 100
    return ResultadoSLA(
        clave,
        nombre,
        objetivo,
        f"{pct:.0f}%",
        pct >= OBJETIVOS["tasa_exito_pct"],
        f"{exitosas} de {len(terminadas)} corridas terminadas salieron bien.",
    )


def sla_calidad(reglas) -> ResultadoSLA:
    clave, nombre = "calidad", "Calidad de datos"
    objetivo = "100% de las reglas de Silver"
    if not reglas:
        return _sin_datos(clave, nombre, objetivo, "Todavía no hay datos en Silver.")
    ok = sum(1 for r in reglas if r.ok)
    fallan = [r.nombre for r in reglas if not r.ok]
    return ResultadoSLA(
        clave,
        nombre,
        objetivo,
        f"{ok} de {len(reglas)}",
        ok == len(reglas),
        "Todas se cumplen." if not fallan else "No se cumple: " + "; ".join(fallan),
    )


def sla_consistencia(chequeos) -> ResultadoSLA:
    clave, nombre = "consistencia", "Consistencia entre capas"
    objetivo = "100% de las revisiones"
    medidos = [c for c in chequeos if c.ok is not None]
    if not medidos:
        return _sin_datos(clave, nombre, objetivo, "Todavía no existen las capas.")
    ok = sum(1 for c in medidos if c.ok)
    fallan = [c.nombre for c in medidos if not c.ok]
    return ResultadoSLA(
        clave,
        nombre,
        objetivo,
        f"{ok} de {len(medidos)}",
        ok == len(medidos),
        (
            "Bronze, Silver y Gold cuadran."
            if not fallan
            else "No cuadra: " + "; ".join(fallan)
        ),
    )


def sla_asistente(evaluacion) -> ResultadoSLA:
    clave, nombre = "asistente", "Asistente de IA"
    objetivo = (
        f"≥ {OBJETIVOS['ia_correctas_pct']:.0f}% correctas y "
        f"≤ {OBJETIVOS['ia_segundos']:.0f} s por respuesta"
    )
    if not evaluacion:
        return _sin_datos(
            clave,
            nombre,
            objetivo,
            "Todavía no se corre la evaluación (python -m src.ia.evaluar).",
        )
    pct = evaluacion["correctas"] / evaluacion["total"] * 100
    cumple = (
        pct >= OBJETIVOS["ia_correctas_pct"]
        and evaluacion["segundos"] <= OBJETIVOS["ia_segundos"]
    )
    return ResultadoSLA(
        clave,
        nombre,
        objetivo,
        f"{pct:.0f}% · {evaluacion['segundos']:.1f} s",
        cumple,
        f"{evaluacion['correctas']} de {evaluacion['total']} correctas en la evaluación "
        f"del {evaluacion['fecha']:%d/%m/%Y %H:%M} h.",
    )


# ---------------------------------------------------------------------------
# Todo junto
# ---------------------------------------------------------------------------


def leer_evaluacion(ruta: Path):
    """Resumen de la última evaluación del asistente (src/ia/evaluar.py), o
    None si todavía no se corre."""
    ruta = Path(ruta)
    if not ruta.exists():
        return None
    res = pd.read_csv(ruta)
    if res.empty or "ok" not in res.columns:
        return None
    return {
        "correctas": int(res["ok"].astype(bool).sum()),
        "total": len(res),
        "segundos": float(res["segundos"].mean()),
        "fecha": pd.Timestamp(ruta.stat().st_mtime, unit="s", tz="UTC").tz_convert(
            "America/Mexico_City"
        ),
    }


def evaluar(corridas, reglas, chequeos, evaluacion, ahora, corrida_actual=None) -> list:
    """Los seis SLA, en el orden en que se muestran."""
    return [
        sla_duracion(corridas, corrida_actual),
        sla_frescura(corridas, ahora, corrida_actual),
        sla_tasa_exito(corridas),
        sla_calidad(reglas),
        sla_consistencia(chequeos),
        sla_asistente(evaluacion),
    ]
