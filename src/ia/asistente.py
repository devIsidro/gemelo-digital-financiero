"""
Asistente financiero con IA generativa local (Fase 5).

Arquitectura RAG con Text-to-SQL, como pide el documento del proyecto
(sección 11): LangChain conecta un modelo Llama 3 que corre localmente en
Ollama con los datos de la capa Gold.

Para cada pregunta:

  1. Clasificar: el modelo decide si es una consulta de datos, un "¿qué
     pasaría si gasto más o menos?" o un "¿cuándo junto X?" (JSON).
  2. Recuperar:
       - datos  -> el modelo escribe el SQL (Text-to-SQL) sobre las tablas
                   del cliente (src/ia/contexto.py); se valida y se ejecuta
                   en DuckDB. Si falla, se le regresa el error una vez.
       - gasto  -> simulador Monte Carlo con el gasto modificado.
       - meta   -> simulador Monte Carlo: en cuántos meses junta el monto.
  3. Redactar: el modelo responde en español claro usando SOLO esos datos.

Los modelos se reciben ya creados, así que las pruebas usan modelos falsos
y no necesitan Ollama.
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.ia import contexto as cx
from src.ia import hechos as hc
from src.ia import herramientas as hz

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODELO = os.environ.get("OLLAMA_MODELO", "llama3.1:8b")

TIPOS = ("datos", "gasto", "meta", "otro")
CARPETA_PROMPTS = Path(__file__).parent / "prompts"


def _prompt(nombre: str) -> str:
    """Los prompts viven en src/ia/prompts/ para poder leerlos y ajustarlos
    sin tocar el código."""
    return (CARPETA_PROMPTS / nombre).read_text(encoding="utf-8").strip()


CAMBIO_GASTO_DEFECTO = 0.10

PREGUNTAS_EJEMPLO = (
    "¿Cuál es mi nivel de riesgo?",
    "¿Qué ocurriría si aumento mis gastos un 20%?",
    "¿Cuándo podría juntar $50,000?",
    "¿Por qué disminuyó mi capacidad de ahorro el mes pasado?",
)

MENSAJE_OTRO = (
    "Puedo ayudarte con preguntas sobre tus finanzas: tu nivel de riesgo, tus "
    "gastos por mes o por categoría, tu capacidad de ahorro, qué pasaría si "
    "gastas más o menos, cuándo podrías juntar cierta cantidad o cómo te "
    "afectaría un crédito hipotecario, un cambio de empleo o abrir un negocio."
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SISTEMA_CLASIFICAR = _prompt("clasificar.txt")

SISTEMA_SQL = _prompt("sql.txt")

SISTEMA_RESPUESTA = _prompt("respuesta.txt")

PLANTILLA_RESPUESTA = _prompt("respuesta_datos.txt")


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

COLUMNAS_PORCENTAJE = {
    "capacidad_ahorro",
    "capacidad_ahorro_12m",
    "prob_impago",
    "prob_numeros_rojos",
    "prob_lograrlo_12m",
    "prob_lograrlo_36m",
}
COLUMNAS_SIN_DINERO = {
    "dti",
    "score_credito",
    "plazo_meses",
    "num_compras",
    "tasa_interes_anual",
}


def _es_dinero(col: str) -> bool:
    return (
        col not in COLUMNAS_SIN_DINERO
        and not col.startswith(("meses", "num_", "mes"))
        and any(
            p in col
            for p in (
                "ingreso",
                "gasto",
                "ahorro",
                "pago",
                "monto",
                "flujo",
                "meta",
                "diferencia",
            )
        )
    )


def formatear_df(df: pd.DataFrame) -> pd.DataFrame:
    """Valores ya escritos como los leería una persona: porcentajes y dólares
    (un modelo pequeño se equivoca menos que si tiene que convertirlos)."""
    out = df.copy()
    for col in out.columns:
        serie = out[col]
        if pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(
            serie
        ):
            if col == "tasa_interes_anual":
                out[col] = serie.map(lambda v: "—" if pd.isna(v) else f"{v:.1f}%")
            elif col in COLUMNAS_PORCENTAJE:
                out[col] = serie.map(lambda v: "—" if pd.isna(v) else f"{v * 100:.1f}%")
            elif _es_dinero(col):
                out[col] = serie.map(
                    lambda v: (
                        "—"
                        if pd.isna(v)
                        else (f"-${-v:,.0f}" if v < 0 else f"${v:,.0f}")
                    )
                )
            else:
                out[col] = serie.map(
                    lambda v: "—" if pd.isna(v) else f"{v:,.2f}".rstrip("0").rstrip(".")
                )
        else:
            out[col] = serie.astype(str)
    return out


def formatear(df: pd.DataFrame) -> str:
    """Tabla en texto (columnas separadas por |), como se ve en «Cómo lo calculé»."""
    if df is None or df.empty:
        return "(sin resultados)"
    return formatear_df(df).to_csv(index=False, sep="|").strip()


# Nombres que entiende una persona (y el modelo) para cada columna.
ETIQUETAS_COLUMNAS = {
    "mes": "Fecha del mes",
    "mes_nombre": "Mes",
    "ingreso": "Ingreso del mes",
    "gasto": "Gasto",
    "ahorro": "Ahorro del mes (ingreso menos gasto)",
    "capacidad_ahorro": "Capacidad de ahorro (parte del ingreso que ahorró)",
    "num_compras": "Número de compras",
    "categoria": "Categoría",
    "tipo": "Tipo de gasto",
    "gasto_total": "Gasto total",
    "gasto_ultimo_mes": "Gasto en el último mes",
    "gasto_mes_anterior": "Gasto en el mes anterior",
    "diferencia": "Diferencia contra el mes anterior",
    "factor": "Factor",
    "efecto": "Efecto en su riesgo de impago",
    "escenario": "Escenario",
    "prob_numeros_rojos": (
        "Probabilidad de quedarse en números rojos (ahorro acumulado bajo cero) en 36 meses"
    ),
    "ahorro_12m": "Ahorro acumulado típico a 12 meses",
    "ahorro_12m_bajo": "Ahorro a 12 meses en un caso malo",
    "ahorro_12m_alto": "Ahorro a 12 meses en un caso bueno",
    "ahorro_36m": "Ahorro acumulado típico a 36 meses",
    "flujo_mes_1": "Flujo de efectivo típico del primer mes",
    "capacidad_ahorro_12m": "Capacidad de ahorro típica del primer año",
    "meta_fondo_emergencia": "Meta del fondo de emergencia (6 meses de gasto)",
    "meses_fondo_emergencia": "Meses típicos para juntar el fondo de emergencia",
    "situacion": "Situación",
    "gasto_mensual": "Gasto mensual",
    "meta": "Meta de ahorro",
    "meses_tipico": "Meses típicos para lograrla",
    "meses_caso_bueno": "Meses en un caso bueno",
    "meses_caso_malo": "Meses en un caso malo",
    "prob_lograrlo_12m": "Probabilidad de lograrla en 12 meses",
    "prob_lograrlo_36m": "Probabilidad de lograrla en 36 meses",
    "ahorro_mensual_tipico": "Ahorro mensual típico",
}


def formatear_para_modelo(df: pd.DataFrame, max_filas: int = 20) -> str:
    """Una línea por fila con «Etiqueta: valor»: a un modelo pequeño le cuesta
    menos leerlo bien que una tabla con nombres técnicos."""
    if df is None or df.empty:
        return "(sin resultados)"
    tabla = formatear_df(df.head(max_filas))
    if "mes" in tabla.columns and "mes_nombre" in tabla.columns:
        tabla = tabla.drop(columns="mes")
    lineas = []
    for _, fila in tabla.iterrows():
        partes = [
            f"{ETIQUETAS_COLUMNAS.get(c, str(c).replace('_', ' '))}: {v}"
            for c, v in fila.items()
        ]
        lineas.append("- " + "; ".join(partes))
    if len(df) > max_filas:
        lineas.append(f"(y {len(df) - max_filas} filas más)")
    return "\n".join(lineas)


ETIQUETAS_PERFIL = {
    "cliente": "Cliente",
    "ingreso_mensual": "Ingreso mensual",
    "gasto_mensual_promedio": "Gasto mensual promedio con tarjeta",
    "ahorro_mensual": "Ahorro mensual (ingreso menos gasto)",
    "capacidad_ahorro": "Capacidad de ahorro (parte del ingreso que ahorra)",
    "prob_impago": "Probabilidad de no pagar su préstamo",
    "nivel_riesgo": "Nivel de riesgo de impago",
    "pago_mensual_prestamo": "Pago mensual de su préstamo",
    "ingreso_mensual_prestamo": "Ingreso declarado en su préstamo",
    "dti": "Pago del préstamo entre su ingreso declarado (veces)",
    "monto_prestamo": "Monto de su préstamo",
    "tasa_interes_anual": "Tasa de interés anual del préstamo",
    "plazo_meses": "Plazo del préstamo (meses)",
    "score_credito": "Score de crédito",
    "categoria_principal": "Categoría donde más gasta",
    "num_compras": "Compras con tarjeta en el histórico",
    "cliente_desde": "Cliente desde",
}


def formatear_perfil(perfil: pd.DataFrame) -> str:
    """El perfil (una fila) como lista de «dato: valor»."""
    fila = formatear_df(perfil).iloc[0]
    return "\n".join(f"- {ETIQUETAS_PERFIL.get(c, c)}: {v}" for c, v in fila.items())


def leer_json(texto: str) -> dict:
    m = re.search(r"\{.*\}", texto, flags=re.S)
    if not m:
        raise ValueError("sin JSON")
    return json.loads(m.group(0))


def _numero(v):
    try:
        return (
            float(str(v).replace(",", "").replace("$", "")) if v is not None else None
        )
    except ValueError:
        return None


def historial_texto(historial: list, turnos: int = 2) -> str:
    if not historial:
        return "(ninguna)"
    lineas = []
    n = 2 * turnos
    for msg in historial[-n:]:
        quien = "Cliente" if msg["rol"] == "user" else "Asistente"
        lineas.append(f"{quien}: {msg['texto']}")
    return "\n".join(lineas)


def ollama_disponible(
    url: str = OLLAMA_URL, modelo: str = OLLAMA_MODELO, timeout: float = 3
) -> tuple:
    """(True, "") si Ollama responde y tiene el modelo; si no, (False, motivo)."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"{url.rstrip('/')}/api/tags", timeout=timeout
        ) as r:
            modelos = [m.get("name", "") for m in json.load(r).get("models", [])]
    except (urllib.error.URLError, OSError, ValueError):
        return False, "sin_servicio"
    nombres = set(modelos) | {m.removesuffix(":latest") for m in modelos}
    if modelo not in nombres:
        return False, "sin_modelo"
    return True, ""


def crear_modelos(url: str = OLLAMA_URL, modelo: str = OLLAMA_MODELO) -> dict:
    """Los tres usos del mismo modelo local: clasificar (JSON), escribir SQL
    y redactar la respuesta."""
    from langchain_ollama import ChatOllama

    comun = {"model": modelo, "base_url": url}
    return {
        "json": ChatOllama(**comun, temperature=0, format="json"),
        "sql": ChatOllama(**comun, temperature=0),
        "texto": ChatOllama(**comun, temperature=0.1),
    }


# ---------------------------------------------------------------------------
# Asistente
# ---------------------------------------------------------------------------


# Muletillas que el modelo a veces escribe aunque el prompt las prohíbe
# («Según los datos, …», «ATENCIÓN: …», consejos genéricos que nadie pidió).
# Se quitan aquí, en Python, para no depender de que el modelo haga caso.
_FUENTE = (
    r"(?:según|de acuerdo con|con base en|basándo(?:me|nos) en)\s+"
    r"(?:los|las|la|el|tus|sus)\s+(?:datos|información|hechos(?:\s+clave)?|"
    r"cifras|resultados)"
    r"(?:\s+(?:de\s+apoyo|clave|que\s+(?:me\s+)?(?:diste|tengo|hay)|"
    r"proporcionad[oa]s|disponibles|anteriores))?"
)
_FUENTE_INICIO = re.compile(r"(^|[.!?]\s+)" + _FUENTE + r"\s*,\s*(\w)", re.I)
_FUENTE_ENTRE_COMAS = re.compile(r",\s+" + _FUENTE + r"\s*,", re.I)
# «Lo siento, pero según los datos, en…» -> «Lo siento, pero en…»
_FUENTE_TRAS_PALABRA = re.compile(r"(?<=\w)\s+" + _FUENTE + r"\s*,(?=\s)", re.I)
_FUENTE_FINAL = re.compile(r",?\s+" + _FUENTE + r"(?=\s*[.;!?])", re.I)
_ETIQUETA = re.compile(
    r"(^|[.!?]\s+)(?:atenci[óo]n|nota|importante|aclaraci[óo]n)\s*:\s*(\w)", re.I
)
# Una oración completa: termina en . ! o ? seguido de espacio o del final
# (así «7.6%» o «$1.5» no cortan la oración).
_RESTO_ORACION = r"(?:[^.!?]|[.!?](?!\s|$))*[.!?](?=\s|$)"
_CONSEJO = re.compile(
    r"(?:^|(?<=[.!?]))\s*"
    r"(?:(?:además|por eso|por lo tanto|así que|por último|finalmente)\s*,\s*)?"
    r"(?:recomendación\s*:|mi recomendación|te recomiendo|recomiendo|te sugiero|"
    r"sugiero|te aconsejo|considera|podrías considerar|"
    r"(?:sería|es) (?:bueno|recomendable|conveniente|importante))" + _RESTO_ORACION,
    re.I,
)
# Preguntas en las que el cliente sí pide consejo.
_PIDE_CONSEJO = re.compile(
    r"recomiend|recomendaci|consej|suger|qu[ée] (?:debo|puedo|deber[íi]a|hago)|"
    r"c[óo]mo (?:puedo|podr[íi]a|hago|mejor|ahorr|baj|reduz|reduc|logr|junt)",
    re.I,
)
_FIN_ORACION = re.compile(r"[.!?](?=\s)")


def pide_consejo(pregunta: str) -> bool:
    return bool(_PIDE_CONSEJO.search(pregunta or ""))


def limpiar_respuesta(texto: str, consejos: bool = False) -> str:
    """Quita «Según los datos, …», etiquetas como «ATENCIÓN:» y, si el cliente
    no pidió consejo, las recomendaciones genéricas. Lo demás no se toca."""
    texto = _FUENTE_INICIO.sub(lambda r: r.group(1) + r.group(2).upper(), texto)
    texto = _FUENTE_ENTRE_COMAS.sub("", texto)
    texto = _FUENTE_TRAS_PALABRA.sub("", texto)
    texto = _FUENTE_FINAL.sub("", texto)
    texto = _ETIQUETA.sub(lambda r: r.group(1) + r.group(2).upper(), texto)
    if not consejos:
        texto = _CONSEJO.sub("", texto)
    return texto.lstrip()


def limpiar_stream(partes, consejos: bool = False):
    """Igual que limpiar_respuesta, pero sobre una respuesta que llega en partes:
    se muestra oración por oración, cuando ya se sabe si hay que quitarla."""
    texto, enviado = "", 0
    for parte in partes:
        texto += parte
        limpio = limpiar_respuesta(texto, consejos)
        fines = [m.end() for m in _FIN_ORACION.finditer(limpio)]
        seguro = fines[-1] if fines else 0
        if seguro > enviado:
            yield limpio[enviado:seguro]
            enviado = seguro
    limpio = limpiar_respuesta(texto, consejos)
    if len(limpio) > enviado:
        yield limpio[enviado:]


@dataclass
class Preparacion:
    """Lo que se recuperó para responder (se muestra en «Cómo lo calculé»)."""

    pregunta: str
    tipo: str
    datos: pd.DataFrame = None
    sql: str = None
    origen: str = ""
    nota: str = ""
    intentos: list = field(default_factory=list)
    hechos: list = field(default_factory=list)
    # Respuesta completa escrita en Python (pregunta con un dato equivocado):
    # se muestra tal cual, sin pasar por el modelo.
    respuesta_directa: str = ""
    # (claves, frase): si la respuesta no menciona ninguna clave, se agrega la
    # frase al final. Así las causas del riesgo siempre aparecen.
    obligatorio: tuple = None


class Asistente:
    def __init__(
        self,
        modelos: dict,
        tablas: dict,
        insumo: pd.Series = None,
        supuestos: dict = None,
        cc_num=None,
    ):
        self.modelos = modelos
        self.tablas = tablas
        self.insumo = insumo
        self.supuestos = supuestos or {}
        self.cc_num = cc_num
        self.con = cx.conectar(tablas)
        texto = StrOutputParser()
        self._clasificar = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", SISTEMA_CLASIFICAR),
                    (
                        "human",
                        "Conversación previa:\n{historial}\n\nPregunta: {pregunta}",
                    ),
                ]
            )
            | modelos["json"]
            | texto
        )
        self._sql = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", SISTEMA_SQL),
                    (
                        "human",
                        "Conversación previa:\n{historial}\n\nPregunta: {pregunta}{error}\nSQL:",
                    ),
                ]
            )
            | modelos["sql"]
            | texto
        )
        self._respuesta = (
            ChatPromptTemplate.from_messages(
                [("system", SISTEMA_RESPUESTA), ("human", PLANTILLA_RESPUESTA)]
            )
            | modelos["texto"]
            | texto
        )

    # -- 1. Clasificar ------------------------------------------------------
    def clasificar(self, pregunta: str, historial: list = None) -> dict:
        try:
            plan = leer_json(
                self._clasificar.invoke(
                    {"pregunta": pregunta, "historial": historial_texto(historial)}
                )
            )
        except (ValueError, json.JSONDecodeError):
            plan = {}
        tipo = str(plan.get("tipo", "datos")).strip().lower()
        if tipo not in TIPOS:
            tipo = "datos"
        if tipo in ("gasto", "meta") and self.insumo is None:
            tipo = "datos"  # sin datos del simulador se contesta con lo que hay
        cambio = _numero(plan.get("cambio_gasto"))
        if tipo == "gasto":
            if cambio is None:
                cambio = CAMBIO_GASTO_DEFECTO
            elif abs(cambio) > 1 and abs(cambio) <= 300:
                cambio = cambio / 100  # el modelo puso 20 en lugar de 0.2
        return {"tipo": tipo, "cambio_gasto": cambio, "meta": _numero(plan.get("meta"))}

    # -- 2. Recuperar -------------------------------------------------------
    def consultar(
        self, pregunta: str, historial: list = None, intentos: int = 2
    ) -> Preparacion:
        prep = Preparacion(
            pregunta, "datos", origen="su historial real de compras y su perfil"
        )
        error = ""
        for _ in range(intentos):
            sql = cx.limpiar_sql(
                self._sql.invoke(
                    {
                        "esquema": cx.describir_esquema(),
                        "pregunta": pregunta,
                        "historial": historial_texto(historial),
                        "error": error,
                    }
                )
            )
            prep.intentos.append(sql)
            try:
                prep.datos = cx.ejecutar_sql(self.con, sql)
                prep.sql = sql
                return prep
            except cx.SQLInvalido as err:
                error = (
                    f"\n\nTu consulta anterior falló:\n{sql}\nError: {err}\n"
                    "Corrígela usando solo las tablas y columnas permitidas."
                )
        # Sin SQL válido: se responde con el resumen mensual y se avisa.
        prep.datos = self.tablas["mis_meses"]
        prep.nota = (
            "No se pudo armar una consulta exacta para esta pregunta; estos son "
            "los datos mensuales generales del cliente."
        )
        return prep

    def preparar(self, pregunta: str, historial: list = None) -> Preparacion:
        prep = self._recuperar(pregunta, historial)
        prep.hechos = self.hechos(prep)
        if prep.tipo == "datos":
            prep.respuesta_directa = hc.revisar_premisa(pregunta, self.tablas)
        return prep

    def hechos(self, prep: Preparacion) -> list:
        """Las comparaciones clave, calculadas en Python (src/ia/hechos.py)."""
        if prep.tipo == "gasto":
            return hc.hechos_gasto(prep.datos)
        if prep.tipo == "meta":
            return hc.hechos_meta(prep.datos)
        if prep.tipo != "datos":
            return []
        texto = prep.pregunta.lower()
        columnas = set(prep.datos.columns) if prep.datos is not None else set()
        de_riesgo = bool(columnas & {"factor", "nivel_riesgo", "prob_impago"}) or any(
            p in texto for p in ("riesgo", "impago", "préstamo", "prestamo", "pagar")
        )
        de_meses = bool(
            columnas & {"mes", "mes_nombre", "capacidad_ahorro", "gasto", "ahorro"}
        ) or any(p in texto for p in ("mes", "ahorr", "gast", "capacidad"))
        hechos = hc.hechos_riesgo(self.tablas) if de_riesgo else []
        if de_riesgo:
            sube, _, claves = hc.factores_riesgo(self.tablas)
            prep.obligatorio = (claves, sube) if sube else None
        if de_meses or not de_riesgo:
            hechos += hc.hechos_meses(self.tablas)
        return hechos

    def _recuperar(self, pregunta: str, historial: list = None) -> Preparacion:
        plan = self.clasificar(pregunta, historial)
        tipo = plan["tipo"]
        if tipo == "otro":
            return Preparacion(pregunta, "otro")
        if tipo == "gasto":
            datos = hz.simular_cambio_gasto(
                self.insumo, plan["cambio_gasto"], self.supuestos, self.cc_num
            )
            return Preparacion(
                pregunta,
                "gasto",
                datos=datos,
                origen="simulación Monte Carlo de 1,000 futuros a 36 meses",
                nota=(
                    f"Cambio de gasto simulado: {plan['cambio_gasto']:+.0%} "
                    "sobre su gasto mensual típico."
                ),
            )
        if tipo == "meta":
            monto = plan["meta"]
            nota = ""
            if not monto or monto <= 0:
                monto = (
                    float(
                        self.tablas["mis_escenarios"]["meta_fondo_emergencia"].iloc[0]
                    )
                    if not self.tablas["mis_escenarios"].empty
                    else 6 * float(self.insumo["gasto_base"])
                )
                nota = (
                    "No dijo un monto: se usa como meta un fondo de emergencia "
                    "de 6 meses de gasto."
                )
            datos = hz.meta_ahorro(self.insumo, monto, self.supuestos, self.cc_num)
            return Preparacion(
                pregunta,
                "meta",
                datos=datos,
                origen="simulación Monte Carlo de 1,000 futuros a 36 meses, si sigue como hoy",
                nota=nota,
            )
        return self.consultar(pregunta, historial)

    # -- 3. Redactar --------------------------------------------------------
    def entradas_respuesta(self, prep: Preparacion, historial: list = None) -> dict:
        return {
            "perfil": formatear_perfil(self.tablas["mi_perfil"]),
            "origen": prep.origen,
            "datos": formatear_para_modelo(prep.datos),
            "nota": f"\nNota: {prep.nota}\n" if prep.nota else "",
            "contexto_extra": (
                "\nDetalle de cada mes (solo si necesitas más datos):\n"
                + formatear_para_modelo(self.tablas["mis_meses"])
                + "\n"
                if prep.tipo == "datos" and prep.datos is not self.tablas["mis_meses"]
                else ""
            ),
            "hechos": "\n".join(f"- {h}" for h in prep.hechos) or "(ninguno)",
            "pregunta": prep.pregunta,
        }

    def redactar(self, prep: Preparacion, historial: list = None):
        """Generador con la respuesta en partes (para mostrarla mientras se escribe)."""
        if prep.tipo == "otro":
            yield MENSAJE_OTRO
            return
        if prep.respuesta_directa:
            yield prep.respuesta_directa
            return
        texto = ""
        for parte in limpiar_stream(
            self._respuesta.stream(self.entradas_respuesta(prep, historial)),
            pide_consejo(prep.pregunta),
        ):
            texto += parte
            yield parte
        if prep.obligatorio:
            claves, frase = prep.obligatorio
            if not any(c in texto for c in claves):
                yield ("" if texto.endswith((" ", "\n")) or not texto else " ") + frase

    def responder(self, pregunta: str, historial: list = None) -> tuple:
        """(texto, preparación) de una sola vez."""
        prep = self.preparar(pregunta, historial)
        return "".join(self.redactar(prep, historial)), prep
