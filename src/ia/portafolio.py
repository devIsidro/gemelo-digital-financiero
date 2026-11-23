"""
Preguntas sobre todo el portafolio de clientes (asistente de IA, Fase 5).

El asistente de src/ia/asistente.py contesta sobre UN cliente. Este módulo
le agrega las preguntas que haría alguien del negocio sobre el portafolio
completo, por ejemplo «¿cuántos clientes están en riesgo alto?» o «¿en qué
categoría se gasta más?».

Funciona igual que el asistente de un cliente (Text-to-SQL sobre DuckDB con
las mismas reglas de seguridad), pero con tablas agregadas de todos los
clientes:

  - clientes                     una fila por cliente (identificador anónimo)
  - resumen_niveles              una fila por nivel de riesgo
  - meses_portafolio             totales de cada mes
  - gastos_categoria_portafolio  gasto de cada mes y categoría
  - escenarios_portafolio        impacto de cada escenario del simulador

Python calcula el panorama general (clientes por nivel de riesgo, cuántos
gastan más de lo que ganan) y el modelo solo redacta.
"""

import json

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.dashboard.datos import CATEGORIAS
from src.dashboard.datos_ejecutivo import (
    TIPO_CONSUMO,
    UMBRAL_RIESGO_DEFICIT,
    nombre_mes,
)
from src.ia import asistente as ia
from src.ia import contexto as cx
from src.ml.explicacion import NIVELES
from src.ml.simulador import ESCENARIOS, NOMBRES_ESCENARIOS

ESQUEMA = {
    "clientes": (
        "Una fila por cliente del portafolio. Montos en USD.",
        {
            "cliente": "identificador anónimo, por ejemplo 'Cliente 0065'",
            "ingreso_mensual": "ingreso mensual simulado (USD)",
            "gasto_mensual_promedio": "gasto promedio al mes con tarjeta (USD)",
            "ahorro_mensual": "ingreso_mensual menos gasto_mensual_promedio (USD)",
            "capacidad_ahorro": "parte del ingreso que ahorra, de 0 a 1",
            "prob_impago": "probabilidad de no pagar su préstamo, de 0 a 1",
            "nivel_riesgo": "'Bajo' (< 10%), 'Medio' (10% a 30%) o 'Alto' (>= 30%)",
            "dti": "pago mensual del préstamo entre el ingreso declarado",
            "score_credito": "score crediticio del préstamo (300 a 850)",
            "monto_prestamo": "monto del préstamo (USD)",
            "tasa_interes_anual": "tasa de interés del préstamo, en % anual",
            "categoria_principal": "categoría donde más gasta",
            "num_compras": "número de compras con tarjeta en el histórico",
            "prob_numeros_rojos": (
                "probabilidad de que su ahorro baje de cero en 36 meses si sigue "
                "como hoy, de 0 a 1"
            ),
        },
    ),
    "resumen_niveles": (
        "Una fila por nivel de riesgo de impago (Bajo, Medio y Alto).",
        {
            "nivel_riesgo": "'Bajo', 'Medio' o 'Alto'",
            "num_clientes": "número de clientes en ese nivel",
            "porcentaje_clientes": "parte del portafolio en ese nivel, de 0 a 1",
            "prob_impago_promedio": "probabilidad de impago promedio, de 0 a 1",
            "ingreso_mediano": "ingreso mensual mediano (USD)",
            "capacidad_ahorro_mediana": "capacidad de ahorro mediana, de 0 a 1",
            "monto_prestamo_total": "suma de los montos de sus préstamos (USD)",
        },
    ),
    "meses_portafolio": (
        "Una fila por mes completo del histórico (jul a dic 2020), en orden.",
        {
            "mes": "primer día del mes (DATE)",
            "mes_nombre": "por ejemplo 'nov 2020'",
            "num_clientes_con_compras": "clientes que compraron ese mes",
            "gasto_total": "gasto de todos los clientes ese mes (USD)",
            "gasto_promedio_cliente": "gasto_total entre num_clientes_con_compras (USD)",
            "num_compras": "número de compras del mes",
        },
    ),
    "gastos_categoria_portafolio": (
        "Una fila por mes y categoría de gasto de todos los clientes.",
        {
            "mes": "primer día del mes (DATE)",
            "mes_nombre": "por ejemplo 'nov 2020'",
            "categoria": "categoría en español, por ejemplo 'Restaurantes'",
            "tipo": "'Esencial' o 'Discrecional'",
            "gasto_total": "gasto de todos los clientes en esa categoría ese mes (USD)",
            "num_clientes": "clientes que compraron en esa categoría ese mes",
            "num_compras": "número de compras",
        },
    ),
    "escenarios_portafolio": (
        "Una fila por escenario del simulador Monte Carlo (36 meses).",
        {
            "escenario": (
                "'Sin cambios', 'Crédito hipotecario', 'Cambio de empleo' o "
                "'Abrir un negocio'"
            ),
            "num_clientes_en_riesgo": (
                "clientes con al menos 10% de probabilidad de quedar en números rojos"
            ),
            "porcentaje_en_riesgo": "parte del portafolio en riesgo, de 0 a 1",
            "ahorro_12m_mediano": "ahorro acumulado mediano a 12 meses (USD)",
            "ahorro_36m_mediano": "ahorro acumulado mediano a 36 meses (USD)",
        },
    ),
}

TABLAS = tuple(ESQUEMA)

PORCENTAJES = ia.COLUMNAS_PORCENTAJE | {
    "porcentaje_clientes",
    "prob_impago_promedio",
    "capacidad_ahorro_mediana",
    "porcentaje_en_riesgo",
}

ETIQUETAS = {
    **ia.ETIQUETAS_COLUMNAS,
    "cliente": "Cliente",
    "ingreso_mensual": "Ingreso mensual",
    "gasto_mensual_promedio": "Gasto mensual promedio",
    "ahorro_mensual": "Ahorro mensual (ingreso menos gasto)",
    "capacidad_ahorro": "Capacidad de ahorro",
    "prob_impago": "Probabilidad de impago",
    "nivel_riesgo": "Nivel de riesgo de impago",
    "dti": "Pago del préstamo entre ingreso declarado (veces)",
    "score_credito": "Score de crédito",
    "monto_prestamo": "Monto del préstamo",
    "tasa_interes_anual": "Tasa de interés anual",
    "categoria_principal": "Categoría donde más gasta",
    "num_compras": "Número de compras",
    "num_clientes": "Número de clientes",
    "porcentaje_clientes": "Porcentaje del portafolio",
    "prob_impago_promedio": "Probabilidad de impago promedio",
    "ingreso_mediano": "Ingreso mensual mediano",
    "capacidad_ahorro_mediana": "Capacidad de ahorro mediana",
    "monto_prestamo_total": "Monto total de préstamos",
    "num_clientes_con_compras": "Clientes con compras en el mes",
    "gasto_total": "Gasto total",
    "gasto_promedio_cliente": "Gasto promedio por cliente",
    "num_clientes_en_riesgo": "Clientes en riesgo de números rojos",
    "porcentaje_en_riesgo": "Parte del portafolio en riesgo de números rojos",
    "ahorro_12m_mediano": "Ahorro acumulado mediano a 12 meses",
    "ahorro_36m_mediano": "Ahorro acumulado mediano a 36 meses",
}

PREGUNTAS_EJEMPLO = (
    "¿Cuántos clientes están en riesgo alto?",
    "¿En qué categoría gastan más los clientes?",
    "¿Quiénes son los 5 clientes con mayor probabilidad de impago?",
    "¿Cuántos clientes gastan más de lo que ganan?",
)

MENSAJE_OTRO = (
    "Puedo ayudarte con preguntas sobre el portafolio de clientes: cuántos hay en "
    "cada nivel de riesgo, quiénes tienen mayor probabilidad de impago, en qué "
    "categorías y meses se gasta más, cuántos gastan más de lo que ganan o cómo "
    "afectaría a los clientes un crédito hipotecario, un cambio de empleo o abrir "
    "un negocio."
)

SISTEMA_CLASIFICAR = ia._prompt("clasificar_portafolio.txt")
SISTEMA_SQL = ia._prompt("sql_portafolio.txt")
SISTEMA_RESPUESTA = ia._prompt("respuesta_portafolio.txt")
PLANTILLA_RESPUESTA = ia._prompt("respuesta_portafolio_datos.txt")


def describir_esquema() -> str:
    partes = []
    for tabla, (desc, columnas) in ESQUEMA.items():
        cols = "\n".join(f"  - {c}: {d}" for c, d in columnas.items())
        partes.append(f"{tabla}: {desc}\n{cols}")
    return "\n\n".join(partes)


# ---------------------------------------------------------------------------
# Tablas del portafolio
# ---------------------------------------------------------------------------


def _mes(periodo) -> pd.Timestamp:
    return periodo.to_timestamp().normalize()


def tablas_portafolio(
    clientes: pd.DataFrame, tx: pd.DataFrame, meses: list, resumen: pd.DataFrame = None
) -> dict:
    """Las tablas de ESQUEMA a partir de lo que ya carga el dashboard.

    clientes: datos_ejecutivo.cargar_clientes. tx: transacciones de Silver.
    meses: meses completos. resumen: simulacion_resumen (o None).
    """
    nivel = clientes["nivel"].astype(str)
    deficit = None
    if resumen is not None and not resumen.empty:
        deficit = resumen[resumen["escenario"] == "base"].set_index("cc_num")[
            "prob_deficit"
        ]
    tabla_clientes = pd.DataFrame(
        {
            "cliente": clientes["cliente"],
            "ingreso_mensual": clientes["ingreso_mensual_simulado"].round(2),
            "gasto_mensual_promedio": clientes["gasto_promedio_mensual"].round(2),
            "ahorro_mensual": (
                clientes["ingreso_mensual_simulado"]
                - clientes["gasto_promedio_mensual"]
            ).round(2),
            "capacidad_ahorro": clientes["capacidad_ahorro"].round(4),
            "prob_impago": clientes["prob_impago"].round(4),
            "nivel_riesgo": nivel,
            "dti": clientes["ratio_endeudamiento"].round(4),
            "score_credito": clientes["score_credito_fuente"].astype("Int64"),
            "monto_prestamo": clientes["monto_prestamo_fuente"].round(2),
            "tasa_interes_anual": clientes["tasa_interes_anual_fuente"].astype(float),
            "categoria_principal": clientes["categoria_principal"]
            .map(CATEGORIAS)
            .fillna(clientes["categoria_principal"]),
            "num_compras": clientes["num_transacciones"].astype(int),
            "prob_numeros_rojos": (
                clientes["cc_num"].map(deficit).round(4)
                if deficit is not None
                else float("nan")
            ),
        }
    ).reset_index(drop=True)

    g = clientes.assign(nivel_txt=nivel).groupby("nivel_txt")
    niveles = pd.DataFrame(
        {
            "num_clientes": g.size(),
            "prob_impago_promedio": g["prob_impago"].mean().round(4),
            "ingreso_mediano": g["ingreso_mensual_simulado"].median().round(2),
            "capacidad_ahorro_mediana": g["capacidad_ahorro"].median().round(4),
            "monto_prestamo_total": g["monto_prestamo_fuente"].sum().round(2),
        }
    ).reindex(list(NIVELES))
    niveles["num_clientes"] = niveles["num_clientes"].fillna(0).astype(int)
    niveles.insert(
        1,
        "porcentaje_clientes",
        (niveles["num_clientes"] / max(len(clientes), 1)).round(4),
    )
    niveles = niveles.rename_axis("nivel_riesgo").reset_index()

    tx = tx[tx["cc_num"].isin(clientes["cc_num"])]
    tx = tx.assign(periodo=tx["trans_date_trans_time"].dt.to_period("M"))
    tx = tx[tx["periodo"].isin(meses)]
    por_mes = (
        tx.groupby("periodo")
        .agg(
            num_clientes_con_compras=("cc_num", "nunique"),
            gasto_total=("amt", "sum"),
            num_compras=("amt", "count"),
        )
        .reindex(meses, fill_value=0)
    )
    meses_tabla = pd.DataFrame(
        {
            "mes": [_mes(p) for p in meses],
            "mes_nombre": [nombre_mes(p) for p in meses],
            "num_clientes_con_compras": por_mes["num_clientes_con_compras"]
            .astype(int)
            .to_numpy(),
            "gasto_total": por_mes["gasto_total"].round(2).to_numpy(),
            "num_compras": por_mes["num_compras"].astype(int).to_numpy(),
        }
    )
    meses_tabla.insert(
        4,
        "gasto_promedio_cliente",
        (
            meses_tabla["gasto_total"]
            / meses_tabla["num_clientes_con_compras"].where(
                meses_tabla["num_clientes_con_compras"] > 0
            )
        ).round(2),
    )

    cat = (
        tx.groupby(["periodo", "category"])
        .agg(
            gasto_total=("amt", "sum"),
            num_clientes=("cc_num", "nunique"),
            num_compras=("amt", "count"),
        )
        .reset_index()
        .sort_values(["periodo", "gasto_total"], ascending=[True, False])
    )
    categorias = pd.DataFrame(
        {
            "mes": [_mes(p) for p in cat["periodo"]],
            "mes_nombre": [nombre_mes(p) for p in cat["periodo"]],
            "categoria": cat["category"]
            .map(CATEGORIAS)
            .fillna(cat["category"])
            .to_numpy(),
            "tipo": cat["category"].map(TIPO_CONSUMO).fillna("Discrecional").to_numpy(),
            "gasto_total": cat["gasto_total"].round(2).to_numpy(),
            "num_clientes": cat["num_clientes"].astype(int).to_numpy(),
            "num_compras": cat["num_compras"].astype(int).to_numpy(),
        }
    )

    return {
        "clientes": tabla_clientes,
        "resumen_niveles": niveles,
        "meses_portafolio": meses_tabla,
        "gastos_categoria_portafolio": categorias,
        "escenarios_portafolio": _escenarios(resumen, clientes),
    }


def _escenarios(resumen: pd.DataFrame, clientes: pd.DataFrame) -> pd.DataFrame:
    columnas = list(ESQUEMA["escenarios_portafolio"][1])
    if resumen is None or resumen.empty:
        return pd.DataFrame(columns=columnas)
    sub = resumen[resumen["cc_num"].isin(clientes["cc_num"])]
    g = sub.groupby("escenario")
    t = pd.DataFrame(
        {
            "num_clientes_en_riesgo": g["prob_deficit"].apply(
                lambda x: int((x >= UMBRAL_RIESGO_DEFICIT).sum())
            ),
            "porcentaje_en_riesgo": g["prob_deficit"]
            .apply(lambda x: (x >= UMBRAL_RIESGO_DEFICIT).mean())
            .round(4),
            "ahorro_12m_mediano": g["ahorro_m12_p50"].median().round(2),
            "ahorro_36m_mediano": g["ahorro_m36_p50"].median().round(2),
        }
    ).reindex([e for e in ESCENARIOS if e in g.groups])
    t.insert(0, "escenario", [NOMBRES_ESCENARIOS[e] for e in t.index])
    return t.reset_index(drop=True)[columnas]


# ---------------------------------------------------------------------------
# Panorama (calculado en Python)
# ---------------------------------------------------------------------------


def _pct(v) -> str:
    return f"{v * 100:.1f}%"


def panorama(tablas: dict) -> list:
    """Las cifras generales del portafolio, ya calculadas: el modelo las usa
    tal cual en lugar de contarlas él."""
    c = tablas["clientes"]
    n = len(c)
    if not n:
        return []
    niveles = tablas["resumen_niveles"].set_index("nivel_riesgo")
    partes = [
        f"{int(niveles.loc[nv, 'num_clientes']):,} en riesgo {nv.lower()} "
        f"({_pct(niveles.loc[nv, 'porcentaje_clientes'])})"
        for nv in NIVELES
        if nv in niveles.index
    ]
    negativos = int((c["ahorro_mensual"] < 0).sum())
    frases = [
        f"El portafolio tiene {n:,} clientes: " + ", ".join(partes) + ".",
        f"La probabilidad de impago promedio del portafolio es {_pct(c['prob_impago'].mean())}.",
        f"{negativos:,} clientes ({_pct(negativos / n)}) gastan más de lo que ganan "
        "(ahorro mensual negativo).",
    ]
    m = tablas["meses_portafolio"]
    if len(m):
        mayor = m.loc[m["gasto_total"].idxmax()]
        frases.append(
            f"El mes con más gasto fue {mayor['mes_nombre']}, con "
            f"{ia.formatear_df(m.loc[[mayor.name], ['gasto_total']]).iloc[0, 0]}."
        )
    return frases


# ---------------------------------------------------------------------------
# Asistente del portafolio
# ---------------------------------------------------------------------------


class AsistentePortafolio:
    """Mismo flujo que ia.Asistente (clasificar, recuperar con Text-to-SQL y
    redactar), pero sobre las tablas del portafolio."""

    sistema = SISTEMA_RESPUESTA  # lo usa src/ia/evaluar.py

    def __init__(self, modelos: dict, tablas: dict):
        self.modelos = modelos
        self.tablas = tablas
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

    def clasificar(self, pregunta: str, historial: list = None) -> str:
        try:
            plan = ia.leer_json(
                self._clasificar.invoke(
                    {"pregunta": pregunta, "historial": ia.historial_texto(historial)}
                )
            )
        except (ValueError, json.JSONDecodeError):
            plan = {}
        tipo = str(plan.get("tipo", "datos")).strip().lower()
        return "otro" if tipo == "otro" else "datos"

    def consultar(
        self, pregunta: str, historial: list = None, intentos: int = 2
    ) -> ia.Preparacion:
        prep = ia.Preparacion(
            pregunta, "datos", origen="los datos de todos los clientes del portafolio"
        )
        error = ""
        for _ in range(intentos):
            sql = cx.limpiar_sql(
                self._sql.invoke(
                    {
                        "esquema": describir_esquema(),
                        "pregunta": pregunta,
                        "historial": ia.historial_texto(historial),
                        "error": error,
                    }
                )
            )
            prep.intentos.append(sql)
            try:
                prep.datos = cx.ejecutar_sql(self.con, sql, tablas=TABLAS)
                prep.sql = sql
                return prep
            except cx.SQLInvalido as err:
                error = (
                    f"\n\nTu consulta anterior falló:\n{sql}\nError: {err}\n"
                    "Corrígela usando solo las tablas y columnas permitidas."
                )
        prep.datos = self.tablas["resumen_niveles"]
        prep.nota = (
            "No se pudo armar una consulta exacta para esta pregunta; este es el "
            "resumen del portafolio por nivel de riesgo."
        )
        return prep

    def preparar(self, pregunta: str, historial: list = None) -> ia.Preparacion:
        if self.clasificar(pregunta, historial) == "otro":
            return ia.Preparacion(pregunta, "otro")
        prep = self.consultar(pregunta, historial)
        directo = ia.respuesta_directa_sql(
            prep.datos, pregunta, prep.sql, ETIQUETAS, PORCENTAJES
        )
        prep.hechos = [directo] if directo else []
        return prep

    def entradas_respuesta(self, prep: ia.Preparacion, historial: list = None) -> dict:
        return {
            "panorama": "\n".join(f"- {f}" for f in panorama(self.tablas)),
            "origen": prep.origen,
            "datos": ia.formatear_para_modelo(
                prep.datos, etiquetas=ETIQUETAS, porcentajes=PORCENTAJES
            ),
            "nota": f"\nNota: {prep.nota}\n" if prep.nota else "",
            "hechos": "\n".join(f"- {h}" for h in prep.hechos) or "(ninguno)",
            "pregunta": prep.pregunta,
        }

    def redactar(self, prep: ia.Preparacion, historial: list = None):
        if prep.tipo == "otro":
            yield MENSAJE_OTRO
            return
        yield from ia.limpiar_stream(
            self._respuesta.stream(self.entradas_respuesta(prep, historial)),
            ia.pide_consejo(prep.pregunta),
        )

    def responder(self, pregunta: str, historial: list = None) -> tuple:
        prep = self.preparar(pregunta, historial)
        return "".join(self.redactar(prep, historial)), prep
