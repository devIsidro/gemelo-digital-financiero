"""
Datos que el asistente de IA puede consultar sobre UN cliente (Fase 5).

El asistente no lee la capa Gold completa: para cada pregunta se arman
cuatro tablas pequeñas, ya filtradas al cliente elegido y con nombres en
español, y se registran en una base DuckDB en memoria. El modelo de lenguaje
escribe el SQL contra esas tablas (Text-to-SQL), así que:

  - nunca ve ni puede consultar a otros clientes,
  - no puede leer archivos ni modificar nada (solo SELECT, ver validar_sql),
  - el esquema que recibe es corto y fácil de entender para un modelo local.
"""

import re

import duckdb
import pandas as pd

from src.dashboard.datos import CATEGORIAS
from src.dashboard.datos_ejecutivo import TIPO_CONSUMO, factores, nombre_mes
from src.ml.simulador import NOMBRES_ESCENARIOS

# Descripción de cada tabla y columna: es lo que el modelo lee para escribir
# el SQL. Los montos están en USD.
ESQUEMA = {
    "mi_perfil": (
        "Una sola fila con el perfil actual del cliente.",
        {
            "cliente": "identificador anónimo, por ejemplo 'Cliente 0065'",
            "ingreso_mensual": "ingreso mensual simulado (USD)",
            "gasto_mensual_promedio": "gasto promedio al mes con tarjeta (USD)",
            "ahorro_mensual": "ingreso_mensual menos gasto_mensual_promedio (USD)",
            "capacidad_ahorro": "parte del ingreso que ahorra, de 0 a 1 (0.25 = 25%)",
            "prob_impago": "probabilidad de no pagar su préstamo, de 0 a 1",
            "nivel_riesgo": "'Bajo' (< 10%), 'Medio' (10% a 30%) o 'Alto' (>= 30%)",
            "pago_mensual_prestamo": "pago mensual de su préstamo (USD)",
            "ingreso_mensual_prestamo": (
                "ingreso declarado en el préstamo (USD), se usa para el DTI"
            ),
            "dti": "pago mensual del préstamo entre ingreso_mensual_prestamo",
            "monto_prestamo": "monto del préstamo (USD)",
            "tasa_interes_anual": "tasa de interés del préstamo, en % anual",
            "plazo_meses": "plazo del préstamo en meses",
            "score_credito": "score crediticio del préstamo (300 a 850)",
            "categoria_principal": "categoría donde más gasta",
            "num_compras": "número de compras con tarjeta en el histórico",
            "cliente_desde": "fecha de su primera compra",
        },
    ),
    "mis_meses": (
        "Una fila por mes completo del histórico (jul a dic 2020), en orden.",
        {
            "mes": "primer día del mes (DATE)",
            "mes_nombre": "por ejemplo 'nov 2020'",
            "ingreso": "ingreso del mes (USD)",
            "gasto": "gasto con tarjeta en el mes (USD)",
            "ahorro": "ingreso menos gasto (USD)",
            "capacidad_ahorro": "ahorro entre ingreso, de 0 a 1",
            "num_compras": "número de compras en el mes",
        },
    ),
    "mis_gastos_categoria": (
        "Una fila por mes y categoría de gasto (solo meses completos).",
        {
            "mes": "primer día del mes (DATE)",
            "mes_nombre": "por ejemplo 'nov 2020'",
            "categoria": "categoría en español, por ejemplo 'Restaurantes'",
            "tipo": "'Esencial' o 'Discrecional'",
            "gasto": "gasto en esa categoría ese mes (USD)",
            "num_compras": "número de compras",
        },
    ),
    "mis_escenarios": (
        "Una fila por escenario del simulador Monte Carlo (1,000 futuros a 36 meses).",
        {
            "escenario": (
                "'Sin cambios', 'Crédito hipotecario', 'Cambio de empleo' o 'Abrir un negocio'"
            ),
            "prob_numeros_rojos": "probabilidad de que su ahorro baje de cero en 36 meses, 0 a 1",
            "ahorro_12m": "ahorro acumulado típico a 12 meses (USD)",
            "ahorro_12m_bajo": "ahorro a 12 meses en un caso malo (percentil 5, USD)",
            "ahorro_12m_alto": "ahorro a 12 meses en un caso bueno (percentil 95, USD)",
            "ahorro_36m": "ahorro acumulado típico a 36 meses (USD)",
            "flujo_mes_1": "flujo de efectivo típico del primer mes (USD)",
            "capacidad_ahorro_12m": "capacidad de ahorro típica del primer año, 0 a 1",
            "meta_fondo_emergencia": "6 meses de gasto (USD)",
            "meses_fondo_emergencia": "meses típicos para juntar ese fondo (vacío si no llega)",
        },
    ),
    "mis_factores_riesgo": (
        "Los factores que más mueven su probabilidad de impago, del más fuerte al menor.",
        {
            "factor": "explicación en palabras, por ejemplo 'Lleva 3 meses en su empleo'",
            "efecto": "'Sube el riesgo' o 'Baja el riesgo'",
        },
    ),
}

TABLAS = tuple(ESQUEMA)


def describir_esquema() -> str:
    """Esquema en texto para el prompt del modelo."""
    partes = []
    for tabla, (desc, columnas) in ESQUEMA.items():
        cols = "\n".join(f"  - {c}: {d}" for c, d in columnas.items())
        partes.append(f"{tabla}: {desc}\n{cols}")
    return "\n\n".join(partes)


# ---------------------------------------------------------------------------
# Tablas de un cliente
# ---------------------------------------------------------------------------


def _mes(periodo) -> pd.Timestamp:
    return periodo.to_timestamp().normalize()


def tablas_cliente(
    fila: pd.Series,
    tx_cliente: pd.DataFrame,
    meses: list,
    resumen_cliente: pd.DataFrame = None,
    aportes: pd.Series = None,
) -> dict:
    """Las tablas de ESQUEMA para un cliente.

    fila: renglón del cliente en clientes (datos_ejecutivo.cargar_clientes).
    tx_cliente: sus transacciones de Silver. meses: meses completos.
    resumen_cliente: sus 4 filas de simulacion_resumen (o None).
    aportes: aporte de cada variable a su riesgo (o None).
    """
    ingreso = float(fila["ingreso_mensual_simulado"])
    gasto = float(fila["gasto_promedio_mensual"])
    perfil = pd.DataFrame(
        [
            {
                "cliente": fila["cliente"],
                "ingreso_mensual": round(ingreso, 2),
                "gasto_mensual_promedio": round(gasto, 2),
                "ahorro_mensual": round(ingreso - gasto, 2),
                "capacidad_ahorro": round(float(fila["capacidad_ahorro"]), 4),
                "prob_impago": round(float(fila["prob_impago"]), 4),
                "nivel_riesgo": fila["nivel"],
                "pago_mensual_prestamo": round(float(fila["pago_mensual_prestamo"]), 2),
                "ingreso_mensual_prestamo": round(
                    float(fila["ingreso_mensual_fuente"]), 2
                ),
                "dti": round(float(fila["ratio_endeudamiento"]), 4),
                "monto_prestamo": round(float(fila["monto_prestamo_fuente"]), 2),
                "tasa_interes_anual": float(fila["tasa_interes_anual_fuente"]),
                "plazo_meses": int(fila["plazo_meses_fuente"]),
                "score_credito": int(fila["score_credito_fuente"]),
                "categoria_principal": CATEGORIAS.get(
                    fila["categoria_principal"], fila["categoria_principal"]
                ),
                "num_compras": int(fila["num_transacciones"]),
                "cliente_desde": pd.Timestamp(fila["primera_transaccion"]).date(),
            }
        ]
    )

    tx = tx_cliente.assign(
        periodo=tx_cliente["trans_date_trans_time"].dt.to_period("M")
    )
    tx = tx[tx["periodo"].isin(meses)]

    por_mes = (
        tx.groupby("periodo")["amt"]
        .agg(gasto="sum", num_compras="count")
        .reindex(meses, fill_value=0)
    )
    mis_meses = pd.DataFrame(
        {
            "mes": [_mes(p) for p in meses],
            "mes_nombre": [nombre_mes(p) for p in meses],
            "ingreso": round(ingreso, 2),
            "gasto": por_mes["gasto"].round(2).to_numpy(),
            "num_compras": por_mes["num_compras"].astype(int).to_numpy(),
        }
    )
    mis_meses["ahorro"] = (mis_meses["ingreso"] - mis_meses["gasto"]).round(2)
    mis_meses["capacidad_ahorro"] = (
        mis_meses["ahorro"] / ingreso if ingreso > 0 else float("nan")
    ).round(4)
    mis_meses = mis_meses[
        [
            "mes",
            "mes_nombre",
            "ingreso",
            "gasto",
            "ahorro",
            "capacidad_ahorro",
            "num_compras",
        ]
    ]

    cat = (
        tx.groupby(["periodo", "category"])["amt"]
        .agg(gasto="sum", num_compras="count")
        .reset_index()
        .sort_values(["periodo", "gasto"], ascending=[True, False])
    )
    gastos_categoria = pd.DataFrame(
        {
            "mes": [_mes(p) for p in cat["periodo"]],
            "mes_nombre": [nombre_mes(p) for p in cat["periodo"]],
            "categoria": cat["category"]
            .map(CATEGORIAS)
            .fillna(cat["category"])
            .to_numpy(),
            "tipo": cat["category"].map(TIPO_CONSUMO).fillna("Discrecional").to_numpy(),
            "gasto": cat["gasto"].round(2).to_numpy(),
            "num_compras": cat["num_compras"].astype(int).to_numpy(),
        }
    )

    tablas = {
        "mi_perfil": perfil,
        "mis_meses": mis_meses,
        "mis_gastos_categoria": gastos_categoria,
        "mis_escenarios": _escenarios(resumen_cliente),
        "mis_factores_riesgo": _factores(aportes, fila),
    }
    return tablas


def _escenarios(resumen: pd.DataFrame) -> pd.DataFrame:
    columnas = list(ESQUEMA["mis_escenarios"][1])
    if resumen is None or resumen.empty:
        return pd.DataFrame(columns=columnas)
    orden = list(NOMBRES_ESCENARIOS)
    r = resumen.set_index("escenario").reindex(orden).dropna(how="all").reset_index()
    return pd.DataFrame(
        {
            "escenario": r["escenario"].map(NOMBRES_ESCENARIOS),
            "prob_numeros_rojos": r["prob_deficit"].round(4),
            "ahorro_12m": r["ahorro_m12_p50"].round(2),
            "ahorro_12m_bajo": r["ahorro_m12_p5"].round(2),
            "ahorro_12m_alto": r["ahorro_m12_p95"].round(2),
            "ahorro_36m": r["ahorro_m36_p50"].round(2),
            "flujo_mes_1": r["flujo_m1_p50"].round(2),
            "capacidad_ahorro_12m": r["capacidad_ahorro_12m_p50"]
            .astype(float)
            .round(4),
            "meta_fondo_emergencia": r["meta_fondo_emergencia"].round(2),
            "meses_fondo_emergencia": r["meses_meta_p50"],
        }
    )[columnas]


def _factores(aportes: pd.Series, fila: pd.Series) -> pd.DataFrame:
    if aportes is None:
        return pd.DataFrame(columns=["factor", "efecto"])
    return factores(aportes, fila, n=5)[["factor", "efecto"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# SQL: validación y ejecución
# ---------------------------------------------------------------------------

PROHIBIDAS = (
    "insert",
    "update",
    "delete",
    "drop",
    "create",
    "alter",
    "attach",
    "detach",
    "copy",
    "export",
    "import",
    "install",
    "load",
    "pragma",
    "set",
    "call",
    "read_csv",
    "read_parquet",
    "read_json",
    "read_text",
    "glob",
)


class SQLInvalido(ValueError):
    """El SQL que propuso el modelo no se puede ejecutar de forma segura."""


def limpiar_sql(texto: str) -> str:
    """Saca el SQL de la respuesta del modelo (quita ```sql, texto extra y
    el punto y coma final)."""
    m = re.search(r"```(?:sql)?\s*(.*?)```", texto, flags=re.S | re.I)
    sql = m.group(1) if m else texto
    inicio = re.search(r"\b(select|with)\b", sql, flags=re.I)
    if inicio:
        desde = inicio.start()
        sql = sql[desde:]
    return sql.strip().rstrip(";").strip()


def validar_sql(sql: str) -> str:
    """Solo una consulta de lectura sobre las tablas del cliente."""
    s = sql.strip()
    if not s:
        raise SQLInvalido("La consulta está vacía.")
    if ";" in s:
        raise SQLInvalido("Solo se permite una consulta.")
    if not re.match(r"^(select|with)\b", s, flags=re.I):
        raise SQLInvalido("La consulta debe empezar con SELECT.")
    sin_textos = re.sub(r"'[^']*'", "''", s.lower())
    # EXTRACT(MONTH FROM mes) usa FROM sin ser una tabla.
    sin_textos = re.sub(r"\bextract\s*\([^)]*\)", "extract()", sin_textos)
    for palabra in PROHIBIDAS:
        if re.search(rf"\b{palabra}\b", sin_textos):
            raise SQLInvalido(f"No se permite usar {palabra.upper()}.")
    usadas = set(re.findall(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)", sin_textos))
    ctes = set(re.findall(r"\b([a-z_][a-z0-9_]*)\s+as\s*\(", sin_textos))
    desconocidas = usadas - set(TABLAS) - ctes
    if desconocidas:
        raise SQLInvalido(
            "Tablas desconocidas: "
            + ", ".join(sorted(desconocidas))
            + ". Solo existen: "
            + ", ".join(TABLAS)
            + "."
        )
    return s


def conectar(tablas: dict) -> duckdb.DuckDBPyConnection:
    """Base en memoria con las tablas del cliente y sin acceso a archivos."""
    con = duckdb.connect(":memory:")
    for nombre, df in tablas.items():
        con.register(nombre, df)
    con.execute("SET enable_external_access = false")
    con.execute("SET lock_configuration = true")
    return con


def ejecutar_sql(con, sql: str, limite: int = 50) -> pd.DataFrame:
    s = validar_sql(sql)
    try:
        return con.execute(
            f"SELECT * FROM ({s}) AS consulta LIMIT {int(limite)}"
        ).fetch_df()
    except duckdb.Error as err:
        raise SQLInvalido(str(err).splitlines()[0]) from err
