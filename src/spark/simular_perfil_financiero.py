"""
Job de PySpark: generación de ingreso e historial crediticio SIMULADOS por
cliente (Fase 4 — Opción A, decisión con Eduardo del 11 sep 2026).

Los datasets de income/credit_history (Loan Default Prediction, Financial
Transactions - Expenses & Income) no comparten un cliente real con el
dataset de transacciones ya ingerido en Silver. Este job construye, de
forma controlada y reproducible (no al azar), un ingreso mensual y un
historial crediticio simulados por cliente (`cc_num`), siguiendo
exactamente las fórmulas documentadas en
`docs/fase4_datos_simulados_ingreso_credito.md`.

IMPORTANTE: estos valores NO representan el ingreso o historial real de
ninguna persona. Es una construcción para poder ejercitar el pipeline
completo (KPIs de riesgo) dentro del "problema de negocio simulado" del
proyecto.

Probado sobre el Silver real completo (924 clientes). Es un puente: cuando
se integre el dataset de Loan Default (decisión del 25 sep 2026), el
ingreso de ese dataset puede reemplazar al simulado aquí.
"""

import sys

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.spark.gold_transactions import build_perfil_cliente

# --- Constantes documentadas (ver docs/fase4_datos_simulados_ingreso_credito.md) ---

# Tramos de ingreso base según tamaño de ciudad (city_pop), en USD/mes.
# Todo el proyecto usa USD, la moneda original del dataset de transacciones
# (clientes de EE.UU.), para que ingreso y gasto se puedan comparar directo.
CIUDAD_PEQUENA_MAX = 50_000
CIUDAD_MEDIANA_MAX = 300_000
INGRESO_BASE_PEQUENA = 12_000
INGRESO_BASE_MEDIANA = 18_000
INGRESO_BASE_GRANDE = 26_000

FACTOR_GASTO = 0.3  # qué tanto pesa el gasto real sobre el ingreso simulado
VARIACION_INGRESO_RANGO = 0.20  # variación pseudoaleatoria ±20%
INGRESO_MINIMO = 8_000
INGRESO_MAXIMO = 120_000

SCORE_MAXIMO = 850
SCORE_MINIMO = 300
FACTOR_PENALIZACION_FRAUDE = 3.0  # puntos de score restados por cada 1% de fraude
BONIFICACION_ANTIGUEDAD_MAX = 40  # puntos ganados por tener muchas transacciones
VARIACION_SCORE_RANGO = 30  # variación pseudoaleatoria ± puntos


def build_spark(app_name: str = "simular_perfil_financiero") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


def _variacion_pseudoaleatoria(
    columna_cc_num: str, sal: str, rango: float
) -> "F.Column":
    """Devuelve un valor determinístico en [-rango, rango], semillado por
    cc_num + una sal distinta por variable (para que ingreso y score no
    varíen de forma idéntica). Mismo cliente -> mismo resultado siempre.
    """
    # pmod (módulo siempre positivo) en vez de abs(hash) % 10_000: abs() del
    # entero más negativo se desborda, y en Spark 4 (modo ANSI, el de la
    # imagen de Docker) eso lanza un error en vez de devolver un valor.
    hash_col = F.hash(F.concat(F.col(columna_cc_num).cast("string"), F.lit(sal)))
    fraccion_0_a_1 = F.pmod(hash_col, F.lit(10_000)) / F.lit(10_000.0)
    return (fraccion_0_a_1 * F.lit(2 * rango)) - F.lit(rango)


def build_perfil_financiero_simulado(silver_df: DataFrame) -> DataFrame:
    """Agrega Silver a nivel de cliente y le asigna un ingreso mensual y un
    historial crediticio simulados, siguiendo las fórmulas documentadas.
    """

    # --- Señales reales por cliente, agregadas desde Silver ---
    # Se reutiliza el mismo perfil de gold_perfil_cliente para que el gasto
    # mensual se calcule en un solo lugar (y no pueda haber dos versiones
    # distintas del mismo número).
    perfil_real = build_perfil_cliente(silver_df).select(
        "cc_num",
        "city_pop",
        "gasto_promedio_mensual",
        "num_transacciones",
        "pct_transacciones_fraude",
    )

    # --- Ingreso mensual simulado ---
    ingreso_base = (
        F.when(F.col("city_pop") < CIUDAD_PEQUENA_MAX, F.lit(INGRESO_BASE_PEQUENA))
        .when(F.col("city_pop") < CIUDAD_MEDIANA_MAX, F.lit(INGRESO_BASE_MEDIANA))
        .otherwise(F.lit(INGRESO_BASE_GRANDE))
    )
    variacion_ingreso = _variacion_pseudoaleatoria(
        "cc_num", "ingreso", VARIACION_INGRESO_RANGO
    )
    ingreso_sin_limite = ingreso_base * (F.lit(1.0) + variacion_ingreso) + (
        F.lit(FACTOR_GASTO) * F.col("gasto_promedio_mensual")
    )
    ingreso_mensual_simulado = F.round(
        F.greatest(
            F.lit(float(INGRESO_MINIMO)),
            F.least(F.lit(float(INGRESO_MAXIMO)), ingreso_sin_limite),
        ),
        2,
    )

    # --- Historial crediticio simulado ---
    penalizacion_fraude = F.col("pct_transacciones_fraude") * F.lit(
        FACTOR_PENALIZACION_FRAUDE
    )
    bonificacion_antiguedad = F.least(
        F.lit(float(BONIFICACION_ANTIGUEDAD_MAX)),
        F.col("num_transacciones") * F.lit(0.5),
    )
    variacion_score = _variacion_pseudoaleatoria(
        "cc_num", "score", float(VARIACION_SCORE_RANGO)
    )
    score_sin_limite = (
        F.lit(float(SCORE_MAXIMO))
        - penalizacion_fraude
        + bonificacion_antiguedad
        + variacion_score
    )
    score_credito_simulado = F.round(
        F.greatest(
            F.lit(float(SCORE_MINIMO)),
            F.least(F.lit(float(SCORE_MAXIMO)), score_sin_limite),
        )
    ).cast("int")

    categoria_credito_simulada = (
        F.when(score_credito_simulado >= 700, F.lit("Bueno"))
        .when(score_credito_simulado >= 550, F.lit("Regular"))
        .otherwise(F.lit("Malo"))
    )

    return (
        perfil_real.withColumn("ingreso_mensual_simulado", ingreso_mensual_simulado)
        .withColumn("score_credito_simulado", score_credito_simulado)
        .withColumn("categoria_credito_simulada", categoria_credito_simulada)
        .select(
            "cc_num",
            "ingreso_mensual_simulado",
            "score_credito_simulado",
            "categoria_credito_simulada",
        )
    )


def run_simulacion_job(
    silver_path: str = "data/silver/transactions",
    salida_path: str = "data/gold/perfil_financiero_simulado",
) -> dict:
    """Corre el job completo: lee Silver, simula ingreso/crédito por
    cliente, escribe el resultado.

    Devuelve un resumen (clientes procesados) para poder loguearlo desde
    el DAG de Airflow.
    """
    spark = build_spark()
    try:
        silver_df = spark.read.parquet(silver_path)

        perfil_df = build_perfil_financiero_simulado(silver_df)
        num_clientes = perfil_df.count()

        # coalesce(1): tabla chica, un solo archivo (ver silver_transactions.py).
        perfil_df.coalesce(1).write.mode("overwrite").parquet(salida_path)

        return {"clientes_procesados": num_clientes}
    finally:
        spark.stop()


if __name__ == "__main__":
    silver_arg = sys.argv[1] if len(sys.argv) > 1 else "data/silver/transactions"
    salida_arg = (
        sys.argv[2] if len(sys.argv) > 2 else "data/gold/perfil_financiero_simulado"
    )

    resumen = run_simulacion_job(silver_arg, salida_arg)
    print(
        f"Simulación completa: {resumen['clientes_procesados']:,} clientes en "
        f"{salida_arg}/ (ingreso e historial crediticio SIMULADOS, Opción A)"
    )
