"""
Job de PySpark: KPIs financieros por cliente en la Capa Gold (Fase 4).

Une las dos tablas Gold que ya existen, un renglón por cliente (cc_num):
  - gold_perfil_cliente (gold_transactions.py): gasto REAL, sacado de Silver.
  - perfil_financiero_simulado (simular_perfil_financiero.py): ingreso e
    historial crediticio SIMULADOS (Opción A, decisión con Eduardo del
    11 sep 2026).

y calcula los KPIs de config/kpis.yaml que ya se pueden calcular con eso:

  - capacidad_ahorro = (ingreso_mensual - gasto_mensual) / ingreso_mensual
    Puede ser negativa: quiere decir que el cliente gasta con la tarjeta más
    de lo que gana en el mes (una señal de riesgo).
  - flujo_efectivo_mensual = ingreso_mensual - gasto_mensual
    Es la base del KPI flujo_efectivo_proyectado; las bandas de confianza
    del simulador Monte Carlo se agregan en una fase posterior.

Todavía NO se calculan ratio_endeudamiento ni prob_impago: necesitan deuda
y etiquetas reales de impago, que vendrán del dataset Loan Default de
Kaggle (decisión del 25 sep 2026).

Todos los montos están en USD. El gasto es la parte real; el ingreso es
simulado, así que estos KPIs NO describen la situación real de ninguna
persona.
"""

import sys

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def build_spark(app_name: str = "gold_kpis") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


def build_kpis_cliente(perfil_df: DataFrame, simulado_df: DataFrame) -> DataFrame:
    """Une el perfil real con el perfil simulado y calcula los KPIs."""

    df = perfil_df.join(simulado_df, on="cc_num", how="inner")

    ingreso = F.col("ingreso_mensual_simulado")
    gasto = F.col("gasto_promedio_mensual")

    # El ingreso simulado nunca es menor a INGRESO_MINIMO (8,000 USD), así que
    # la división es segura; el when() es solo una protección adicional.
    capacidad_ahorro = F.when(ingreso > 0, F.round((ingreso - gasto) / ingreso, 4))

    return df.withColumn(
        "flujo_efectivo_mensual", F.round(ingreso - gasto, 2)
    ).withColumn("capacidad_ahorro", capacidad_ahorro)


def run_kpis_job(
    perfil_path: str = "data/gold/perfil_cliente",
    simulado_path: str = "data/gold/perfil_financiero_simulado",
    salida_path: str = "data/gold/kpis_cliente",
) -> dict:
    """Corre el job completo: lee las dos tablas Gold, calcula KPIs, escribe.

    Devuelve un resumen para poder loguearlo desde el DAG de Airflow. Si
    alguna de las dos tablas trae clientes que la otra no tiene, falla a
    propósito: significa que se corrieron sobre versiones distintas de
    Silver y los KPIs no serían confiables.
    """
    spark = build_spark()
    try:
        perfil_df = spark.read.parquet(perfil_path)
        simulado_df = spark.read.parquet(simulado_path)

        clientes_perfil = perfil_df.count()
        clientes_simulado = simulado_df.count()

        kpis_df = build_kpis_cliente(perfil_df, simulado_df)
        clientes_kpis = kpis_df.count()

        if not (clientes_perfil == clientes_simulado == clientes_kpis):
            raise ValueError(
                "Las tablas Gold no cuadran: perfil_cliente tiene "
                f"{clientes_perfil} clientes, perfil_financiero_simulado "
                f"{clientes_simulado} y la unión {clientes_kpis}. Vuelve a "
                "correr ambos jobs sobre el mismo Silver."
            )

        # coalesce(1): tabla chica, un solo archivo (ver silver_transactions.py).
        kpis_df.coalesce(1).write.mode("overwrite").parquet(salida_path)

        resumen = kpis_df.agg(
            F.expr("percentile_approx(capacidad_ahorro, 0.5)").alias("mediana"),
            F.sum(F.when(F.col("capacidad_ahorro") < 0, 1).otherwise(0)).alias(
                "negativos"
            ),
        ).collect()[0]

        return {
            "clientes": clientes_kpis,
            "capacidad_ahorro_mediana": resumen["mediana"],
            "clientes_ahorro_negativo": resumen["negativos"],
        }
    finally:
        spark.stop()


if __name__ == "__main__":
    perfil_arg = sys.argv[1] if len(sys.argv) > 1 else "data/gold/perfil_cliente"
    simulado_arg = (
        sys.argv[2] if len(sys.argv) > 2 else "data/gold/perfil_financiero_simulado"
    )
    salida_arg = sys.argv[3] if len(sys.argv) > 3 else "data/gold/kpis_cliente"

    r = run_kpis_job(perfil_arg, simulado_arg, salida_arg)
    print(
        f"KPIs Gold: {r['clientes']:,} clientes en {salida_arg}/ — "
        f"capacidad de ahorro mediana {r['capacidad_ahorro_mediana']:.1%}, "
        f"{r['clientes_ahorro_negativo']} clientes con ahorro negativo "
        "(ingreso SIMULADO, Opción A)"
    )
