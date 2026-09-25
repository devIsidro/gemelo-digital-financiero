"""
Job de PySpark: KPIs financieros por cliente en la Capa Gold (Fase 4).

Une, un renglón por cliente (cc_num):
  - gold_perfil_cliente (gold_transactions.py): gasto REAL, sacado de Silver.
  - perfil_financiero_simulado (simular_perfil_financiero.py): ingreso e
    historial crediticio SIMULADOS con fórmulas (Opción A, decisión con
    Eduardo del 11 sep 2026).
  - el préstamo asignado a cada cliente con la llave sintética
    (asignar_prestamos.py + Silver de Loan Default).

y calcula los KPIs de config/kpis.yaml que ya se pueden calcular:

  - capacidad_ahorro = (ingreso_mensual - gasto_mensual) / ingreso_mensual
    Se calcula con las dos fuentes de ingreso, para decidir con Eduardo
    cuál se queda:
      capacidad_ahorro                  -> ingreso simulado con fórmula
      capacidad_ahorro_ingreso_prestamo -> ingreso del préstamo asignado
    Puede ser negativa: el cliente gasta con la tarjeta más de lo que gana.
  - flujo_efectivo_mensual = ingreso_mensual (simulado) - gasto_mensual
    Base del KPI flujo_efectivo_proyectado (falta la parte Monte Carlo).
  - ratio_endeudamiento, en las dos formas posibles (pendiente de decidir
    con Eduardo cuál se usa):
      ratio_endeudamiento_total -> monto del préstamo / ingreso anual
                                   (la fórmula del catálogo de KPIs)
      ratio_endeudamiento_dti   -> DTIRatio del dataset: parte del ingreso
                                   mensual que se va en pagos de deuda

prob_impago todavía NO se calcula: es un modelo que se entrena con las
etiquetas de impago del dataset Loan Default (siguiente fase).

Todos los montos están en USD. El gasto es real; el ingreso, el score y el
préstamo son simulados o asignados, así que estos KPIs NO describen la
situación real de ninguna persona.
"""

import sys

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def build_spark(app_name: str = "gold_kpis") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


def preparar_prestamo_asignado(
    asignacion_df: DataFrame, loans_df: DataFrame
) -> DataFrame:
    """Une la asignación cliente -> préstamo con los datos del préstamo."""
    return asignacion_df.select("cc_num", "loan_id").join(
        loans_df.select(
            "loan_id", "income_anual", "loan_amount", "credit_score", "dti_ratio"
        ),
        on="loan_id",
        how="inner",
    )


def _capacidad_ahorro(ingreso: "F.Column", gasto: "F.Column") -> "F.Column":
    # Los ingresos nunca son 0 (el simulado tiene mínimo de 8,000 USD y
    # Silver de préstamos descarta ingresos <= 0); el when() es una
    # protección adicional contra dividir entre cero.
    return F.when(ingreso > 0, F.round((ingreso - gasto) / ingreso, 4))


def build_kpis_cliente(
    perfil_df: DataFrame, simulado_df: DataFrame, prestamo_df: DataFrame
) -> DataFrame:
    """Une perfil real, perfil simulado y préstamo asignado; calcula KPIs.

    prestamo_df: salida de preparar_prestamo_asignado().
    """

    df = perfil_df.join(simulado_df, on="cc_num", how="inner").join(
        prestamo_df, on="cc_num", how="inner"
    )

    gasto = F.col("gasto_promedio_mensual")
    ingreso_simulado = F.col("ingreso_mensual_simulado")
    ingreso_prestamo = F.col("ingreso_mensual_prestamo")

    df = df.withColumn(
        "ingreso_mensual_prestamo", F.round(F.col("income_anual") / F.lit(12.0), 2)
    )

    return df.select(
        *perfil_df.columns,
        "ingreso_mensual_simulado",
        "score_credito_simulado",
        "categoria_credito_simulada",
        "loan_id",
        "ingreso_mensual_prestamo",
        F.col("credit_score").alias("score_credito_prestamo"),
        F.col("loan_amount").alias("deuda_total_prestamo"),
        F.round(ingreso_simulado - gasto, 2).alias("flujo_efectivo_mensual"),
        _capacidad_ahorro(ingreso_simulado, gasto).alias("capacidad_ahorro"),
        _capacidad_ahorro(ingreso_prestamo, gasto).alias(
            "capacidad_ahorro_ingreso_prestamo"
        ),
        F.round(F.col("loan_amount") / F.col("income_anual"), 4).alias(
            "ratio_endeudamiento_total"
        ),
        F.col("dti_ratio").alias("ratio_endeudamiento_dti"),
    )


def run_kpis_job(
    perfil_path: str = "data/gold/perfil_cliente",
    simulado_path: str = "data/gold/perfil_financiero_simulado",
    salida_path: str = "data/gold/kpis_cliente",
    asignacion_path: str = "data/gold/asignacion_prestamos",
    loans_path: str = "data/silver/loan_default",
) -> dict:
    """Corre el job completo: lee las tablas de entrada, calcula KPIs, escribe.

    Devuelve un resumen para poder loguearlo desde el DAG de Airflow. Si
    alguna tabla trae clientes que otra no tiene, falla a propósito:
    significa que se corrieron sobre versiones distintas de Silver y los
    KPIs no serían confiables.
    """
    spark = build_spark()
    try:
        perfil_df = spark.read.parquet(perfil_path)
        simulado_df = spark.read.parquet(simulado_path)
        prestamo_df = preparar_prestamo_asignado(
            spark.read.parquet(asignacion_path), spark.read.parquet(loans_path)
        )

        clientes_perfil = perfil_df.count()
        clientes_simulado = simulado_df.count()
        clientes_prestamo = prestamo_df.count()

        kpis_df = build_kpis_cliente(perfil_df, simulado_df, prestamo_df)
        clientes_kpis = kpis_df.count()

        if not (
            clientes_perfil == clientes_simulado == clientes_prestamo == clientes_kpis
        ):
            raise ValueError(
                "Las tablas no cuadran: perfil_cliente tiene "
                f"{clientes_perfil} clientes, perfil_financiero_simulado "
                f"{clientes_simulado}, préstamos asignados {clientes_prestamo} "
                f"y la unión {clientes_kpis}. Vuelve a correr el DAG completo."
            )

        # coalesce(1): tabla chica, un solo archivo (ver silver_transactions.py).
        kpis_df.coalesce(1).write.mode("overwrite").parquet(salida_path)

        kpis_df = spark.read.parquet(salida_path)
        resumen = kpis_df.agg(
            F.expr("percentile_approx(capacidad_ahorro, 0.5)").alias("mediana"),
            F.sum(F.when(F.col("capacidad_ahorro") < 0, 1).otherwise(0)).alias(
                "negativos"
            ),
            F.expr("percentile_approx(capacidad_ahorro_ingreso_prestamo, 0.5)").alias(
                "mediana_prestamo"
            ),
            F.sum(
                F.when(F.col("capacidad_ahorro_ingreso_prestamo") < 0, 1).otherwise(0)
            ).alias("negativos_prestamo"),
            F.expr("percentile_approx(ratio_endeudamiento_total, 0.5)").alias(
                "ratio_total"
            ),
            F.expr("percentile_approx(ratio_endeudamiento_dti, 0.5)").alias(
                "ratio_dti"
            ),
        ).collect()[0]

        return {
            "clientes": clientes_kpis,
            "capacidad_ahorro_mediana": resumen["mediana"],
            "clientes_ahorro_negativo": resumen["negativos"],
            "capacidad_ahorro_prestamo_mediana": resumen["mediana_prestamo"],
            "clientes_ahorro_prestamo_negativo": resumen["negativos_prestamo"],
            "ratio_endeudamiento_total_mediana": resumen["ratio_total"],
            "ratio_endeudamiento_dti_mediana": resumen["ratio_dti"],
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
    print(f"KPIs Gold: {r['clientes']:,} clientes en {salida_arg}/")
    print(
        f"  capacidad de ahorro (ingreso simulado): mediana "
        f"{r['capacidad_ahorro_mediana']:.1%}, {r['clientes_ahorro_negativo']} negativos"
    )
    print(
        f"  capacidad de ahorro (ingreso del préstamo): mediana "
        f"{r['capacidad_ahorro_prestamo_mediana']:.1%}, "
        f"{r['clientes_ahorro_prestamo_negativo']} negativos"
    )
    print(
        f"  ratio de endeudamiento: total {r['ratio_endeudamiento_total_mediana']:.2f} "
        f"| DTI {r['ratio_endeudamiento_dti_mediana']:.2f} (medianas)"
    )
