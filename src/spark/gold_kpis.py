"""
Job de PySpark: KPIs financieros por cliente en la Capa Gold (Fase 4).

Une, un renglón por cliente (cc_num):
  - gold_perfil_cliente (gold_transactions.py): gasto REAL, sacado de Silver.
  - perfil_financiero_simulado (simular_perfil_financiero.py): ingreso
    mensual SIMULADO con fórmula (Opción A).
  - el préstamo asignado a cada cliente con la llave sintética
    (asignar_prestamos.py + Silver de Loan Default).

Decisiones con Eduardo (oct 2026), detalle en docs/fase4_decisiones_kpis.md:

  - Dos ingresos, separados a propósito y con nombres distintos:
      ingreso_mensual_simulado -> SOLO para capacidad_ahorro y
                                  flujo_efectivo_mensual.
      ingreso_mensual_fuente   -> ingreso del dataset Loan Default / 12.
                                  Para el DTI y el modelo de impago.
  - score_credito_fuente: el score que trae el préstamo asignado. Viene de la
    fuente; NO lo calcula ningún modelo nuestro.
  - ratio_endeudamiento (DTI), calculado por nosotros, todo mensual:
      pago_mensual_prestamo / ingreso_mensual_fuente
    El DTIRatio que trae el dataset NO se usa: el dataset no documenta cómo
    se calculó y no se puede reproducir con sus propias columnas.

KPIs:
  - capacidad_ahorro = (ingreso_mensual_simulado - gasto) / ingreso_mensual_simulado
    Puede ser negativa: el cliente gasta con la tarjeta más de lo que gana.
  - flujo_efectivo_mensual = ingreso_mensual_simulado - gasto
    Base de flujo_efectivo_proyectado (falta la parte Monte Carlo).
  - ratio_endeudamiento (DTI). Puede ser mayor a 1 (ver documento).

prob_impago todavía NO se calcula: es un modelo que se entrena con las
etiquetas de impago del dataset Loan Default (siguiente fase).

Todos los montos están en USD. El gasto es real; el ingreso simulado y el
préstamo asignado no son de esa persona, así que estos KPIs NO describen la
situación real de nadie.
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
            "loan_id",
            "income_anual",
            "loan_amount",
            "interest_rate",
            "loan_term",
            "credit_score",
        ),
        on="loan_id",
        how="inner",
    )


def pago_mensual(
    monto: "F.Column", tasa_anual_pct: "F.Column", plazo_meses: "F.Column"
) -> "F.Column":
    """Pago mensual fijo de un crédito amortizable (fórmula estándar):

        pago = monto * r / (1 - (1 + r) ** -n)

    con r = tasa anual en % / 100 / 12 (tasa mensual) y n = plazo en meses.
    Si la tasa fuera 0, el pago es monto / n (el dataset no trae tasas de 0,
    pero se protege igual).
    """
    r = tasa_anual_pct / F.lit(100.0) / F.lit(12.0)
    con_interes = monto * r / (F.lit(1.0) - F.pow(F.lit(1.0) + r, -plazo_meses))
    return F.when(r > 0, con_interes).otherwise(monto / plazo_meses)


def build_kpis_cliente(
    perfil_df: DataFrame, simulado_df: DataFrame, prestamo_df: DataFrame
) -> DataFrame:
    """Une perfil real, ingreso simulado y préstamo asignado; calcula KPIs.

    prestamo_df: salida de preparar_prestamo_asignado().
    """

    df = perfil_df.join(simulado_df, on="cc_num", how="inner").join(
        prestamo_df, on="cc_num", how="inner"
    )

    gasto = F.col("gasto_promedio_mensual")
    ingreso_simulado = F.col("ingreso_mensual_simulado")
    ingreso_fuente = F.col("income_anual") / F.lit(12.0)
    pago = pago_mensual(
        F.col("loan_amount"), F.col("interest_rate"), F.col("loan_term")
    )

    # El ingreso simulado nunca es menor a 8,000 USD y Silver de préstamos
    # descarta ingresos <= 0; los when() son protección extra contra
    # dividir entre cero.
    return df.select(
        *perfil_df.columns,
        # --- Ingreso simulado (solo ahorro y flujo) ---
        "ingreso_mensual_simulado",
        # --- Préstamo asignado (datos de la fuente, Loan Default) ---
        "loan_id",
        F.round(ingreso_fuente, 2).alias("ingreso_mensual_fuente"),
        F.col("credit_score").alias("score_credito_fuente"),
        F.col("loan_amount").alias("monto_prestamo_fuente"),
        F.col("interest_rate").alias("tasa_interes_anual_fuente"),
        F.col("loan_term").alias("plazo_meses_fuente"),
        F.round(pago, 2).alias("pago_mensual_prestamo"),
        # --- KPIs ---
        F.round(ingreso_simulado - gasto, 2).alias("flujo_efectivo_mensual"),
        F.when(
            ingreso_simulado > 0,
            F.round((ingreso_simulado - gasto) / ingreso_simulado, 4),
        ).alias("capacidad_ahorro"),
        F.when(ingreso_fuente > 0, F.round(pago / ingreso_fuente, 4)).alias(
            "ratio_endeudamiento"
        ),
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
            F.expr("percentile_approx(ratio_endeudamiento, 0.5)").alias("dti"),
            F.sum(F.when(F.col("ratio_endeudamiento") > 1, 1).otherwise(0)).alias(
                "dti_mayor_a_1"
            ),
        ).collect()[0]

        return {
            "clientes": clientes_kpis,
            "capacidad_ahorro_mediana": resumen["mediana"],
            "clientes_ahorro_negativo": resumen["negativos"],
            "ratio_endeudamiento_mediana": resumen["dti"],
            "clientes_dti_mayor_a_1": resumen["dti_mayor_a_1"],
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
        f"  capacidad de ahorro: mediana {r['capacidad_ahorro_mediana']:.1%}, "
        f"{r['clientes_ahorro_negativo']} negativos"
    )
    print(
        f"  ratio de endeudamiento (DTI): mediana "
        f"{r['ratio_endeudamiento_mediana']:.2f}, "
        f"{r['clientes_dti_mayor_a_1']} clientes con DTI > 1"
    )
