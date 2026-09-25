"""
Job de PySpark: construcción de la Capa Gold — perfil de cliente
(Fase 4 / Semana 13).

Agrega el dataset de transacciones ya limpio en Silver a nivel de cliente
(cc_num), sin depender de los datasets de ingreso/crédito que todavía no
se han ingerido (ver docs/fase4_diseno_capa_gold.md). Es la primera tabla
Gold del proyecto: es la mitad "gasto" de los KPIs capacidad_ahorro y
flujo_efectivo_proyectado, y sirve de base para la simulación de ingreso
(simular_perfil_financiero.py) y el modelo de riesgo en Fase 5.

Todos los montos están en USD, la moneda original del dataset de Kaggle.

Silver NO se modifica — este job solo lee de ahí y escribe un dataset
nuevo en Gold.
"""

import sys

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def build_spark(app_name: str = "gold_transactions") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


def build_perfil_cliente(silver_df: DataFrame) -> DataFrame:
    """Agrega Silver a nivel de cliente (cc_num) para la tabla
    gold_perfil_cliente.
    """

    df = silver_df

    # --- Categoría principal por cliente: la de mayor gasto acumulado ---
    gasto_por_categoria = df.groupBy("cc_num", "category").agg(
        F.sum("amt").alias("gasto_categoria")
    )
    ventana_categoria = Window.partitionBy("cc_num").orderBy(F.desc("gasto_categoria"))
    categoria_principal = (
        gasto_por_categoria.withColumn("rank", F.row_number().over(ventana_categoria))
        .filter(F.col("rank") == 1)
        .select("cc_num", F.col("category").alias("categoria_principal"))
    )

    # --- Agregados generales por cliente ---
    perfil = df.groupBy("cc_num").agg(
        F.sum("amt").alias("gasto_total"),
        F.count("trans_num").alias("num_transacciones"),
        F.round(F.avg("is_fraud") * 100, 3).alias("pct_transacciones_fraude"),
        F.min("trans_date_trans_time").alias("primera_transaccion"),
        F.max("trans_date_trans_time").alias("ultima_transaccion"),
        # city_pop es fijo por cliente (una dirección por tarjeta en el
        # dataset); se conserva porque la simulación de ingreso lo usa.
        F.first("city_pop").alias("city_pop"),
    )

    # --- Gasto promedio mensual (montos en USD, moneda original del dataset) ---
    # Meses de actividad = días entre la primera y la última transacción del
    # cliente / 30.44 (días promedio de un mes), con mínimo de 1 mes para
    # clientes con muy poca actividad. Antes se usaba avg(amt) * 30, que
    # suponía 1 compra al día y subestimaba el gasto real ~3 veces (los
    # clientes reales hacen ~100 compras al mes).
    perfil = perfil.withColumn(
        "meses_activo",
        F.round(
            F.greatest(
                F.lit(1.0),
                F.datediff("ultima_transaccion", "primera_transaccion") / F.lit(30.44),
            ),
            2,
        ),
    ).withColumn(
        "gasto_promedio_mensual",
        F.round(F.col("gasto_total") / F.col("meses_activo"), 2),
    )

    return perfil.join(categoria_principal, on="cc_num", how="left")


def run_gold_job(
    silver_path: str = "data/silver/transactions",
    gold_path: str = "data/gold/perfil_cliente",
) -> dict:
    """Corre el job completo: lee Silver, agrega por cliente, escribe Gold.

    Devuelve un resumen (clientes procesados) para poder loguearlo desde
    el DAG de Airflow.
    """
    spark = build_spark()
    try:
        silver_df = spark.read.parquet(silver_path)

        perfil_df = build_perfil_cliente(silver_df)
        num_clientes = perfil_df.count()

        # coalesce(1): es una tabla chica (un renglón por cliente); un solo
        # archivo evita el error de escritura de Spark en Docker/Windows
        # (ver comentario en silver_transactions.py).
        perfil_df.coalesce(1).write.mode("overwrite").parquet(gold_path)

        return {"clientes_procesados": num_clientes}
    finally:
        spark.stop()


if __name__ == "__main__":
    silver_arg = sys.argv[1] if len(sys.argv) > 1 else "data/silver/transactions"
    gold_arg = sys.argv[2] if len(sys.argv) > 2 else "data/gold/perfil_cliente"

    resumen = run_gold_job(silver_arg, gold_arg)
    print(f"Gold completo: {resumen['clientes_procesados']:,} clientes en {gold_arg}/")
