"""
Job de PySpark: construcción de la Capa Gold — perfil de cliente
(Fase 4 / Semana 13, borrador).

Agrega el dataset de transacciones ya limpio en Silver a nivel de cliente
(cc_num), sin depender de los datasets de ingreso/crédito que todavía no
se han ingerido (ver docs/fase4_diseno_capa_gold.md). Es la primera tabla
Gold del proyecto: alimenta la mitad "gasto" del KPI
flujo_efectivo_proyectado y la tasa_exito_ingesta, y sirve de base para el
modelo predictivo de riesgo en Fase 5.

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
        F.round(F.avg("amt") * F.lit(30), 2).alias("gasto_promedio_mensual_aprox"),
        F.count("trans_num").alias("num_transacciones"),
        F.round(F.avg("is_fraud") * 100, 3).alias("pct_transacciones_fraude"),
        F.max("trans_date_trans_time").alias("ultima_transaccion"),
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

        perfil_df.write.mode("overwrite").parquet(gold_path)

        return {"clientes_procesados": num_clientes}
    finally:
        spark.stop()


if __name__ == "__main__":
    silver_arg = sys.argv[1] if len(sys.argv) > 1 else "data/silver/transactions"
    gold_arg = sys.argv[2] if len(sys.argv) > 2 else "data/gold/perfil_cliente"

    resumen = run_gold_job(silver_arg, gold_arg)
    print(f"Gold completo: {resumen['clientes_procesados']:,} clientes en {gold_arg}/")
