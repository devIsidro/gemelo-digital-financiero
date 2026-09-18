"""
Job de PySpark: limpieza y normalización de la Capa Silver — dataset de
transacciones (Fase 3 / Semana 9-10).

Lee el Parquet crudo de Bronze y aplica:
  - Deduplicación por trans_num (identificador único de la transacción).
  - Reglas de negocio: descarta filas sin llave, con monto <= 0, o con
    fecha de transacción nula.
  - Normalización: castea tipos, recorta/uniforma texto en category y
    merchant, recalcula year/month/day a partir de la fecha real (por si
    el particionado de Bronze quedara desalineado).

Bronze NO se modifica — este job solo lee de ahí y escribe un dataset
nuevo en Silver, particionado igual por year/month/day.
"""

import sys

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def build_spark(app_name: str = "silver_transactions") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


def clean_transactions(bronze_df: DataFrame) -> DataFrame:
    """Aplica limpieza, deduplicación y reglas de negocio sobre Bronze."""

    df = bronze_df

    # --- Reglas de negocio: descartar filas inválidas ---
    df = df.filter(
        F.col("trans_num").isNotNull()
        & F.col("cc_num").isNotNull()
        & F.col("trans_date_trans_time").isNotNull()
        & (F.col("amt") > 0)
    )

    # --- Normalización de tipos y texto ---
    df = df.withColumn("amt", F.col("amt").cast("double"))
    df = df.withColumn("is_fraud", F.col("is_fraud").cast("int"))
    df = df.withColumn("city_pop", F.col("city_pop").cast("long"))
    df = df.withColumn("trans_date_trans_time", F.to_timestamp("trans_date_trans_time"))
    df = df.withColumn("category", F.trim(F.lower(F.col("category"))))
    df = df.withColumn("merchant", F.trim(F.col("merchant")))

    # --- Deduplicación por llave de negocio ---
    df = df.dropDuplicates(["trans_num"])

    # --- Recalcular particiones a partir de la fecha real ---
    df = (
        df.withColumn("year", F.year("trans_date_trans_time"))
        .withColumn("month", F.month("trans_date_trans_time"))
        .withColumn("day", F.dayofmonth("trans_date_trans_time"))
    )

    return df


def run_silver_job(
    bronze_path: str = "data/bronze/transactions",
    silver_path: str = "data/silver/transactions",
) -> dict:
    """Corre el job completo: lee Bronze, limpia, escribe Silver.

    Devuelve un resumen (filas leídas, filas escritas, filas descartadas)
    para poder loguearlo desde el DAG de Airflow.
    """
    spark = build_spark()
    try:
        bronze_df = spark.read.parquet(bronze_path)
        filas_bronze = bronze_df.count()

        silver_df = clean_transactions(bronze_df)
        filas_silver = silver_df.count()

        # coalesce(4): junta el resultado en pocos archivos grandes por
        # partición en vez de cientos de archivos chiquitos (el valor por
        # defecto de Spark). Con muchos archivos chiquitos, el paso final
        # de escritura (mover cada archivo de su carpeta temporal al lugar
        # definitivo) puede fallar de forma intermitente en Docker Desktop
        # sobre Windows con "FileNotFoundException" — visto en la práctica
        # corriendo este job localmente. Menos archivos = muchas menos
        # operaciones de mover archivos = mucho menos probable que falle.
        silver_df.coalesce(4).write.mode("overwrite").partitionBy(
            "year", "month", "day"
        ).parquet(silver_path)

        return {
            "filas_bronze": filas_bronze,
            "filas_silver": filas_silver,
            "filas_descartadas": filas_bronze - filas_silver,
        }
    finally:
        spark.stop()


if __name__ == "__main__":
    bronze_arg = sys.argv[1] if len(sys.argv) > 1 else "data/bronze/transactions"
    silver_arg = sys.argv[2] if len(sys.argv) > 2 else "data/silver/transactions"

    resumen = run_silver_job(bronze_arg, silver_arg)
    print(
        f"Silver completo: {resumen['filas_silver']:,} filas escritas "
        f"({resumen['filas_descartadas']} descartadas de "
        f"{resumen['filas_bronze']:,} en Bronze) en {silver_arg}/"
    )
