"""
Job de PySpark: Capa Silver del dataset Loan Default Prediction (Fase 4).

Lee el Parquet crudo de Bronze y aplica:
  - Nombres de columna en snake_case, igual que el resto del proyecto
    (Default -> is_default, igual que is_fraud en transacciones).
  - Tipos correctos y columnas Yes/No convertidas a 0/1.
  - Reglas de negocio: descarta préstamos sin ID, con ingreso o monto <= 0,
    score fuera de 300-850, DTIRatio fuera de 0-1 o etiqueta de impago
    distinta de 0/1.
  - Deduplicación por loan_id.

Después de escribir, valida el resultado y falla (lanza error) si algo no
cuadra, para que Airflow marque la tarea en rojo y no se construya Gold con
datos malos. Bronze NO se modifica.

Montos en USD. Income es ANUAL en este dataset.
"""

import sys

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

COLUMNAS_SI_NO = {
    "HasMortgage": "has_mortgage",
    "HasDependents": "has_dependents",
    "HasCoSigner": "has_cosigner",
}


def build_spark(app_name: str = "silver_loans") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


def _si_no_a_entero(columna: str) -> "F.Column":
    valor = F.lower(F.trim(F.col(columna)))
    return F.when(valor == "yes", 1).when(valor == "no", 0)


def clean_loans(bronze_df: DataFrame) -> DataFrame:
    """Renombra, castea, aplica reglas de negocio y deduplica."""

    df = bronze_df.select(
        F.trim(F.col("LoanID")).alias("loan_id"),
        F.col("Age").cast("int").alias("age"),
        F.col("Income").cast("double").alias("income_anual"),
        F.col("LoanAmount").cast("double").alias("loan_amount"),
        F.col("CreditScore").cast("int").alias("credit_score"),
        F.col("MonthsEmployed").cast("int").alias("months_employed"),
        F.col("NumCreditLines").cast("int").alias("num_credit_lines"),
        F.col("InterestRate").cast("double").alias("interest_rate"),
        F.col("LoanTerm").cast("int").alias("loan_term"),
        F.col("DTIRatio").cast("double").alias("dti_ratio"),
        F.trim(F.col("Education")).alias("education"),
        F.trim(F.col("EmploymentType")).alias("employment_type"),
        F.trim(F.col("MaritalStatus")).alias("marital_status"),
        *[_si_no_a_entero(orig).alias(nuevo) for orig, nuevo in COLUMNAS_SI_NO.items()],
        F.trim(F.col("LoanPurpose")).alias("loan_purpose"),
        F.col("Default").cast("int").alias("is_default"),
    )

    # --- Reglas de negocio: descartar préstamos inválidos ---
    df = df.filter(
        F.col("loan_id").isNotNull()
        & (F.col("loan_id") != "")
        & (F.col("income_anual") > 0)
        & (F.col("loan_amount") > 0)
        & F.col("credit_score").between(300, 850)
        & F.col("dti_ratio").between(0, 1)
        & F.col("is_default").isin(0, 1)
        & F.col("has_mortgage").isNotNull()
        & F.col("has_dependents").isNotNull()
        & F.col("has_cosigner").isNotNull()
    )

    # --- Deduplicación por llave de negocio ---
    return df.dropDuplicates(["loan_id"])


def validar_loans(silver_df: DataFrame) -> list:
    """Devuelve la lista de reglas de calidad que NO se cumplen (vacía = OK)."""
    fallas = []
    total = silver_df.count()
    if total == 0:
        return ["Silver de préstamos está vacío"]

    if silver_df.select("loan_id").distinct().count() != total:
        fallas.append("loan_id no es único")

    tasa_impago = silver_df.agg(F.avg("is_default")).collect()[0][0]
    # El dataset original trae ~11.6% de impago. Si sale muy distinto, algo
    # se rompió en la limpieza (ej. se filtraron de más los impagos).
    if not 0.05 <= tasa_impago <= 0.25:
        fallas.append(f"tasa de impago fuera de lo esperado: {tasa_impago:.1%}")

    return fallas


def run_silver_loans_job(
    bronze_path: str = "data/bronze/loan_default",
    silver_path: str = "data/silver/loan_default",
) -> dict:
    """Lee Bronze, limpia, escribe Silver y valida. Lanza error si la
    validación falla."""
    spark = build_spark()
    try:
        bronze_df = spark.read.parquet(bronze_path)
        filas_bronze = bronze_df.count()

        silver_df = clean_loans(bronze_df)
        filas_silver = silver_df.count()

        # coalesce(1): no está particionado y pesa pocos MB; un solo archivo
        # evita el error de escritura de Spark en Docker/Windows.
        silver_df.coalesce(1).write.mode("overwrite").parquet(silver_path)

        fallas = validar_loans(spark.read.parquet(silver_path))
        if fallas:
            raise ValueError(
                f"Validación de Silver préstamos falló: {'; '.join(fallas)}"
            )

        return {
            "filas_bronze": filas_bronze,
            "filas_silver": filas_silver,
            "filas_descartadas": filas_bronze - filas_silver,
        }
    finally:
        spark.stop()


if __name__ == "__main__":
    bronze_arg = sys.argv[1] if len(sys.argv) > 1 else "data/bronze/loan_default"
    silver_arg = sys.argv[2] if len(sys.argv) > 2 else "data/silver/loan_default"

    r = run_silver_loans_job(bronze_arg, silver_arg)
    print(
        f"Silver préstamos: {r['filas_silver']:,} escritos "
        f"({r['filas_descartadas']} descartados de {r['filas_bronze']:,})"
    )
