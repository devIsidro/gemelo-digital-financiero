"""
DAG del pipeline Bronze -> Silver -> Validacion de calidad, para el dataset
de transacciones (Credit Card Transactions Fraud Detection).

Tres pasos encadenados (Fase 2 y Fase 3 / Semana 5-11):
  1. ingest_source     -> lee el CSV crudo y lo escribe en la Capa Bronze.
  2. clean_silver       -> limpia, deduplica y aplica reglas de negocio
                            (PySpark) para producir la Capa Silver.
  3. validate_quality   -> corre la bateria de validaciones de calidad
                            sobre Silver. Si alguna regla falla, la tarea
                            (y por lo tanto el DAG) se marca en rojo.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.bronze_ingest.ingest_fraud_transactions import ingest_csv_to_bronze
from src.spark.silver_transactions import run_silver_job
from src.spark.validate_silver import imprimir_reporte, validar_silver

default_args = {
    "owner": "jorge",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

RAW_CSV_PATH = "data/raw/fraudTest.csv"
BRONZE_PATH = "data/bronze/transactions"
SILVER_PATH = "data/silver/transactions"


def ingest_source(**context):
    """Corre la ingesta real del CSV crudo hacia la Capa Bronze."""
    rows = ingest_csv_to_bronze(RAW_CSV_PATH)
    print(f"Ingesta Bronze completa: {rows:,} filas escritas")


def clean_silver(**context):
    """Corre el job de PySpark que limpia Bronze y produce Silver."""
    resumen = run_silver_job(BRONZE_PATH, SILVER_PATH)
    print(
        f"Silver completo: {resumen['filas_silver']:,} filas escritas "
        f"({resumen['filas_descartadas']} descartadas de "
        f"{resumen['filas_bronze']:,} en Bronze)"
    )


def validate_quality(**context):
    """Corre la bateria de validaciones de calidad sobre Silver.

    Si alguna regla falla, lanza una excepcion para que Airflow marque
    la tarea (y el DAG) como fallido -- asi un problema de calidad de
    datos se ve de inmediato en la interfaz, no queda escondido.
    """
    resultados = validar_silver(SILVER_PATH)
    todo_ok = imprimir_reporte(resultados)
    if not todo_ok:
        fallas = [r.nombre for r in resultados if not r.ok]
        raise ValueError(f"Validacion de calidad fallo en: {', '.join(fallas)}")


with DAG(
    dag_id="bronze_ingest",
    description="Pipeline Bronze -> Silver -> validacion de calidad (transacciones)",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2026, 8, 10),
    catchup=False,
    tags=["bronze", "silver", "calidad"],
) as dag:
    ingest_task = PythonOperator(
        task_id="ingest_source",
        python_callable=ingest_source,
    )

    clean_task = PythonOperator(
        task_id="clean_silver",
        python_callable=clean_silver,
    )

    validate_task = PythonOperator(
        task_id="validate_quality",
        python_callable=validate_quality,
    )

    ingest_task >> clean_task >> validate_task
