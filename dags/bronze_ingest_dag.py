"""
DAG de ingesta hacia la Capa Bronze — dataset de transacciones (Credit Card
Transactions Fraud Detection). Llama a la lógica real en
src/bronze_ingest/ingest_fraud_transactions.py (Fase 2 / Semana 5-6).
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.bronze_ingest.ingest_fraud_transactions import ingest_csv_to_bronze

default_args = {
    "owner": "jorge",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

RAW_CSV_PATH = "data/raw/fraudTest.csv"


def ingest_source(**context):
    """Corre la ingesta real del CSV crudo hacia la Capa Bronze."""
    rows = ingest_csv_to_bronze(RAW_CSV_PATH)
    print(f"Ingesta Bronze completa: {rows:,} filas escritas")


with DAG(
    dag_id="bronze_ingest",
    description="Ingesta de datos crudos hacia la Capa Bronze",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2026, 8, 10),
    catchup=False,
    tags=["bronze", "ingesta"],
) as dag:
    ingest_task = PythonOperator(
        task_id="ingest_source",
        python_callable=ingest_source,
    )
