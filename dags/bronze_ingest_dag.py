"""
DAG de ejemplo — plantilla de arranque para la ingesta hacia la Capa Bronze.

Esto es un ESQUELETO (Fase 1 / Semana 1: repositorio creado), no la implementación
final de la Fase 2. Reemplaza la función `ingest_source` con la lógica real de
lectura del dataset elegido y escritura hacia MinIO/Delta Lake.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "jorge",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}


def ingest_source(**context):
    """Placeholder: aquí va la lectura del dataset crudo y su escritura en Bronze."""
    # TODO (Fase 2 / Semana 5-6): leer CSV/API fuente, escribir a MinIO en formato
    # Delta/Parquet, particionado por year/month/day, preservando el dato crudo.
    print("Ingesta Bronze: pendiente de implementar")


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
