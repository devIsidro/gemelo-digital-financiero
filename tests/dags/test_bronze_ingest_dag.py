"""Prueba mínima: valida que el DAG cargue sin errores de sintaxis/importación."""

from airflow.models import DagBag


def test_bronze_ingest_dag_loads():
    dagbag = DagBag(dag_folder="dags", include_examples=False)
    assert dagbag.import_errors == {}
    assert "bronze_ingest" in dagbag.dags
