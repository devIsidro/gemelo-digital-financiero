"""
DAG del pipeline Bronze -> Silver -> Validacion de calidad -> Gold, para el
dataset de transacciones (Credit Card Transactions Fraud Detection).

Pasos encadenados (Fase 2 a Fase 4):
  1. ingest_source     -> lee el CSV crudo y lo escribe en la Capa Bronze.
  2. clean_silver       -> limpia, deduplica y aplica reglas de negocio
                            (PySpark) para producir la Capa Silver.
  3. validate_quality   -> corre la bateria de validaciones de calidad
                            sobre Silver. Si alguna regla falla, la tarea
                            (y por lo tanto el DAG) se marca en rojo y Gold
                            NO se construye con datos malos.
  4. build_gold_perfil  -> perfil por cliente con el gasto real (Gold).
  5. simular_ingreso_credito -> ingreso e historial crediticio SIMULADOS
                            por cliente (Opcion A).
  6. calcular_kpis      -> une 4 y 5 y calcula capacidad_ahorro y
                            flujo_efectivo_mensual por cliente.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.bronze_ingest.ingest_fraud_transactions import ingest_csv_to_bronze
from src.spark.gold_kpis import run_kpis_job
from src.spark.gold_transactions import run_gold_job
from src.spark.silver_transactions import run_silver_job
from src.spark.simular_perfil_financiero import run_simulacion_job
from src.spark.validate_silver import imprimir_reporte, validar_silver

default_args = {
    "owner": "jorge",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

RAW_CSV_PATH = "data/raw/fraudTest.csv"
BRONZE_PATH = "data/bronze/transactions"
SILVER_PATH = "data/silver/transactions"
GOLD_PERFIL_PATH = "data/gold/perfil_cliente"
GOLD_SIMULADO_PATH = "data/gold/perfil_financiero_simulado"
GOLD_KPIS_PATH = "data/gold/kpis_cliente"


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


def build_gold_perfil(**context):
    """Construye gold_perfil_cliente (gasto real por cliente) desde Silver."""
    resumen = run_gold_job(SILVER_PATH, GOLD_PERFIL_PATH)
    print(f"Gold perfil_cliente: {resumen['clientes_procesados']:,} clientes")


def simular_ingreso_credito(**context):
    """Genera el ingreso e historial crediticio SIMULADOS por cliente."""
    resumen = run_simulacion_job(SILVER_PATH, GOLD_SIMULADO_PATH)
    print(
        f"Perfil financiero simulado: {resumen['clientes_procesados']:,} clientes "
        "(valores SIMULADOS, Opcion A)"
    )


def calcular_kpis(**context):
    """Calcula los KPIs financieros por cliente en Gold."""
    r = run_kpis_job(GOLD_PERFIL_PATH, GOLD_SIMULADO_PATH, GOLD_KPIS_PATH)
    print(
        f"KPIs Gold: {r['clientes']:,} clientes, capacidad de ahorro mediana "
        f"{r['capacidad_ahorro_mediana']:.1%}, "
        f"{r['clientes_ahorro_negativo']} con ahorro negativo"
    )


with DAG(
    dag_id="bronze_ingest",
    description="Pipeline Bronze -> Silver -> calidad -> Gold (transacciones)",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2026, 8, 10),
    catchup=False,
    tags=["bronze", "silver", "calidad", "gold"],
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

    gold_perfil_task = PythonOperator(
        task_id="build_gold_perfil",
        python_callable=build_gold_perfil,
    )

    simular_task = PythonOperator(
        task_id="simular_ingreso_credito",
        python_callable=simular_ingreso_credito,
    )

    kpis_task = PythonOperator(
        task_id="calcular_kpis",
        python_callable=calcular_kpis,
    )

    (
        ingest_task
        >> clean_task
        >> validate_task
        >> gold_perfil_task
        >> simular_task
        >> kpis_task
    )
