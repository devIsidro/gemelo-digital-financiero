"""Pruebas de la capa de datos del Panel Operativo (solo pandas, sin Spark ni
Streamlit). Las consultas a Airflow se prueban contra una base SQLite con las
mismas columnas de dag_run y task_instance."""

import sqlite3

import pandas as pd
import pytest

from src.dashboard import datos


def _prestamos(**cambios):
    base = {
        "loan_id": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        "income_anual": [50_000.0] * 10,
        "loan_amount": [10_000.0] * 10,
        "credit_score": [650] * 10,
        "is_default": [0] * 9 + [1],
    }
    base.update(cambios)
    return pd.DataFrame(base)


def _reglas_que_fallan(resultados):
    return [r.nombre for r in resultados if not r.ok]


# ---------------------------------------------------------------------------
# Calidad de préstamos
# ---------------------------------------------------------------------------


def test_validar_prestamos_todo_bien():
    resultados = datos.validar_prestamos(_prestamos())
    assert len(resultados) == 5
    assert _reglas_que_fallan(resultados) == []


def test_validar_prestamos_detecta_duplicados_y_rangos():
    df = _prestamos(
        loan_id=["A", "A", "C", "D", "E", "F", "G", "H", "I", "J"],
        credit_score=[650] * 9 + [900],
        loan_amount=[10_000.0] * 9 + [0.0],
    )
    fallan = _reglas_que_fallan(datos.validar_prestamos(df))
    assert "loan_id no nulo y único" in fallan
    assert "score entre 300 y 850" in fallan
    assert "ingreso y monto > 0" in fallan


def test_validar_prestamos_tasa_de_impago_fuera_de_rango():
    df = _prestamos(is_default=[1] * 5 + [0] * 5)  # 50 % de impago
    assert _reglas_que_fallan(datos.validar_prestamos(df)) == [
        "tasa de impago esperada (5%–25%)"
    ]


# ---------------------------------------------------------------------------
# Volumen
# ---------------------------------------------------------------------------


def test_volumen_por_dia_y_categoria():
    silver = pd.DataFrame(
        {
            "trans_num": ["1", "2", "3"],
            "category": ["gas_transport", "gas_transport", "algo_nuevo"],
            "is_fraud": [0, 1, 0],
            "year": [2020, 2020, 2020],
            "month": [7, 6, 6],
            "day": [1, 30, 30],
        }
    )
    por_dia = datos.volumen_por_dia(silver)
    assert list(por_dia["transacciones"]) == [2, 1]
    assert por_dia["fecha"].iloc[0] == pd.Timestamp("2020-06-30")

    por_cat = datos.volumen_por_categoria(silver)
    assert dict(zip(por_cat["categoria"], por_cat["transacciones"])) == {
        "Gasolina y transporte": 2,
        "Algo nuevo": 1,
    }
    assert datos.pct_fraude(silver) == pytest.approx(100 / 3)


# ---------------------------------------------------------------------------
# Airflow
# ---------------------------------------------------------------------------


@pytest.fixture
def airflow_db(tmp_path):
    ruta = tmp_path / "airflow.db"
    con = sqlite3.connect(ruta)
    con.executescript("""
        CREATE TABLE dag_run (
            id INTEGER PRIMARY KEY, dag_id TEXT, run_id TEXT, run_type TEXT,
            state TEXT, start_date TEXT, end_date TEXT
        );
        CREATE TABLE task_instance (
            dag_id TEXT, run_id TEXT, task_id TEXT, state TEXT,
            duration REAL, try_number INTEGER
        );
        """)
    corridas = [
        (
            "bronze_ingest",
            "r1",
            "scheduled",
            "success",
            "2026-10-01 14:00:00+00:00",
            "2026-10-01 14:06:00+00:00",
        ),
        (
            "bronze_ingest",
            "r2",
            "scheduled",
            "failed",
            "2026-10-02 14:00:00+00:00",
            "2026-10-02 14:02:00+00:00",
        ),
        (
            "bronze_ingest",
            "r3",
            "manual",
            "success",
            "2026-10-03 14:00:00+00:00",
            "2026-10-03 14:04:00+00:00",
        ),
        ("bronze_ingest", "r4", "manual", "running", "2026-10-04 14:00:00+00:00", None),
        (
            "otro_dag",
            "x1",
            "manual",
            "failed",
            "2026-10-04 14:00:00+00:00",
            "2026-10-04 14:01:00+00:00",
        ),
    ]
    con.executemany(
        "INSERT INTO dag_run (dag_id, run_id, run_type, state, start_date, end_date) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        corridas,
    )
    con.executemany(
        "INSERT INTO task_instance VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("bronze_ingest", "r2", "ingest_source", "success", 27.5, 1),
            ("bronze_ingest", "r2", "clean_silver", "failed", 41.0, 4),
            ("bronze_ingest", "r2", "validate_quality", "upstream_failed", None, 0),
        ],
    )
    con.commit()
    con.close()
    return f"sqlite:///{ruta}"


def test_corridas_solo_del_dag_y_en_orden(airflow_db):
    df = datos.corridas(airflow_db)
    assert list(df["run_id"]) == ["r1", "r2", "r3", "r4"]
    assert list(df["tipo"]) == ["Programada", "Programada", "Manual", "Manual"]
    assert df["minutos"].iloc[0] == pytest.approx(6)
    # Hora de Ciudad de México (UTC-6): 14:00 UTC -> 08:00.
    assert df["inicio"].iloc[0].hour == 8


def test_resumen_corridas_cuenta_solo_terminadas(airflow_db):
    res = datos.resumen_corridas(datos.corridas(airflow_db))
    assert res["terminadas"] == 3  # la corrida en curso no cuenta
    assert res["exitosas"] == 2
    assert res["tasa_exito"] == pytest.approx(200 / 3)
    assert res["duracion_tipica"] == pytest.approx(5)  # mediana de 6 y 4 min
    assert res["ultima"]["run_id"] == "r4"


def test_resumen_corridas_sin_terminadas():
    df = pd.DataFrame({"state": ["running"], "minutos": [None]})
    res = datos.resumen_corridas(df)
    assert res["tasa_exito"] is None
    assert res["duracion_tipica"] is None


def test_pasos_de_corrida_rellena_pendientes(airflow_db):
    pasos = datos.pasos_de_corrida(airflow_db, "r2")
    assert list(pasos["task_id"]) == [t for t, _, _ in datos.PASOS]
    estados = dict(zip(pasos["task_id"], pasos["state"]))
    assert estados["clean_silver"] == "failed"
    assert estados["validate_quality"] == "upstream_failed"
    assert estados["modelo_prob_impago"] == "pendiente"
    reintentos = pasos.set_index("task_id").loc["clean_silver", "try_number"]
    assert reintentos == 4


def test_sin_conexion_a_airflow_lanza_error(tmp_path):
    # El panel atrapa este error y sigue funcionando sin la parte de Airflow.
    with pytest.raises(Exception):
        datos.corridas(f"sqlite:///{tmp_path / 'no_existe' / 'airflow.db'}")
