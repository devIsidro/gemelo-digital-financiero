"""Pruebas del modelo de probabilidad de impago (Fase 5)."""

import random

import pytest
from pyspark.ml.functions import vector_to_array
from pyspark.sql import Row, SparkSession

from src.ml.explicacion import explicar
from src.ml.modelo_impago import (
    VARIABLES_CATEGORICAS,
    VARIABLES_NUMERICAS,
    calcular_prob_impago,
    entrenar,
    preparar_variables,
)


@pytest.fixture(scope="module")
def spark():
    sesion = (
        SparkSession.builder.master("local[1]")
        .appName("test_modelo_impago")
        .getOrCreate()
    )
    yield sesion
    sesion.stop()


def _prestamos(spark, n=600):
    """Préstamos de ejemplo donde la tasa de interés alta provoca impago, y
    donde ya viene un DTIRatio de la fuente que el modelo NO debe usar."""
    rng = random.Random(7)
    filas = []
    for i in range(n):
        tasa = rng.uniform(2, 25)
        impago = 1 if rng.random() < (0.02 + (tasa - 2) / 23 * 0.5) else 0
        filas.append(
            Row(
                loan_id=f"L{i}",
                age=rng.randint(18, 69),
                income_anual=float(rng.randint(15_000, 150_000)),
                loan_amount=float(rng.randint(5_000, 250_000)),
                credit_score=rng.randint(300, 849),
                months_employed=rng.randint(0, 119),
                num_credit_lines=rng.randint(1, 4),
                interest_rate=tasa,
                loan_term=rng.choice([12, 24, 36, 48, 60]),
                dti_ratio=rng.uniform(0.1, 0.9),
                education=rng.choice(["High School", "Bachelor's"]),
                employment_type=rng.choice(["Full-time", "Unemployed"]),
                marital_status=rng.choice(["Single", "Married"]),
                has_mortgage=rng.randint(0, 1),
                has_dependents=rng.randint(0, 1),
                loan_purpose=rng.choice(["Home", "Auto"]),
                has_cosigner=rng.randint(0, 1),
                is_default=impago,
            )
        )
    return spark.createDataFrame(filas)


@pytest.fixture(scope="module")
def entrenado(spark):
    loans = _prestamos(spark)
    modelo, metricas = entrenar(loans)
    return loans, modelo, metricas


def test_no_usa_el_ingreso_simulado_ni_el_dti_de_la_fuente():
    """Decisiones con Eduardo: el modelo usa el ingreso de la fuente y el DTI
    calculado por nosotros."""
    variables = set(VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS)
    assert "ingreso_mensual_simulado" not in variables
    assert "dti_ratio" not in variables
    assert {"ingreso_mensual_fuente", "dti"} <= variables


def test_dti_igual_que_el_kpi(spark):
    """36,000 a 36 meses sin interés con ingreso anual de 120,000 -> 0.10,
    el mismo resultado que da ratio_endeudamiento en gold_kpis."""
    df = spark.createDataFrame(
        [
            Row(
                income_anual=120000.0,
                loan_amount=36000.0,
                interest_rate=0.0,
                loan_term=36,
            )
        ]
    )
    fila = preparar_variables(df).collect()[0]
    assert fila["ingreso_mensual_fuente"] == pytest.approx(10000.0)
    assert fila["dti"] == pytest.approx(0.10)


def test_aprende_la_senal_y_reporta_metricas(entrenado):
    _, _, metricas = entrenado
    assert metricas["auc_prueba"] > 0.6
    assert metricas["prestamos_entrenamiento"] + metricas["prestamos_prueba"] == 600
    coef = {c["variable"]: c["coeficiente"] for c in metricas["coeficientes"]}
    assert coef["interest_rate"] > 0  # más tasa -> más riesgo, como en los datos
    assert not any(v.endswith("=__unknown") for v in coef)


def test_es_reproducible(spark):
    loans = _prestamos(spark)
    _, m1 = entrenar(loans)
    _, m2 = entrenar(loans)
    assert m1["auc_prueba"] == m2["auc_prueba"]
    assert m1["coeficientes"] == m2["coeficientes"]


def test_la_explicacion_reproduce_la_probabilidad(spark, entrenado):
    """Con el intercepto, coeficientes, medias y desviaciones del JSON se
    obtiene la misma probabilidad que da Spark (lo usa el reporte de casos)."""
    loans, modelo, metricas = entrenado
    datos = preparar_variables(loans)
    spark_prob = [
        f["p"]
        for f in modelo.transform(datos)
        .select(vector_to_array("probability").getItem(1).alias("p"))
        .collect()
    ]
    prob, aportes = explicar(datos.toPandas(), metricas)
    assert list(prob) == pytest.approx(spark_prob, abs=0.002)
    assert list(aportes.columns) == [c["variable"] for c in metricas["coeficientes"]]


def test_prob_impago_por_cliente(spark, entrenado):
    loans, modelo, _ = entrenado
    asignacion = spark.createDataFrame(
        [Row(cc_num="c1", loan_id="L1"), Row(cc_num="c2", loan_id="L2")]
    )
    filas = {
        f["cc_num"]: f
        for f in calcular_prob_impago(modelo, asignacion, loans).collect()
    }
    assert set(filas) == {"c1", "c2"}
    for f in filas.values():
        assert 0.0 <= f["prob_impago"] <= 1.0
