"""Pruebas de los KPIs de Gold (Fase 4): ahorro, flujo y endeudamiento."""

import pytest
from pyspark.sql import Row, SparkSession

from src.spark.gold_kpis import build_kpis_cliente, preparar_prestamo_asignado


@pytest.fixture(scope="module")
def spark():
    sesion = (
        SparkSession.builder.master("local[1]").appName("test_gold_kpis").getOrCreate()
    )
    yield sesion
    sesion.stop()


def _kpis(spark, clientes):
    """clientes: lista de (cc_num, gasto_mensual, ingreso_simulado_mensual,
    ingreso_anual_prestamo, monto_prestamo, dti)."""
    perfil = spark.createDataFrame(
        [Row(cc_num=c[0], gasto_promedio_mensual=float(c[1])) for c in clientes]
    )
    simulado = spark.createDataFrame(
        [
            Row(
                cc_num=c[0],
                ingreso_mensual_simulado=float(c[2]),
                score_credito_simulado=800,
                categoria_credito_simulada="Bueno",
            )
            for c in clientes
        ]
    )
    asignacion = spark.createDataFrame(
        [Row(cc_num=c[0], loan_id=f"L{c[0]}") for c in clientes]
    )
    loans = spark.createDataFrame(
        [
            Row(
                loan_id=f"L{c[0]}",
                income_anual=float(c[3]),
                loan_amount=float(c[4]),
                credit_score=600,
                dti_ratio=float(c[5]),
            )
            for c in clientes
        ]
    )
    prestamo = preparar_prestamo_asignado(asignacion, loans)
    filas = build_kpis_cliente(perfil, simulado, prestamo).collect()
    return {f["cc_num"]: f for f in filas}


def test_capacidad_ahorro_sigue_la_formula_del_catalogo(spark):
    kpis = _kpis(spark, [("a", 6000, 10000, 120000, 60000, 0.3)])
    assert kpis["a"]["capacidad_ahorro"] == pytest.approx(0.4)
    assert kpis["a"]["flujo_efectivo_mensual"] == pytest.approx(4000.0)


def test_capacidad_ahorro_con_ingreso_del_prestamo(spark):
    """120,000 al año = 10,000 al mes; gasta 6,000 -> ahorra 40%."""
    kpis = _kpis(spark, [("a", 6000, 50000, 120000, 60000, 0.3)])
    assert kpis["a"]["ingreso_mensual_prestamo"] == pytest.approx(10000.0)
    assert kpis["a"]["capacidad_ahorro_ingreso_prestamo"] == pytest.approx(0.4)


def test_gastar_mas_de_lo_que_se_gana_da_ahorro_negativo(spark):
    kpis = _kpis(spark, [("b", 12000, 10000, 120000, 60000, 0.3)])
    assert kpis["b"]["capacidad_ahorro"] == pytest.approx(-0.2)
    assert kpis["b"]["capacidad_ahorro_ingreso_prestamo"] == pytest.approx(-0.2)
    assert kpis["b"]["flujo_efectivo_mensual"] == pytest.approx(-2000.0)


def test_ratio_endeudamiento_en_sus_dos_formas(spark):
    """Préstamo de 150,000 con ingreso anual de 100,000 -> ratio total 1.5;
    la forma DTI se toma tal cual del dataset."""
    kpis = _kpis(spark, [("c", 1000, 5000, 100000, 150000, 0.42)])
    assert kpis["c"]["ratio_endeudamiento_total"] == pytest.approx(1.5)
    assert kpis["c"]["ratio_endeudamiento_dti"] == pytest.approx(0.42)
    assert kpis["c"]["deuda_total_prestamo"] == pytest.approx(150000.0)
    assert kpis["c"]["loan_id"] == "Lc"


def test_un_renglon_por_cliente(spark):
    datos = [(c, 1, 10, 1200, 100, 0.2) for c in ("a", "b", "c")]
    kpis = _kpis(spark, datos)
    assert set(kpis) == {"a", "b", "c"}
