"""Pruebas de los KPIs de Gold (Fase 4): capacidad de ahorro y flujo mensual."""

import pytest
from pyspark.sql import Row, SparkSession

from src.spark.gold_kpis import build_kpis_cliente


@pytest.fixture(scope="module")
def spark():
    sesion = (
        SparkSession.builder.master("local[1]").appName("test_gold_kpis").getOrCreate()
    )
    yield sesion
    sesion.stop()


def _kpis(spark, clientes):
    """clientes: lista de (cc_num, gasto_mensual, ingreso_mensual)."""
    perfil = spark.createDataFrame(
        [Row(cc_num=c, gasto_promedio_mensual=float(g)) for c, g, _ in clientes]
    )
    simulado = spark.createDataFrame(
        [Row(cc_num=c, ingreso_mensual_simulado=float(i)) for c, _, i in clientes]
    )
    filas = build_kpis_cliente(perfil, simulado).collect()
    return {f["cc_num"]: f for f in filas}


def test_capacidad_ahorro_sigue_la_formula_del_catalogo(spark):
    kpis = _kpis(spark, [("a", 6000, 10000)])
    assert kpis["a"]["capacidad_ahorro"] == pytest.approx(0.4)
    assert kpis["a"]["flujo_efectivo_mensual"] == pytest.approx(4000.0)


def test_gastar_mas_de_lo_que_se_gana_da_ahorro_negativo(spark):
    kpis = _kpis(spark, [("b", 12000, 10000)])
    assert kpis["b"]["capacidad_ahorro"] == pytest.approx(-0.2)
    assert kpis["b"]["flujo_efectivo_mensual"] == pytest.approx(-2000.0)


def test_un_renglon_por_cliente(spark):
    kpis = _kpis(spark, [("a", 1, 10), ("b", 2, 10), ("c", 3, 10)])
    assert set(kpis) == {"a", "b", "c"}
