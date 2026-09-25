"""Pruebas del perfil de cliente en Gold (Fase 4), sobre todo del gasto mensual."""

import datetime

import pytest
from pyspark.sql import Row, SparkSession

from src.spark.gold_transactions import build_perfil_cliente


@pytest.fixture(scope="module")
def spark():
    sesion = (
        SparkSession.builder.master("local[1]")
        .appName("test_gold_transactions")
        .getOrCreate()
    )
    yield sesion
    sesion.stop()


def _fila(cc_num, fecha, amt, category="misc", is_fraud=0, trans_num=None):
    return Row(
        trans_num=trans_num or f"{cc_num}_{fecha.isoformat()}_{amt}",
        cc_num=cc_num,
        amt=amt,
        category=category,
        merchant="tienda",
        trans_date_trans_time=fecha,
        is_fraud=is_fraud,
        city_pop=1000,
        year=fecha.year,
        month=fecha.month,
        day=fecha.day,
    )


def test_gasto_mensual_usa_el_gasto_real_del_periodo(spark):
    """100 compras de 10 USD entre el 1 de julio y el 31 de agosto (61 días,
    ~2 meses) = ~500 USD al mes. El cálculo viejo (promedio por compra * 30)
    habría dado 300 USD."""
    inicio = datetime.datetime(2020, 7, 1, 12)
    fin = datetime.datetime(2020, 8, 31, 12)
    filas = [_fila("c1", inicio, 10.0, trans_num=f"i{i}") for i in range(50)]
    filas += [_fila("c1", fin, 10.0, trans_num=f"f{i}") for i in range(50)]
    perfil = build_perfil_cliente(spark.createDataFrame(filas)).collect()[0]

    assert perfil["gasto_total"] == pytest.approx(1000.0)
    assert perfil["meses_activo"] == pytest.approx(2.0, abs=0.01)
    assert perfil["gasto_promedio_mensual"] == pytest.approx(500.0, rel=0.01)


def test_cliente_con_un_solo_dia_cuenta_como_un_mes(spark):
    """Con actividad de un solo día no se divide entre ~0 meses."""
    dia = datetime.datetime(2020, 7, 1, 10)
    filas = [_fila("c2", dia, 40.0), _fila("c2", dia, 60.0, trans_num="otra")]
    perfil = build_perfil_cliente(spark.createDataFrame(filas)).collect()[0]

    assert perfil["meses_activo"] == 1.0
    assert perfil["gasto_promedio_mensual"] == pytest.approx(100.0)


def test_categoria_principal_es_la_de_mayor_gasto(spark):
    dia = datetime.datetime(2020, 7, 1)
    filas = [
        _fila("c3", dia, 5.0, category="food", trans_num="a"),
        _fila("c3", dia, 5.0, category="food", trans_num="b"),
        _fila("c3", dia, 50.0, category="travel", trans_num="c"),
    ]
    perfil = build_perfil_cliente(spark.createDataFrame(filas)).collect()[0]

    assert perfil["categoria_principal"] == "travel"
    assert perfil["num_transacciones"] == 3
