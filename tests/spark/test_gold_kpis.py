"""Pruebas de los KPIs de Gold (Fase 4): ahorro, flujo y endeudamiento (DTI)."""

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


def _cliente(
    cc_num,
    gasto=6000.0,
    ingreso_simulado=10000.0,
    ingreso_anual_fuente=120000.0,
    monto=36000.0,
    tasa=0.0,
    plazo=36,
    score=650,
):
    return dict(
        cc_num=cc_num,
        gasto=gasto,
        ingreso_simulado=ingreso_simulado,
        ingreso_anual_fuente=ingreso_anual_fuente,
        monto=monto,
        tasa=tasa,
        plazo=plazo,
        score=score,
    )


def _kpis(spark, clientes):
    perfil = spark.createDataFrame(
        [Row(cc_num=c["cc_num"], gasto_promedio_mensual=c["gasto"]) for c in clientes]
    )
    simulado = spark.createDataFrame(
        [
            Row(cc_num=c["cc_num"], ingreso_mensual_simulado=c["ingreso_simulado"])
            for c in clientes
        ]
    )
    asignacion = spark.createDataFrame(
        [Row(cc_num=c["cc_num"], loan_id=f"L{c['cc_num']}") for c in clientes]
    )
    loans = spark.createDataFrame(
        [
            Row(
                loan_id=f"L{c['cc_num']}",
                income_anual=c["ingreso_anual_fuente"],
                loan_amount=c["monto"],
                interest_rate=c["tasa"],
                loan_term=c["plazo"],
                credit_score=c["score"],
            )
            for c in clientes
        ]
    )
    prestamo = preparar_prestamo_asignado(asignacion, loans)
    filas = build_kpis_cliente(perfil, simulado, prestamo).collect()
    return {f["cc_num"]: f for f in filas}


def test_capacidad_ahorro_usa_el_ingreso_simulado(spark):
    """Decisión con Eduardo: ahorro y flujo usan el ingreso simulado, aunque
    el ingreso de la fuente sea distinto."""
    k = _kpis(spark, [_cliente("a", gasto=6000, ingreso_simulado=10000)])["a"]
    assert k["capacidad_ahorro"] == pytest.approx(0.4)
    assert k["flujo_efectivo_mensual"] == pytest.approx(4000.0)


def test_gastar_mas_de_lo_que_se_gana_da_ahorro_negativo(spark):
    k = _kpis(spark, [_cliente("b", gasto=12000, ingreso_simulado=10000)])["b"]
    assert k["capacidad_ahorro"] == pytest.approx(-0.2)
    assert k["flujo_efectivo_mensual"] == pytest.approx(-2000.0)


def test_pago_mensual_sin_interes(spark):
    """36,000 a 36 meses sin interés = 1,000 al mes."""
    k = _kpis(spark, [_cliente("c", monto=36000, tasa=0.0, plazo=36)])["c"]
    assert k["pago_mensual_prestamo"] == pytest.approx(1000.0)


def test_pago_mensual_con_interes(spark):
    """Caso de libro: 10,000 al 12% anual a 12 meses = 888.49 al mes."""
    k = _kpis(spark, [_cliente("d", monto=10000, tasa=12.0, plazo=12)])["d"]
    assert k["pago_mensual_prestamo"] == pytest.approx(888.49, abs=0.01)


def test_dti_es_pago_mensual_entre_ingreso_mensual_de_la_fuente(spark):
    """Pago de 1,000 al mes con ingreso anual de 120,000 (10,000 al mes)
    -> DTI 0.10. Usa el ingreso de la fuente, no el simulado."""
    k = _kpis(
        spark,
        [
            _cliente(
                "e",
                ingreso_simulado=50000,
                ingreso_anual_fuente=120000,
                monto=36000,
                tasa=0.0,
                plazo=36,
            )
        ],
    )["e"]
    assert k["ingreso_mensual_fuente"] == pytest.approx(10000.0)
    assert k["ratio_endeudamiento"] == pytest.approx(0.10)


def test_dti_puede_ser_mayor_a_1(spark):
    """Si el pago mensual supera el ingreso mensual, el DTI pasa de 1 y no
    se recorta (se documenta)."""
    k = _kpis(
        spark,
        [_cliente("f", ingreso_anual_fuente=12000, monto=24000, tasa=0.0, plazo=12)],
    )["f"]
    assert k["ratio_endeudamiento"] == pytest.approx(2.0)


def test_score_y_columnas_de_la_fuente(spark):
    k = _kpis(spark, [_cliente("g", score=712)])["g"]
    assert k["score_credito_fuente"] == 712
    assert k["loan_id"] == "Lg"


def test_no_quedan_columnas_descartadas(spark):
    """El score simulado y las versiones alternativas de los KPIs se
    descartaron; no deben aparecer para que nadie las mezcle."""
    k = _kpis(spark, [_cliente("h")])["h"]
    descartadas = {
        "score_credito_simulado",
        "categoria_credito_simulada",
        "capacidad_ahorro_ingreso_prestamo",
        "ratio_endeudamiento_total",
        "ratio_endeudamiento_dti",
    }
    assert descartadas.isdisjoint(k.asDict())


def test_un_renglon_por_cliente(spark):
    k = _kpis(spark, [_cliente(c) for c in ("a", "b", "c")])
    assert set(k) == {"a", "b", "c"}
