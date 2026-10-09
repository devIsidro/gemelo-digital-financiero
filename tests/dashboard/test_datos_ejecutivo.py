"""Pruebas de la capa de datos del Dashboard Ejecutivo (solo pandas)."""

import pandas as pd
import pytest

from src.dashboard import datos_ejecutivo as de
from src.dashboard.datos import CATEGORIAS
from src.ml.explicacion import frase_factor, nivel_riesgo


def _clientes():
    return pd.DataFrame(
        {
            "cc_num": [30, 10, 20],
            "ingreso_mensual_simulado": [1000.0, 2000.0, 3000.0],
            "gasto_promedio_mensual": [500.0, 2500.0, 600.0],
            "capacidad_ahorro": [0.5, -0.25, 0.8],
            "dti": [0.3, 1.5, 0.8],
            "prob_impago": [0.05, 0.45, 0.10],
        }
    ).assign(nivel=lambda d: nivel_riesgo(d["prob_impago"]))


def _tx():
    # Julio completo (31 días) con el cliente 10; junio solo un día.
    julio = pd.date_range("2020-07-01 10:00", periods=31, freq="D")
    return pd.DataFrame(
        {
            "cc_num": [10] * 31 + [20],
            "amt": [10.0] * 31 + [99.0],
            "category": ["grocery_pos"] * 30 + ["travel", "home"],
            "trans_date_trans_time": list(julio) + [pd.Timestamp("2020-06-30 23:00")],
        }
    )


def test_niveles_de_riesgo_y_cortes():
    niveles = nivel_riesgo(pd.Series([0.0, 0.0999, 0.10, 0.2999, 0.30, 0.9]))
    assert list(niveles) == ["Bajo", "Bajo", "Medio", "Medio", "Alto", "Alto"]


def test_ids_anonimos_y_estables():
    ids = de.ids_clientes(pd.Series([30, 10, 20]))
    assert list(ids) == ["Cliente 0003", "Cliente 0001", "Cliente 0002"]


def test_todas_las_categorias_tienen_tipo_de_consumo():
    assert set(CATEGORIAS) == set(de.TIPO_CONSUMO)
    assert set(de.TIPO_CONSUMO.values()) == {"Esencial", "Discrecional"}


def test_resumen_y_tabla_de_niveles():
    c = _clientes()
    r = de.resumen_portafolio(c)
    assert r["clientes"] == 3
    assert r["ahorro_negativo"] == 1
    assert r["dti_mayor_1"] == 1
    assert r["riesgo_alto"] == 1
    t = de.tabla_niveles(c, {"Bajo": 0.05, "Medio": 0.17, "Alto": 0.41})
    assert list(t.index) == ["Bajo", "Medio", "Alto"]
    assert list(t["clientes"]) == [1, 1, 1]
    assert t.loc["Alto", "impago_real"] == 0.41


def test_alertas_solo_riesgo_alto_y_sin_explicacion():
    al = de.alertas(_clientes().assign(cliente=["C3", "C1", "C2"]), None)
    assert list(al["cliente"]) == ["C1"]
    assert list(al["factor"]) == ["—"]


def test_meses_completos_ignora_junio_incompleto():
    assert [str(p) for p in de.meses_completos(_tx())] == ["2020-07"]


def test_ingresos_vs_gastos_promedia_por_cliente():
    ivg = de.ingresos_vs_gastos(_tx(), _clientes())
    assert list(ivg["mes"]) == ["jul 2020"]
    # 310 USD gastados en julio entre 3 clientes (dos no gastaron: cuentan 0).
    assert ivg["gasto"].iloc[0] == pytest.approx(310 / 3)
    assert ivg["ingreso"].iloc[0] == pytest.approx(2000)


def test_mapa_calor_suma_100_por_ciento():
    calor = de.mapa_calor(_tx())
    assert len(calor) == 7 * 24
    assert calor["pct"].sum() == pytest.approx(1.0)
    dia, hora = de.hora_pico(calor)
    assert hora == 10


def test_consumo_por_categoria_y_esencial():
    consumo = de.consumo_por_categoria(_tx())
    assert consumo.iloc[0]["categoria"] == "Supermercado (tienda)"
    # travel (10) es discrecional; grocery (300) y home (99) son esenciales.
    assert de.pct_esencial(consumo) == pytest.approx(399 / 409)


def test_sin_metricas_nuevas_no_se_explica():
    assert not de.se_puede_explicar(None)
    assert not de.se_puede_explicar({"coeficientes": [{"variable": "age"}]})
    assert de.explicar_clientes(_clientes(), {"coeficientes": []}) is None


def test_frases_de_factores():
    fila = pd.Series(
        {"dti": 12.33, "months_employed": 3, "employment_type": "Unemployed"}
    )
    assert frase_factor("dti", fila) == "Su pago mensual es 12.3 veces su ingreso"
    assert frase_factor("months_employed", fila) == "Lleva 3 meses en su empleo"
    assert frase_factor("employment_type=Full-time", fila) == "Está desempleado"
