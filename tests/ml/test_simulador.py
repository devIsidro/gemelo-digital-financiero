"""Pruebas del simulador Monte Carlo (solo numpy y pandas)."""

import numpy as np
import pandas as pd
import pytest

from src.ml import simulador as sm

# Supuestos sin incertidumbre: el resultado se puede calcular a mano.
FIJO = sm.Supuestos(
    simulaciones=20,
    inflacion_anual=0.0,
    aumento_salarial_anual=0.0,
    prob_perder_ingreso_mensual=0.0,
    factor_diciembre=1.0,
    cv_minimo=0.0,
)


def test_sin_incertidumbre_da_el_calculo_a_mano():
    sim = sm.simular_cliente(10_000, 4_000, 0.0, "base", FIJO)
    assert np.allclose(sim["flujo"], 6_000)
    assert sim["ahorro"][0, 11] == pytest.approx(72_000)


def test_diciembre_sube_el_gasto_en_los_meses_12_24_y_36():
    sup = sm.Supuestos(**{**FIJO.__dict__, "factor_diciembre": 1.8})
    gasto = sm.simular_cliente(10_000, 4_000, 0.0, "base", sup)["gasto"][0]
    assert gasto[11] == pytest.approx(7_200)
    assert gasto[23] == pytest.approx(7_200)
    assert gasto[0] == pytest.approx(4_000)


def test_es_reproducible_con_la_misma_semilla():
    a = sm.simular_cliente(10_000, 4_000, 0.3, "negocio", semilla=5)
    b = sm.simular_cliente(10_000, 4_000, 0.3, "negocio", semilla=5)
    assert np.array_equal(a["ahorro"], b["ahorro"])


def test_pago_del_credito_hipotecario():
    # 360,000 a 10% anual y 20 años: pago conocido de ≈ 3,474.
    assert sm.pago_mensual(360_000, 0.10, 240) == pytest.approx(3474.08, abs=0.5)
    sim = sm.simular_cliente(10_000, 4_000, 0.0, "credito_hipotecario", FIJO)
    assert sim["info"]["pago_mensual"] == pytest.approx(3474.08, abs=0.5)
    assert np.allclose(sim["flujo"], 6_000 - sim["info"]["pago_mensual"])


def test_cambio_de_empleo_deja_meses_sin_ingreso_desde_el_mes_del_cambio():
    par = sm.ParametrosEscenarios(
        cambio_empleo=sm.CambioEmpleo(
            mes_cambio=3,
            meses_sin_ingreso_promedio=2.0,
            cambio_salario_min=0.1,
            cambio_salario_max=0.1,
        )
    )
    ingreso = sm.simular_cliente(10_000, 4_000, 0.0, "cambio_empleo", FIJO, par)[
        "ingreso"
    ]
    assert (ingreso[:, :2] == 10_000).all()  # antes del cambio, igual
    assert ingreso[:, -1] == pytest.approx(11_000)  # después, 10% más
    assert (ingreso == 0).any()  # alguna trayectoria tuvo meses sin ingreso


def test_negocio_reporta_la_recuperacion_de_la_inversion():
    sim = sm.simular_cliente(10_000, 4_000, 0.2, "negocio")
    r = sm.resumir(sim, 4_000)
    assert 0 <= r["prob_recuperar_inversion"] <= 1
    assert sim["info"]["inversion"] == pytest.approx(60_000)


def test_resumen_y_bandas_ordenados():
    sim = sm.simular_cliente(10_000, 8_000, 0.4, "credito_hipotecario")
    r = sm.resumir(sim, 8_000)
    for h in sm.HORIZONTES:
        assert r[f"ahorro_m{h}_p5"] <= r[f"ahorro_m{h}_p50"] <= r[f"ahorro_m{h}_p95"]
    assert 0 <= r["prob_deficit_12m"] <= r["prob_deficit"] <= 1
    b = sm.bandas(sim)
    assert len(b) == 36
    assert (b["p5"] <= b["p25"]).all() and (b["p75"] <= b["p95"]).all()


def test_escenario_desconocido_falla():
    with pytest.raises(ValueError):
        sm.simular_cliente(10_000, 4_000, 0.2, "loteria")


def _tx():
    filas = []
    for cliente, montos in [
        (1, [100, 120, 80, 100, 100, 180]),
        (2, [50, 50, 50, 50, 50, 90]),
    ]:
        for mes, monto in zip(range(7, 13), montos):
            dias = pd.Period(f"2020-{mes:02d}").days_in_month
            for d in range(1, dias + 1):  # un día con monto y el resto en cero
                filas.append(
                    (cliente, monto if d == 1 else 0.0, pd.Timestamp(2020, mes, d))
                )
    return pd.DataFrame(filas, columns=["cc_num", "amt", "trans_date_trans_time"])


def test_insumos_salen_del_gasto_real_sin_diciembre():
    kpis = pd.DataFrame(
        {
            "cc_num": [1, 2, 3],
            "ingreso_mensual_simulado": [1_000.0, 900.0, 800.0],
            "gasto_promedio_mensual": [110.0, 55.0, 70.0],
        }
    )
    ins, factor = sm.insumos_clientes(_tx(), kpis)
    uno = ins.set_index("cc_num")
    assert uno.loc[1, "gasto_base"] == pytest.approx(100)  # julio a noviembre
    assert uno.loc[2, "cv_gasto"] == pytest.approx(
        sm.Supuestos().cv_minimo
    )  # sin variación
    assert uno.loc[3, "gasto_base"] == pytest.approx(70)  # sin datos: usa Gold
    assert factor == pytest.approx((180 + 90) / 150, abs=0.001)


@pytest.mark.parametrize(
    "escenario,parametros",
    [
        ("base", sm.PARAMETROS_ESCENARIOS),
        # Extremos de los controles del dashboard.
        *[
            ("credito_hipotecario", sm.ParametrosEscenarios(credito_hipotecario=c))
            for c in (
                sm.CreditoHipotecario(1.0, 0.05, 60),
                sm.CreditoHipotecario(5.0, 0.16, 360),
                sm.CreditoHipotecario(3.0, 0.0, 240),
            )
        ],
        *[
            ("cambio_empleo", sm.ParametrosEscenarios(cambio_empleo=c))
            for c in (
                sm.CambioEmpleo(1, 0.0, -0.30, -0.30),
                sm.CambioEmpleo(24, 4.0, -0.30, 0.50),
            )
        ],
        *[
            ("negocio", sm.ParametrosEscenarios(negocio=c))
            for c in (
                sm.Negocio(inversion_meses_ingreso=24, margen_medio=-0.10),
                sm.Negocio(inversion_meses_ingreso=1, prob_cierre_24m=0.0),
            )
        ],
    ],
)
@pytest.mark.parametrize(
    "ingreso,gasto,cv",
    [(10_000, 4_000, 0.3), (5_000, 9_000, 0.8), (0.0, 4_000, 0.3), (10_000, 0.0, 0.05)],
)
def test_casos_extremos_dan_numeros_validos(escenario, parametros, ingreso, gasto, cv):
    sup = sm.Supuestos(simulaciones=200)
    sim = sm.simular_cliente(ingreso, gasto, cv, escenario, sup, parametros)
    assert all(np.isfinite(sim[k]).all() for k in ("ingreso", "gasto", "ahorro"))
    r = sm.resumir(sim, gasto, sup)
    assert all(v is None or np.isfinite(v) for v in r.values())
    assert 0 <= r["prob_deficit"] <= 1
