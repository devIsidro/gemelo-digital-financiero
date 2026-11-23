"""Pruebas del reporte ejecutivo en PDF (solo pandas y fpdf2)."""

from datetime import datetime

import pandas as pd
import pytest

from src.ml.explicacion import nivel_riesgo
from src.reportes import reporte_ejecutivo as rep

GENERADO = datetime(2026, 11, 23, 18, 5)


def _clientes():
    return pd.DataFrame(
        {
            "cc_num": [30, 10, 20, 40],
            "cliente": ["Cliente 0003", "Cliente 0001", "Cliente 0002", "Cliente 0004"],
            "ingreso_mensual_simulado": [1000.0, 2000.0, 3000.0, 1500.0],
            "gasto_promedio_mensual": [500.0, 2500.0, 600.0, 1400.0],
            "capacidad_ahorro": [0.5, -0.25, 0.8, 0.07],
            "dti": [0.3, 1.5, 0.8, 0.6],
            "prob_impago": [0.05, 0.45, 0.10, 0.35],
        }
    ).assign(nivel=lambda d: nivel_riesgo(d["prob_impago"]))


def _simulacion():
    filas = []
    for cc in [10, 20, 30, 40]:
        for esc, deficit in [
            ("base", 0.0),
            ("credito_hipotecario", 0.5 if cc in (10, 40) else 0.0),
            ("cambio_empleo", 0.2 if cc == 10 else 0.0),
            ("negocio", 0.0),
        ]:
            filas.append(
                {
                    "cc_num": cc,
                    "escenario": esc,
                    "prob_deficit": deficit,
                    "ahorro_m12_p50": 1000.0,
                    "ahorro_m36_p50": 3000.0,
                    "capacidad_ahorro_12m_p50": 0.2,
                    "flujo_m6_p50": 100.0,
                }
            )
    return pd.DataFrame(filas)


def test_datos_del_portafolio_usan_las_cifras_del_dashboard():
    d = rep.armar_datos(_clientes(), resumen_sim=_simulacion(), periodo="jul – dic")
    assert d.resumen["clientes"] == 4
    assert d.resumen["riesgo_alto"] == 2
    assert list(d.alertas["cliente"]) == ["Cliente 0001", "Cliente 0004"]
    assert d.escenarios.loc["credito_hipotecario", "en_riesgo"] == pytest.approx(0.5)
    assert d.titulo_segmento == "Todo el portafolio"


def test_hallazgos_se_calculan_con_los_datos():
    d = rep.armar_datos(_clientes(), resumen_sim=_simulacion())
    texto = " ".join(d.hallazgos)
    assert "2 de 4 clientes (50%) están en riesgo alto" in texto
    assert "1 clientes del portafolio gastan más de lo que ganan" in texto
    assert "pasarían de 0% a 50%" in texto


def test_segmento_filtra_kpis_y_alertas_pero_no_los_niveles():
    d = rep.armar_datos(_clientes(), segmento="Bajo")
    assert d.resumen["clientes"] == 1
    assert d.alertas.empty
    # Los niveles siempre se muestran para todo el portafolio.
    assert int(d.niveles["clientes"].sum()) == 4
    assert d.titulo_segmento == "Segmento de riesgo bajo"
    assert "25% del portafolio" in d.hallazgos[0]


def test_genera_un_pdf_valido_con_y_sin_simulacion():
    for sim in (_simulacion(), None):
        d = rep.armar_datos(_clientes(), resumen_sim=sim, periodo="jul – dic 2020")
        pdf = rep.generar_pdf(d, GENERADO)
        assert pdf.startswith(b"%PDF")
        assert len(pdf) > 1000


def test_segmento_vacio_no_truena():
    vacio = _clientes().iloc[0:0]
    d = rep.armar_datos(vacio)
    assert d.hallazgos == ["El segmento no tiene clientes."]
    assert rep.generar_pdf(d, GENERADO).startswith(b"%PDF")


def test_texto_compatible_con_las_fuentes_del_pdf():
    assert rep.latin("≥ 30% − 5 → «ok» ñ") == ">= 30% - 5 -> «ok» ñ"
    assert rep.latin("emoji 🚀") == "emoji ?"


def test_nombre_del_archivo():
    todos = rep.armar_datos(_clientes())
    alto = rep.armar_datos(_clientes(), segmento="Alto")
    assert (
        rep.nombre_archivo(todos, GENERADO)
        == "reporte_ejecutivo_portafolio_20261123.pdf"
    )
    assert (
        rep.nombre_archivo(alto, GENERADO)
        == "reporte_ejecutivo_riesgo_alto_20261123.pdf"
    )


def test_nunca_muestra_el_numero_de_tarjeta():
    d = rep.armar_datos(_clientes())
    # La tabla de alertas solo lleva el identificador anónimo.
    assert "cc_num" not in d.alertas.columns
    assert d.alertas["cliente"].str.startswith("Cliente ").all()
