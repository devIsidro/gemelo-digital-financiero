"""Pruebas del catálogo de datos del Panel Operativo."""

from pathlib import Path

import pandas as pd

from src.dashboard import catalogo as cat
from src.dashboard.datos import Rutas


def _tabla(nombre, capa):
    return next(t for t in cat.CATALOGO if t.nombre == nombre and t.capa == capa)


def test_todas_las_tablas_estan_documentadas():
    for t in cat.CATALOGO:
        assert t.capa in cat.CAPAS
        assert t.descripcion and t.origen and t.paso and t.llave
        assert t.columnas
        assert all(cat.descripcion(t, c) for c in t.columnas)


def test_los_datos_personales_no_pasan_a_silver():
    personales = _tabla("transactions", "Bronze").personales
    assert {"first", "last", "street", "dob"} <= personales
    assert not personales & set(_tabla("transactions", "Silver").columnas)


def test_columnas_repetitivas_del_simulador():
    t = _tabla("simulacion_resumen", "Gold")
    assert cat.descripcion(t, "ahorro_m12_p50") == (
        "ahorro acumulado del mes 12, caso típico (mediana) (USD)"
    )
    assert cat.descripcion(t, "flujo_m1_p5").startswith("flujo de efectivo del mes 1")
    assert cat.descripcion(t, "columna_nueva") == ""


def test_estado_se_lee_de_los_archivos(tmp_path: Path):
    rutas = Rutas(tmp_path)
    destino = tmp_path / "gold" / "prob_impago_cliente"
    destino.mkdir(parents=True)
    pd.DataFrame(
        {
            "cc_num": [1, 2],
            "loan_id": ["a", "b"],
            "prob_impago": [0.1, 0.4],
            "extra": [1, 2],
        }
    ).to_parquet(destino / "parte-0.parquet")

    r = cat.resumen(rutas).set_index(["capa", "tabla"])
    fila = r.loc[("Gold", "prob_impago_cliente")]
    assert fila["existe"] and fila["filas"] == 2 and fila["columnas"] == 4
    assert fila["sin_documentar"] == ["extra"]
    assert fila["actualizada"] is not None
    # Lo que todavía no se genera aparece como inexistente, sin romper.
    assert not r.loc[("Gold", "kpis_cliente"), "existe"]

    dicc = cat.columnas(_tabla("prob_impago_cliente", "Gold"), rutas)
    assert list(dicc["columna"]) == ["cc_num", "loan_id", "prob_impago", "extra"]
    assert dicc.iloc[-1]["descripcion"] == "Sin documentar"


def test_tablas_particionadas(tmp_path: Path):
    rutas = Rutas(tmp_path)
    for dia in (1, 2):
        d = (
            tmp_path
            / "silver"
            / "transactions"
            / "year=2020"
            / "month=7"
            / f"day={dia}"
        )
        d.mkdir(parents=True)
        pd.DataFrame({"trans_num": [f"t{dia}"], "amt": [10.0]}).to_parquet(
            d / "p.parquet"
        )
    estado = cat.estado(_tabla("transactions", "Silver"), rutas)
    assert estado["filas"] == 2
    assert {"year", "month", "day"} <= set(estado["columnas"])
