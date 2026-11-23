"""Pruebas del linaje de datos (sin Streamlit ni Spark)."""

import re
from pathlib import Path

import pandas as pd
import pytest

from src.dashboard import catalogo, linaje

RAIZ = Path(__file__).resolve().parents[2]
TAREAS_DAG = set(
    re.findall(
        r'task_id="([a-z_]+)"',
        (RAIZ / "dags" / "bronze_ingest_dag.py").read_text(encoding="utf-8"),
    )
)


def test_las_tablas_del_linaje_existen_en_el_catalogo():
    en_catalogo = {(t.capa, t.nombre) for t in catalogo.CATALOGO}
    for n in linaje.NODOS.values():
        if n.capa in ("Bronze", "Silver", "Gold"):
            assert (n.capa, n.nombre) in en_catalogo, n


@pytest.mark.parametrize("ind", linaje.INDICADORES, ids=lambda i: i.nombre)
def test_cada_paso_es_del_dag_y_su_codigo_existe(ind):
    for a in ind.aristas:
        assert a.desde in linaje.NODOS and a.hacia in linaje.NODOS
        if a.paso != "—":
            assert set(a.paso.split(" + ")) <= TAREAS_DAG, a.paso
        for archivo in a.codigo.split(" · "):
            assert (RAIZ / archivo).exists(), archivo


@pytest.mark.parametrize("ind", linaje.INDICADORES, ids=lambda i: i.nombre)
def test_cada_indicador_se_sigue_desde_una_fuente_hasta_el_uso(ind):
    hacia = {}
    for a in ind.aristas:
        hacia.setdefault(a.hacia, set()).add(a.desde)
    # De "uso" hacia atrás tiene que llegarse a la tabla Gold y a un CSV.
    alcanzados, pendientes = set(), ["uso"]
    while pendientes:
        n = pendientes.pop()
        for previo in hacia.get(n, ()):
            if previo not in alcanzados:
                alcanzados.add(previo)
                pendientes.append(previo)
    assert ind.tabla in alcanzados
    assert any(linaje.NODOS[n].capa == "Fuente" for n in alcanzados)
    # Y ningún nodo del indicador queda suelto.
    assert set(linaje.nodos_de(ind)) - {"uso"} <= alcanzados


def test_la_columna_existe_en_la_tabla_documentada():
    for ind in linaje.INDICADORES:
        tabla = next(
            t
            for t in catalogo.CATALOGO
            if (t.capa, t.nombre) == ("Gold", linaje.NODOS[ind.tabla].nombre)
        )
        assert catalogo.descripcion(tabla, ind.columna), ind.columna


def test_estado_desde_el_catalogo_y_tabla_de_pasos():
    resumen = pd.DataFrame(
        {
            "capa": ["Silver", "Gold", "Gold"],
            "tabla": ["transactions", "kpis_cliente", "prob_impago_cliente"],
            "filas": [555719, 924, None],
        }
    )
    estado = linaje.estado_desde_catalogo(resumen)
    assert estado == {"silver_tx": 555719, "kpis": 924}

    ind = linaje.indicador("Gasto mensual")
    pasos = linaje.tabla_pasos(ind, estado)
    assert pasos.iloc[0]["De"] == "Fuente · fraudTest.csv"
    assert pasos.iloc[-1]["A"].startswith("Uso")
    assert pasos.loc[pasos["A"] == "Silver · transactions", "Filas en destino"].iloc[0]
    assert list(pasos.columns) == [
        "De",
        "A",
        "Paso del DAG",
        "Qué pasa",
        "Código",
        "Filas en destino",
    ]


def test_grafo_resalta_la_tabla_del_indicador():
    ind = linaje.indicador("Probabilidad de impago")
    dot = linaje.grafo(ind, {"prob": 924})
    assert dot.startswith("digraph linaje {") and dot.rstrip().endswith("}")
    assert '"prob" [label="Gold\\nprob_impago_cliente\\n924 filas"' in dot
    assert "penwidth=2.2" in dot
    assert '"silver_loans" -> "prob"' in dot
