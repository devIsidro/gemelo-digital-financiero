"""Pruebas de la llave sintética cliente -> préstamo (Fase 4, Opción A)."""

import pytest
from pyspark.sql import Row, SparkSession

from src.spark.asignar_prestamos import asignar_prestamos


@pytest.fixture(scope="module")
def spark():
    sesion = (
        SparkSession.builder.master("local[1]")
        .appName("test_asignar_prestamos")
        .getOrCreate()
    )
    yield sesion
    sesion.stop()


def _datos(spark, num_clientes=5, num_prestamos=100):
    # Gasto: el cliente c0 gasta menos, c4 gasta más.
    perfil = spark.createDataFrame(
        [
            Row(cc_num=f"c{i}", gasto_promedio_mensual=float(1000 * (i + 1)))
            for i in range(num_clientes)
        ]
    )
    # Ingreso: el préstamo L000 es el de menor ingreso, L099 el de mayor.
    loans = spark.createDataFrame(
        [
            Row(loan_id=f"L{j:03d}", income_anual=float(20000 + 1000 * j))
            for j in range(num_prestamos)
        ]
    )
    return perfil, loans


def _asignacion(spark, **kwargs):
    perfil, loans = _datos(spark, **kwargs)
    filas = asignar_prestamos(perfil, loans).collect()
    return {f["cc_num"]: f for f in filas}


def test_quien_mas_gasta_recibe_el_prestamo_de_mayor_ingreso(spark):
    asignacion = _asignacion(spark)
    orden_por_gasto = [asignacion[f"c{i}"]["loan_id"] for i in range(5)]
    assert orden_por_gasto == sorted(orden_por_gasto)
    # Percentiles: 0.1, 0.3, 0.5, 0.7, 0.9 de 100 préstamos -> L010, L030, ...
    assert orden_por_gasto == ["L010", "L030", "L050", "L070", "L090"]


def test_es_reproducible(spark):
    assert _asignacion(spark) == _asignacion(spark)


def test_uno_a_uno_y_todos_asignados(spark):
    """Incluso con casi tantos préstamos como clientes, nadie comparte."""
    asignacion = _asignacion(spark, num_clientes=7, num_prestamos=8)
    loan_ids = [f["loan_id"] for f in asignacion.values()]
    assert len(asignacion) == 7
    assert len(set(loan_ids)) == 7


def test_falla_si_no_alcanzan_los_prestamos(spark):
    perfil, loans = _datos(spark, num_clientes=5, num_prestamos=3)
    with pytest.raises(ValueError, match="no alcanza"):
        asignar_prestamos(perfil, loans)
