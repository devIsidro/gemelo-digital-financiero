"""
Pruebas del job de simulación de ingreso/crédito (Fase 4 / Opción A).

Verifican, con datos de ejemplo (no datos reales de BBVA ni de Kaggle), que
las fórmulas documentadas en docs/fase4_datos_simulados_ingreso_credito.md
se comportan como se espera: son reproducibles, respetan los límites
definidos, y la categoría de crédito corresponde al score calculado.
"""

import pytest
from pyspark.sql import Row, SparkSession

from src.spark.simular_perfil_financiero import (
    INGRESO_MAXIMO,
    INGRESO_MINIMO,
    SCORE_MAXIMO,
    SCORE_MINIMO,
    build_perfil_financiero_simulado,
)


@pytest.fixture(scope="module")
def spark():
    sesion = (
        SparkSession.builder.master("local[1]")
        .appName("test_simular_perfil_financiero")
        .getOrCreate()
    )
    yield sesion
    sesion.stop()


def _fila_transaccion(cc_num, city_pop, amt, is_fraud, trans_num):
    return Row(
        trans_num=trans_num,
        cc_num=cc_num,
        amt=amt,
        category="misc",
        merchant="tienda",
        trans_date_trans_time="2024-01-01 10:00:00",
        is_fraud=is_fraud,
        city_pop=city_pop,
        year=2024,
        month=1,
        day=1,
    )


def _silver_df_de_ejemplo(spark):
    """Simula 3 clientes con perfiles de gasto/fraude/ciudad distintos:
    - cliente_ciudad_grande_sin_fraude: ciudad grande, gasto alto, 0% fraude.
    - cliente_ciudad_pequena_con_fraude: ciudad pequeña, gasto bajo, 100% fraude.
    - cliente_frecuente: muchas transacciones (antigüedad simulada alta).
    """
    filas = []
    for i in range(10):
        filas.append(
            _fila_transaccion(
                "cliente_ciudad_grande_sin_fraude", 500_000, 1000.0, 0, f"t_a_{i}"
            )
        )
    for i in range(10):
        filas.append(
            _fila_transaccion(
                "cliente_ciudad_pequena_con_fraude", 10_000, 50.0, 1, f"t_b_{i}"
            )
        )
    for i in range(200):
        filas.append(
            _fila_transaccion("cliente_frecuente", 100_000, 20.0, 0, f"t_c_{i}")
        )
    return spark.createDataFrame(filas)


def test_es_reproducible(spark):
    """Mismo cliente, misma corrida -> mismo ingreso y score siempre."""
    silver_df = _silver_df_de_ejemplo(spark)

    resultado_1 = build_perfil_financiero_simulado(silver_df).toPandas()
    resultado_2 = build_perfil_financiero_simulado(silver_df).toPandas()

    resultado_1 = resultado_1.sort_values("cc_num").reset_index(drop=True)
    resultado_2 = resultado_2.sort_values("cc_num").reset_index(drop=True)

    assert resultado_1.equals(resultado_2)


def test_ingreso_y_score_dentro_de_los_limites_documentados(spark):
    silver_df = _silver_df_de_ejemplo(spark)
    resultado = build_perfil_financiero_simulado(silver_df).toPandas()

    assert (resultado["ingreso_mensual_simulado"] >= INGRESO_MINIMO).all()
    assert (resultado["ingreso_mensual_simulado"] <= INGRESO_MAXIMO).all()
    assert (resultado["score_credito_simulado"] >= SCORE_MINIMO).all()
    assert (resultado["score_credito_simulado"] <= SCORE_MAXIMO).all()


def test_categoria_credito_coincide_con_el_score(spark):
    silver_df = _silver_df_de_ejemplo(spark)
    resultado = build_perfil_financiero_simulado(silver_df).toPandas()

    for _, fila in resultado.iterrows():
        score = fila["score_credito_simulado"]
        categoria = fila["categoria_credito_simulada"]
        if score >= 700:
            assert categoria == "Bueno"
        elif score >= 550:
            assert categoria == "Regular"
        else:
            assert categoria == "Malo"


def test_mas_fraude_penaliza_el_score(spark):
    """Un cliente con 100% de transacciones fraudulentas debe terminar con
    un score menor que uno con 0%, incluso con la variación pseudoaleatoria
    (la penalización por fraude, hasta 300 pts, es mucho mayor que el rango
    de variación, ±30 pts).
    """
    silver_df = _silver_df_de_ejemplo(spark)
    resultado = (
        build_perfil_financiero_simulado(silver_df).toPandas().set_index("cc_num")
    )

    score_sin_fraude = resultado.loc[
        "cliente_ciudad_grande_sin_fraude", "score_credito_simulado"
    ]
    score_con_fraude = resultado.loc[
        "cliente_ciudad_pequena_con_fraude", "score_credito_simulado"
    ]

    assert score_con_fraude < score_sin_fraude


def test_ciudad_grande_simula_mayor_ingreso_base(spark):
    """A igualdad de las demás señales, una ciudad grande debe simular un
    ingreso mayor que una ciudad pequeña (el gasto de ambos clientes de
    ejemplo también difiere, pero la diferencia de ingreso base ya es
    grande: 26,000 vs 12,000)."""
    silver_df = _silver_df_de_ejemplo(spark)
    resultado = (
        build_perfil_financiero_simulado(silver_df).toPandas().set_index("cc_num")
    )

    ingreso_ciudad_grande = resultado.loc[
        "cliente_ciudad_grande_sin_fraude", "ingreso_mensual_simulado"
    ]
    ingreso_ciudad_pequena = resultado.loc[
        "cliente_ciudad_pequena_con_fraude", "ingreso_mensual_simulado"
    ]

    assert ingreso_ciudad_grande > ingreso_ciudad_pequena


def test_columnas_esperadas(spark):
    silver_df = _silver_df_de_ejemplo(spark)
    resultado = build_perfil_financiero_simulado(silver_df)

    assert set(resultado.columns) == {
        "cc_num",
        "ingreso_mensual_simulado",
        "score_credito_simulado",
        "categoria_credito_simulada",
    }
    assert resultado.count() == 3
