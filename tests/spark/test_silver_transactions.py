"""Pruebas de las reglas de negocio del job de Silver (Fase 3 / Semana 9)."""

import datetime

import pytest
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from src.spark.silver_transactions import clean_transactions

# Esquema explícito: evita que Spark falle al inferir tipos cuando una
# fila de prueba trae un valor None (ej. trans_num nulo para probar la
# regla de negocio que descarta filas sin llave).
SCHEMA = StructType(
    [
        StructField("trans_num", StringType(), True),
        StructField("cc_num", IntegerType(), True),
        StructField("trans_date_trans_time", TimestampType(), True),
        StructField("amt", DoubleType(), True),
        StructField("category", StringType(), True),
        StructField("merchant", StringType(), True),
        StructField("is_fraud", IntegerType(), True),
        StructField("city_pop", LongType(), True),
        StructField("year", IntegerType(), True),
        StructField("month", IntegerType(), True),
        StructField("day", IntegerType(), True),
    ]
)


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.appName("test_silver_transactions")
        .master("local[1]")
        .getOrCreate()
    )
    yield session
    session.stop()


def _fila(**kwargs):
    base = dict(
        trans_num="ok1",
        cc_num=111,
        trans_date_trans_time=datetime.datetime(2026, 1, 1),
        amt=50.0,
        category="grocery_pos",
        merchant="M1",
        is_fraud=0,
        city_pop=100,
        year=2026,
        month=1,
        day=1,
    )
    base.update(kwargs)
    return Row(**base)


def test_deduplica_por_trans_num(spark):
    df = spark.createDataFrame([_fila(), _fila()], schema=SCHEMA)  # dos filas idénticas
    resultado = clean_transactions(df)
    assert resultado.count() == 1


def test_descarta_monto_invalido(spark):
    df = spark.createDataFrame([_fila(trans_num="malo", amt=-10.0)], schema=SCHEMA)
    resultado = clean_transactions(df)
    assert resultado.count() == 0


def test_descarta_sin_llave(spark):
    df = spark.createDataFrame([_fila(trans_num=None)], schema=SCHEMA)
    resultado = clean_transactions(df)
    assert resultado.count() == 0


def test_normaliza_categoria(spark):
    df = spark.createDataFrame([_fila(category="  Grocery_POS  ")], schema=SCHEMA)
    resultado = clean_transactions(df)
    fila = resultado.collect()[0]
    assert fila["category"] == "grocery_pos"


def test_conserva_fila_valida(spark):
    df = spark.createDataFrame([_fila()], schema=SCHEMA)
    resultado = clean_transactions(df)
    assert resultado.count() == 1
