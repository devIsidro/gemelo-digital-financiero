"""Pruebas de la Capa Silver de préstamos (Loan Default, Fase 4)."""

import pytest
from pyspark.sql import Row, SparkSession

from src.spark.silver_loans import clean_loans, validar_loans


@pytest.fixture(scope="module")
def spark():
    sesion = (
        SparkSession.builder.master("local[1]")
        .appName("test_silver_loans")
        .getOrCreate()
    )
    yield sesion
    sesion.stop()


def _prestamo(**cambios):
    base = dict(
        LoanID="L1",
        Age=40,
        Income=60000,
        LoanAmount=100000,
        CreditScore=650,
        MonthsEmployed=24,
        NumCreditLines=2,
        InterestRate=10.5,
        LoanTerm=36,
        DTIRatio=0.4,
        Education="Bachelor's",
        EmploymentType="Full-time",
        MaritalStatus="Single",
        HasMortgage="Yes",
        HasDependents="No",
        LoanPurpose="Home",
        HasCoSigner="No",
        Default=0,
    )
    base.update(cambios)
    return Row(**base)


def _limpiar(spark, filas):
    return clean_loans(spark.createDataFrame(filas))


def test_renombra_y_convierte_si_no(spark):
    fila = _limpiar(spark, [_prestamo()]).collect()[0]
    assert fila["loan_id"] == "L1"
    assert fila["income_anual"] == 60000.0
    assert fila["is_default"] == 0
    assert (fila["has_mortgage"], fila["has_dependents"], fila["has_cosigner"]) == (
        1,
        0,
        0,
    )


@pytest.mark.parametrize(
    "cambio",
    [
        {"Income": 0},
        {"LoanAmount": -5},
        {"CreditScore": 900},
        {"DTIRatio": 1.5},
        {"Default": 2},
        {"HasCoSigner": "Tal vez"},
    ],
)
def test_descarta_prestamos_invalidos(spark, cambio):
    assert _limpiar(spark, [_prestamo(**cambio)]).count() == 0


def test_deduplica_por_loan_id(spark):
    assert _limpiar(spark, [_prestamo(), _prestamo()]).count() == 1


def test_validacion_detecta_tasa_de_impago_rara(spark):
    """Si todos los préstamos salen en impago, algo se rompió."""
    filas = [_prestamo(LoanID=f"L{i}", Default=1) for i in range(10)]
    fallas = validar_loans(_limpiar(spark, filas))
    assert any("tasa de impago" in f for f in fallas)


def test_validacion_pasa_con_datos_normales(spark):
    filas = [_prestamo(LoanID=f"L{i}", Default=1 if i < 1 else 0) for i in range(10)]
    assert validar_loans(_limpiar(spark, filas)) == []
