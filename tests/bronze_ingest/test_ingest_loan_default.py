"""Pruebas de la ingesta de Loan Default a Bronze."""

import pandas as pd
import pytest

from src.bronze_ingest.ingest_loan_default import (
    COLUMNAS_ESPERADAS,
    ingest_loan_default_to_bronze,
)


def _csv_valido(ruta, filas=3):
    datos = {c: [1] * filas for c in COLUMNAS_ESPERADAS}
    datos["LoanID"] = [f"L{i}" for i in range(filas)]
    pd.DataFrame(datos).to_csv(ruta, index=False)


def test_escribe_todas_las_filas(tmp_path):
    csv = tmp_path / "loans.csv"
    _csv_valido(csv, filas=5)
    bronze = tmp_path / "bronze"

    assert ingest_loan_default_to_bronze(str(csv), str(bronze)) == 5
    assert len(pd.read_parquet(bronze)) == 5


def test_reejecutar_no_duplica(tmp_path):
    csv = tmp_path / "loans.csv"
    _csv_valido(csv, filas=5)
    bronze = tmp_path / "bronze"

    ingest_loan_default_to_bronze(str(csv), str(bronze))
    ingest_loan_default_to_bronze(str(csv), str(bronze))
    assert len(pd.read_parquet(bronze)) == 5


def test_falla_con_otro_csv(tmp_path):
    csv = tmp_path / "otro.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="no parece ser el dataset Loan Default"):
        ingest_loan_default_to_bronze(str(csv), str(tmp_path / "bronze"))
