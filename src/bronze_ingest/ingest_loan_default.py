"""
Ingesta (Fase 4) del dataset "Loan Default Prediction" (Kaggle,
nikhil1e9/loan-default) hacia la Capa Bronze.

Mismo principio que ingest_fraud_transactions.py: se guarda el dato crudo
tal como viene, sin limpiar ni validar reglas de negocio (eso es trabajo de
Silver). A diferencia de las transacciones, este dataset no tiene fechas,
así que no se particiona: se escribe como un solo archivo Parquet.

Este dataset NO comparte clientes con el de transacciones. Cada préstamo se
asigna a un cliente con una llave sintética documentada (Opción A), ver
docs/fase4_llave_sintetica_prestamos.md.
"""

import shutil
import sys
from pathlib import Path

import pandas as pd

COLUMNAS_ESPERADAS = [
    "LoanID",
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio",
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner",
    "Default",
]


def ingest_loan_default_to_bronze(
    csv_path: str, bronze_root: str = "data/bronze/loan_default"
) -> int:
    """Lee el CSV crudo y lo escribe como Parquet en Bronze.

    Devuelve el número de filas escritas. Falla si el CSV no trae las
    columnas esperadas (por ejemplo, si se descargó otro dataset por error).
    """
    df = pd.read_csv(csv_path)

    faltantes = [c for c in COLUMNAS_ESPERADAS if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"El CSV {csv_path} no parece ser el dataset Loan Default: "
            f"le faltan las columnas {faltantes}"
        )

    out_dir = Path(bronze_root)
    # Se borra la carpeta antes de escribir para que una re-ejecución no
    # deje archivos viejos mezclados con los nuevos.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_dir / "loan_default.parquet", index=False)
    return len(df)


if __name__ == "__main__":
    ruta = sys.argv[1] if len(sys.argv) > 1 else "data/raw/Loan_default.csv"
    filas = ingest_loan_default_to_bronze(ruta)
    print(
        f"Ingesta completa: {filas:,} préstamos escritos en data/bronze/loan_default/"
    )
