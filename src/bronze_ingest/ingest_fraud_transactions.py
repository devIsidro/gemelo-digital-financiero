"""
Ingesta real (Fase 2) del dataset "Credit Card Transactions Fraud Detection"
hacia la Capa Bronze.

Lee el CSV crudo tal como viene de Kaggle y lo escribe particionado por
year/month/day en formato Parquet. A propósito NO se limpia ni se valida
nada de negocio aquí (nulos, duplicados, formatos) — eso es trabajo de la
Capa Silver. Bronze solo preserva el dato crudo de forma auditable.
"""

import sys
from pathlib import Path

import pandas as pd


def ingest_csv_to_bronze(csv_path: str, bronze_root: str = "data/bronze/transactions") -> int:
    """Lee el CSV crudo y lo escribe particionado por year/month/day en Bronze.

    Devuelve el número de filas escritas.
    """
    df = pd.read_csv(csv_path)

    # El CSV de Kaggle trae una columna de índice sin nombre ("Unnamed: 0").
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])

    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["year"] = df["trans_date_trans_time"].dt.year
    df["month"] = df["trans_date_trans_time"].dt.month
    df["day"] = df["trans_date_trans_time"].dt.day

    out_dir = Path(bronze_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_dir, partition_cols=["year", "month", "day"], index=False)
    return len(df)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python ingest_fraud_transactions.py <ruta_al_csv>")
        sys.exit(1)

    rows = ingest_csv_to_bronze(sys.argv[1])
    print(f"Ingesta completa: {rows:,} filas escritas en data/bronze/transactions/")