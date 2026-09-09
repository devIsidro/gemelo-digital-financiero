"""
Data profiling de la fuente "Credit Card Transactions Fraud Detection"
(Fase 2 / Semana 5). Genera un perfil columna por columna sobre el CSV
crudo: tipo, % de nulos, cardinalidad y rango/ejemplos. Se usa como base
para el diccionario de datos.
"""

import json
import sys

import pandas as pd


def profile_csv(csv_path: str) -> list[dict]:
    df = pd.read_csv(csv_path)
    perfil = []

    for col in df.columns:
        serie = df[col]
        info = {
            "columna": col,
            "dtype": str(serie.dtype),
            "n_nulos": int(serie.isna().sum()),
            "pct_nulos": round(100 * serie.isna().mean(), 2),
            "n_unicos": int(serie.nunique()),
            "ejemplos": [str(v) for v in serie.dropna().unique()[:3]],
        }
        if pd.api.types.is_numeric_dtype(serie):
            info["min"] = float(serie.min())
            info["max"] = float(serie.max())
            info["media"] = round(float(serie.mean()), 2)
        perfil.append(info)

    return perfil


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python data_profiling.py <ruta_al_csv>")
        sys.exit(1)

    resultado = profile_csv(sys.argv[1])
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
