"""
Validación de calidad de datos sobre la Capa Silver (Fase 3 / Semana 11).

Corre una batería de expectativas de negocio sobre el dataset de
transacciones en Silver y reporta cuáles pasan y cuáles no. Es el
equivalente funcional a una suite de Great Expectations / dbt tests:
mismo objetivo (detectar datos que no cumplen las reglas del negocio
antes de que lleguen a Gold), implementado con pandas para no depender
de librerías inestables en este entorno.
"""

import sys
from dataclasses import dataclass

import pandas as pd


@dataclass
class ResultadoExpectativa:
    nombre: str
    ok: bool
    detalle: str


def validar_silver(
    silver_path: str = "data/silver/transactions",
) -> list[ResultadoExpectativa]:
    df = pd.read_parquet(silver_path)
    resultados = []

    # 1. trans_num no nulo y único (llave de negocio)
    nulos_trans_num = df["trans_num"].isna().sum()
    duplicados_trans_num = df["trans_num"].duplicated().sum()
    resultados.append(
        ResultadoExpectativa(
            "trans_num no nulo y único",
            nulos_trans_num == 0 and duplicados_trans_num == 0,
            f"{nulos_trans_num} nulos, {duplicados_trans_num} duplicados",
        )
    )

    # 2. cc_num no nulo
    nulos_cc = df["cc_num"].isna().sum()
    resultados.append(
        ResultadoExpectativa("cc_num no nulo", nulos_cc == 0, f"{nulos_cc} nulos")
    )

    # 3. amt > 0
    invalidos_amt = (df["amt"] <= 0).sum()
    resultados.append(
        ResultadoExpectativa(
            "amt > 0", invalidos_amt == 0, f"{invalidos_amt} filas con monto <= 0"
        )
    )

    # 4. is_fraud solo puede ser 0 o 1
    invalidos_fraud = (~df["is_fraud"].isin([0, 1])).sum()
    resultados.append(
        ResultadoExpectativa(
            "is_fraud en {0, 1}",
            invalidos_fraud == 0,
            f"{invalidos_fraud} valores fuera de rango",
        )
    )

    # 5. category no vacía
    vacios_category = (df["category"].isna() | (df["category"].str.strip() == "")).sum()
    resultados.append(
        ResultadoExpectativa(
            "category no vacía", vacios_category == 0, f"{vacios_category} filas vacías"
        )
    )

    # 6. trans_date_trans_time dentro de un rango razonable (no futuro, no antes de 2010)
    fechas = pd.to_datetime(df["trans_date_trans_time"])
    fuera_rango = ((fechas < "2010-01-01") | (fechas > pd.Timestamp.now())).sum()
    resultados.append(
        ResultadoExpectativa(
            "fecha de transacción en rango válido",
            fuera_rango == 0,
            f"{fuera_rango} fechas fuera de rango",
        )
    )

    return resultados


def imprimir_reporte(resultados: list[ResultadoExpectativa]) -> bool:
    print("=== Reporte de calidad de datos — Capa Silver ===")
    todo_ok = True
    for r in resultados:
        estado = "PASS" if r.ok else "FAIL"
        if not r.ok:
            todo_ok = False
        print(f"[{estado}] {r.nombre} — {r.detalle}")
    print("=" * 50)
    print("RESULTADO GENERAL:", "TODO OK" if todo_ok else "HAY FALLAS")
    return todo_ok


if __name__ == "__main__":
    ruta = sys.argv[1] if len(sys.argv) > 1 else "data/silver/transactions"
    resultados = validar_silver(ruta)
    todo_ok = imprimir_reporte(resultados)
    sys.exit(0 if todo_ok else 1)
