"""
Capa de datos del Dashboard Ejecutivo (Fase 4, entregable "Dashboard
ejecutivo v1" del plan; sección 12 del documento del proyecto).

Todo lo que el dashboard muestra se calcula aquí, separado de la parte
visual (ejecutivo.py), para poder probarlo con pytest sin Streamlit.

Fuentes (todas ya construidas por el DAG):
  - Gold: kpis_cliente, prob_impago_cliente y las métricas del modelo.
  - Silver: transacciones (para el histórico mensual, el mapa de calor y el
    consumo por categoría) y préstamos (para explicar la probabilidad).

Solo usa pandas y numpy (la imagen del dashboard no tiene Spark).
"""

import json

import numpy as np
import pandas as pd

from src.dashboard.datos import CATEGORIAS, Rutas
from src.ml.explicacion import (
    NIVELES,
    explicar,
    frase_factor,
    nivel_riesgo,
    preparar_prestamos,
)

# Clasificación de consumo por categoría (regla fija, v1). "Esencial" es lo
# que un hogar no puede dejar de pagar; "discrecional" es lo que se puede
# recortar. El dataset no trae comercios reales ni cargos recurrentes, así
# que no se pueden detectar suscripciones; se clasifica la categoría.
TIPO_CONSUMO = {
    "grocery_pos": "Esencial",
    "grocery_net": "Esencial",
    "gas_transport": "Esencial",
    "home": "Esencial",
    "health_fitness": "Esencial",
    "kids_pets": "Esencial",
    "shopping_pos": "Discrecional",
    "shopping_net": "Discrecional",
    "entertainment": "Discrecional",
    "food_dining": "Discrecional",
    "travel": "Discrecional",
    "personal_care": "Discrecional",
    "misc_pos": "Discrecional",
    "misc_net": "Discrecional",
}

DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
MESES = [
    "ene",
    "feb",
    "mar",
    "abr",
    "may",
    "jun",
    "jul",
    "ago",
    "sep",
    "oct",
    "nov",
    "dic",
]

COLUMNAS_PRESTAMO = [
    "loan_id",
    "age",
    "months_employed",
    "num_credit_lines",
    "interest_rate",
    "loan_term",
    "loan_amount",
    "credit_score",
    "education",
    "employment_type",
    "marital_status",
    "has_mortgage",
    "has_dependents",
    "has_cosigner",
    "loan_purpose",
]


# ---------------------------------------------------------------------------
# Clientes
# ---------------------------------------------------------------------------


def cargar_metricas(rutas: Rutas):
    if not rutas.metricas_modelo.exists():
        return None
    return json.loads(rutas.metricas_modelo.read_text(encoding="utf-8"))


def se_puede_explicar(metricas) -> bool:
    """Las métricas traen intercepto, medias y desviaciones desde la versión
    del modelo de oct 2026. Si el DAG no se ha vuelto a correr, no hay."""
    if not metricas or "intercepto" not in metricas:
        return False
    return all("media" in c for c in metricas.get("coeficientes", []))


def ids_clientes(cc_num: pd.Series) -> pd.Series:
    """Identificador anónimo y estable: 'Cliente 0001', … según el orden del
    número de tarjeta. El dashboard nunca muestra el número de tarjeta."""
    orden = cc_num.rank(method="first").astype(int)
    return orden.map(lambda n: f"Cliente {n:04d}")


def unir_clientes(
    kpis: pd.DataFrame, prob: pd.DataFrame, loans: pd.DataFrame
) -> pd.DataFrame:
    """Un renglón por cliente: KPIs de Gold + probabilidad de impago + las
    características de su préstamo asignado (para explicar el riesgo)."""
    c = kpis.merge(prob[["cc_num", "prob_impago"]], on="cc_num", how="left")
    c = c.merge(loans[COLUMNAS_PRESTAMO], on="loan_id", how="left")
    # Mismo nombre que usa el modelo para el DTI calculado.
    c["dti"] = c["ratio_endeudamiento"]
    c["nivel"] = nivel_riesgo(c["prob_impago"])
    c["cliente"] = ids_clientes(c["cc_num"])
    return c.sort_values("cliente").reset_index(drop=True)


def cargar_clientes(rutas: Rutas) -> pd.DataFrame:
    return unir_clientes(
        pd.read_parquet(rutas.gold_kpis),
        pd.read_parquet(rutas.gold_prob),
        pd.read_parquet(rutas.silver_loans, columns=COLUMNAS_PRESTAMO),
    )


def explicar_clientes(clientes: pd.DataFrame, metricas) -> pd.DataFrame:
    """Aporte de cada variable a la probabilidad de cada cliente (una
    columna por variable), o None si las métricas no lo permiten."""
    if not se_puede_explicar(metricas):
        return None
    _, aportes = explicar(clientes, metricas)
    aportes.index = clientes.index
    return aportes


def factores(aportes: pd.Series, fila: pd.Series, n: int = 6) -> pd.DataFrame:
    """Las n variables que más mueven el riesgo de un cliente."""
    top = aportes.reindex(aportes.abs().sort_values(ascending=False).index[:n])
    return pd.DataFrame(
        {
            "variable": top.index,
            "factor": [frase_factor(v, fila) for v in top.index],
            "aporte": top.to_numpy(),
            "efecto": np.where(top.to_numpy() > 0, "Sube el riesgo", "Baja el riesgo"),
        }
    )


def factor_principal(aportes: pd.Series, fila: pd.Series) -> str:
    """El factor que más sube el riesgo de ese cliente."""
    sube = aportes[aportes > 0]
    if sube.empty:
        return "—"
    return frase_factor(sube.idxmax(), fila)


# ---------------------------------------------------------------------------
# Portafolio
# ---------------------------------------------------------------------------


def resumen_portafolio(c: pd.DataFrame) -> dict:
    n = len(c)
    return {
        "clientes": n,
        "gasto_mensual": float(c["gasto_promedio_mensual"].median()) if n else None,
        "ingreso_mensual": float(c["ingreso_mensual_simulado"].median()) if n else None,
        "ahorro": float(c["capacidad_ahorro"].median()) if n else None,
        "ahorro_negativo": int((c["capacidad_ahorro"] < 0).sum()),
        "dti": float(c["dti"].median()) if n else None,
        "dti_mayor_1": int((c["dti"] > 1).sum()),
        "prob_impago": float(c["prob_impago"].median()) if n else None,
        "riesgo_alto": int((c["nivel"] == "Alto").sum()),
    }


def tabla_niveles(c: pd.DataFrame, impago_real: dict = None) -> pd.DataFrame:
    """Clientes por nivel de riesgo, con su perfil típico."""
    g = c.groupby("nivel", observed=False)
    t = pd.DataFrame(
        {
            "clientes": g.size(),
            "prob_promedio": g["prob_impago"].mean(),
            "ahorro_mediano": g["capacidad_ahorro"].median(),
            "dti_mediano": g["dti"].median(),
        }
    ).reindex(list(NIVELES))
    t["clientes"] = t["clientes"].fillna(0).astype(int)
    t["pct"] = t["clientes"] / max(len(c), 1)
    if impago_real is not None:
        t["impago_real"] = [impago_real.get(n) for n in t.index]
    return t


def impago_real_por_nivel(loans: pd.DataFrame, metricas) -> dict:
    """Validación de los niveles: de los 255 mil préstamos del dataset, qué
    porcentaje cayó en impago en cada nivel según el modelo."""
    if not se_puede_explicar(metricas):
        return None
    datos = preparar_prestamos(loans)
    prob, _ = explicar(datos, metricas)
    niveles = nivel_riesgo(pd.Series(prob, index=datos.index))
    tasa = datos.groupby(niveles, observed=False)["is_default"].mean()
    return {n: float(tasa.get(n)) for n in NIVELES}


def alertas(c: pd.DataFrame, aportes: pd.DataFrame) -> pd.DataFrame:
    """Alertas tempranas: clientes en riesgo alto, del mayor al menor."""
    altos = c[c["nivel"] == "Alto"].sort_values("prob_impago", ascending=False)
    res = altos[
        ["cliente", "prob_impago", "dti", "capacidad_ahorro", "gasto_promedio_mensual"]
    ].copy()
    if aportes is not None:
        res["factor"] = [
            factor_principal(aportes.loc[i], c.loc[i]) for i in altos.index
        ]
    else:
        res["factor"] = "—"
    return res.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Transacciones (módulo histórico)
# ---------------------------------------------------------------------------


def cargar_transacciones(rutas: Rutas) -> pd.DataFrame:
    tx = pd.read_parquet(
        rutas.silver_tx,
        columns=["cc_num", "amt", "category", "trans_date_trans_time"],
    )
    tx["category"] = tx["category"].astype(str)
    return tx


def meses_completos(tx: pd.DataFrame) -> list:
    """Meses con todos sus días en los datos. El dataset empieza el 21 de
    junio de 2020: ese junio incompleto haría ver un gasto falsamente bajo."""
    fechas = tx["trans_date_trans_time"]
    periodo = fechas.dt.to_period("M")
    dias = fechas.dt.date.groupby(periodo).nunique()
    return [p for p, d in dias.items() if d == p.days_in_month]


def nombre_mes(periodo) -> str:
    return f"{MESES[periodo.month - 1]} {periodo.year}"


def ingresos_vs_gastos(tx: pd.DataFrame, clientes: pd.DataFrame) -> pd.DataFrame:
    """Por mes completo: gasto promedio por cliente (real, de Silver) contra
    su ingreso promedio (simulado, Opción A). Los clientes que no gastaron en
    un mes cuentan con 0, así que el promedio no se infla."""
    n = len(clientes)
    if n == 0:
        return pd.DataFrame(columns=["mes", "orden", "ingreso", "gasto", "ahorro"])
    completos = meses_completos(tx)
    sub = tx[tx["cc_num"].isin(clientes["cc_num"])]
    periodo = sub["trans_date_trans_time"].dt.to_period("M")
    gasto = sub.groupby(periodo)["amt"].sum().reindex(completos, fill_value=0.0) / n
    ingreso = float(clientes["ingreso_mensual_simulado"].mean())
    res = pd.DataFrame(
        {
            "mes": [nombre_mes(p) for p in completos],
            "orden": range(len(completos)),
            "ingreso": ingreso,
            "gasto": gasto.to_numpy(),
        }
    )
    res["ahorro"] = (res["ingreso"] - res["gasto"]) / res["ingreso"]
    return res


def mapa_calor(tx: pd.DataFrame) -> pd.DataFrame:
    """% de las transacciones por día de la semana y hora del día."""
    t = tx["trans_date_trans_time"]
    conteo = (
        pd.DataFrame({"dia": t.dt.dayofweek, "hora": t.dt.hour})
        .value_counts()
        .rename("transacciones")
    )
    idx = pd.MultiIndex.from_product([range(7), range(24)], names=["dia", "hora"])
    res = conteo.reindex(idx, fill_value=0).reset_index()
    total = max(int(res["transacciones"].sum()), 1)
    res["pct"] = res["transacciones"] / total
    res["dia_nombre"] = res["dia"].map(lambda d: DIAS[d])
    return res


def consumo_por_categoria(tx: pd.DataFrame) -> pd.DataFrame:
    """Gasto por categoría, en español y clasificado esencial/discrecional."""
    g = tx.groupby("category")["amt"].sum().sort_values(ascending=False)
    res = pd.DataFrame(
        {
            "categoria": [CATEGORIAS.get(c, c) for c in g.index],
            "tipo": [TIPO_CONSUMO.get(c, "Discrecional") for c in g.index],
            "gasto": g.to_numpy(),
        }
    )
    total = res["gasto"].sum()
    res["pct"] = res["gasto"] / total if total else 0.0
    return res


def pct_esencial(consumo: pd.DataFrame) -> float:
    total = consumo["gasto"].sum()
    if not total:
        return 0.0
    return float(consumo.loc[consumo["tipo"] == "Esencial", "gasto"].sum() / total)


def hora_pico(calor: pd.DataFrame) -> tuple:
    """(día, hora) con más transacciones."""
    fila = calor.loc[calor["transacciones"].idxmax()]
    return fila["dia_nombre"], int(fila["hora"])
