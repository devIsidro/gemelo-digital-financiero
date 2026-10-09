"""
Capa de datos del Panel Operativo.

Todo lo que el panel muestra se calcula aquí, separado de la parte visual
(operativo.py), para poder probarlo con pytest sin levantar Streamlit.

Fuentes:
  - Parquet de Bronze, Silver y Gold en data/ (montado en el contenedor).
  - Métricas del modelo de impago (data/gold/modelo_impago_metricas.json).
  - Metadatos de Airflow (corridas y pasos del DAG bronze_ingest), leídos
    directo de la base de datos de Airflow. Si no hay conexión, el panel
    sigue funcionando sin esa parte.

Solo usa pandas (la imagen del panel no tiene Spark).
"""

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.spark.validate_silver import ResultadoExpectativa, validar_silver

DAG_ID = "bronze_ingest"
ZONA_HORARIA = "America/Mexico_City"

# Pasos del DAG en el orden en que se muestran, agrupados por etapa.
PASOS = [
    ("ingest_source", "Ingesta a Bronze", "Transacciones"),
    ("clean_silver", "Limpieza en Silver", "Transacciones"),
    ("validate_quality", "Validación de calidad", "Transacciones"),
    ("ingest_loan_default", "Ingesta a Bronze", "Préstamos"),
    ("clean_silver_loans", "Limpieza y validación", "Préstamos"),
    ("build_gold_perfil", "Perfil de cliente", "Gold y modelo"),
    ("simular_ingreso_credito", "Ingreso simulado", "Gold y modelo"),
    ("asignar_prestamos", "Llave sintética", "Gold y modelo"),
    ("calcular_kpis", "KPIs financieros", "Gold y modelo"),
    ("modelo_prob_impago", "Modelo de impago", "Gold y modelo"),
]


@dataclass
class Rutas:
    base: Path = Path("data")

    @property
    def bronze_tx(self):
        return self.base / "bronze" / "transactions"

    @property
    def silver_tx(self):
        return self.base / "silver" / "transactions"

    @property
    def bronze_loans(self):
        return self.base / "bronze" / "loan_default"

    @property
    def silver_loans(self):
        return self.base / "silver" / "loan_default"

    @property
    def gold_kpis(self):
        return self.base / "gold" / "kpis_cliente"

    @property
    def gold_asignacion(self):
        return self.base / "gold" / "asignacion_prestamos"

    @property
    def gold_prob(self):
        return self.base / "gold" / "prob_impago_cliente"

    @property
    def metricas_modelo(self):
        return self.base / "gold" / "modelo_impago_metricas.json"


def _contar(ruta: Path, columna: str):
    """Número de filas de un Parquet (leyendo una sola columna), o None si
    todavía no existe."""
    if not ruta.exists():
        return None
    return len(pd.read_parquet(ruta, columns=[columna]))


# ---------------------------------------------------------------------------
# Volumen por capa
# ---------------------------------------------------------------------------


def conteos(rutas: Rutas) -> dict:
    """Filas por capa. None = la tabla todavía no existe."""
    return {
        "bronze_tx": _contar(rutas.bronze_tx, "trans_num"),
        "silver_tx": _contar(rutas.silver_tx, "trans_num"),
        "bronze_loans": _contar(rutas.bronze_loans, "LoanID"),
        "silver_loans": _contar(rutas.silver_loans, "loan_id"),
        "clientes_gold": _contar(rutas.gold_kpis, "cc_num"),
    }


def transacciones_silver(rutas: Rutas) -> pd.DataFrame:
    """Columnas de Silver que usan las gráficas de volumen."""
    return pd.read_parquet(
        rutas.silver_tx,
        columns=["trans_num", "category", "is_fraud", "year", "month", "day"],
    )


def volumen_por_dia(silver: pd.DataFrame) -> pd.DataFrame:
    por_dia = (
        silver.groupby(["year", "month", "day"], observed=True)
        .size()
        .reset_index(name="transacciones")
    )
    por_dia["fecha"] = pd.to_datetime(por_dia[["year", "month", "day"]])
    return por_dia[["fecha", "transacciones"]].sort_values("fecha")


# Nombres en español de las categorías del dataset (pos = compra en tienda,
# net = compra en línea).
CATEGORIAS = {
    "gas_transport": "Gasolina y transporte",
    "grocery_pos": "Supermercado (tienda)",
    "grocery_net": "Supermercado (en línea)",
    "home": "Hogar",
    "shopping_pos": "Compras (tienda)",
    "shopping_net": "Compras (en línea)",
    "kids_pets": "Niños y mascotas",
    "entertainment": "Entretenimiento",
    "personal_care": "Cuidado personal",
    "food_dining": "Restaurantes",
    "health_fitness": "Salud y ejercicio",
    "misc_pos": "Varios (tienda)",
    "misc_net": "Varios (en línea)",
    "travel": "Viajes",
}


def volumen_por_categoria(silver: pd.DataFrame) -> pd.DataFrame:
    por_cat = silver["category"].value_counts().reset_index()
    por_cat.columns = ["categoria", "transacciones"]
    por_cat["categoria"] = por_cat["categoria"].map(
        lambda c: CATEGORIAS.get(c, str(c).replace("_", " ").capitalize())
    )
    return por_cat


# ---------------------------------------------------------------------------
# Calidad de datos
# ---------------------------------------------------------------------------


def validar_transacciones(rutas: Rutas) -> list:
    """Las mismas 6 reglas que corre el paso validate_quality del DAG."""
    return validar_silver(str(rutas.silver_tx))


def validar_prestamos(loans: pd.DataFrame) -> list:
    """Reglas de calidad del Silver de préstamos (las mismas que aplica
    silver_loans.py, revisadas aquí con pandas)."""
    r = []
    dup = int(loans["loan_id"].duplicated().sum())
    nulos = int(loans["loan_id"].isna().sum())
    r.append(
        ResultadoExpectativa(
            "loan_id no nulo y único",
            dup == 0 and nulos == 0,
            f"{nulos} nulos, {dup} duplicados",
        )
    )
    malos = int(((loans["income_anual"] <= 0) | (loans["loan_amount"] <= 0)).sum())
    r.append(
        ResultadoExpectativa(
            "ingreso y monto > 0", malos == 0, f"{malos} préstamos con valores <= 0"
        )
    )
    fuera = int((~loans["credit_score"].between(300, 850)).sum())
    r.append(
        ResultadoExpectativa(
            "score entre 300 y 850", fuera == 0, f"{fuera} fuera de rango"
        )
    )
    malos_def = int((~loans["is_default"].isin([0, 1])).sum())
    r.append(
        ResultadoExpectativa(
            "is_default en {0, 1}",
            malos_def == 0,
            f"{malos_def} valores fuera de rango",
        )
    )
    tasa = float(loans["is_default"].mean()) if len(loans) else 0.0
    r.append(
        ResultadoExpectativa(
            "tasa de impago esperada (5%–25%)",
            0.05 <= tasa <= 0.25,
            f"{tasa:.1%} de impago",
        )
    )
    return r


def prestamos_silver(rutas: Rutas) -> pd.DataFrame:
    return pd.read_parquet(
        rutas.silver_loans,
        columns=[
            "loan_id",
            "income_anual",
            "loan_amount",
            "credit_score",
            "is_default",
        ],
    )


# ---------------------------------------------------------------------------
# Gold y modelo
# ---------------------------------------------------------------------------


def resumen_gold(rutas: Rutas) -> dict:
    """Salud de la capa Gold y del modelo de impago. Campos en None si la
    tabla todavía no existe."""
    res = {
        "clientes": None,
        "prestamos_asignados": None,
        "prestamos_unicos": None,
        "prob_impago_mediana": None,
        "clientes_riesgo_alto": None,
        "modelo": None,
    }
    if rutas.gold_kpis.exists():
        k = pd.read_parquet(rutas.gold_kpis, columns=["cc_num"])
        res["clientes"] = len(k)
    if rutas.gold_asignacion.exists():
        a = pd.read_parquet(rutas.gold_asignacion, columns=["cc_num", "loan_id"])
        res["prestamos_asignados"] = len(a)
        res["prestamos_unicos"] = int(a["loan_id"].nunique())
    if rutas.gold_prob.exists():
        p = pd.read_parquet(rutas.gold_prob, columns=["cc_num", "prob_impago"])
        res["prob_impago_mediana"] = float(p["prob_impago"].median())
        res["clientes_riesgo_alto"] = int((p["prob_impago"] >= 0.30).sum())
    if rutas.metricas_modelo.exists():
        res["modelo"] = json.loads(rutas.metricas_modelo.read_text(encoding="utf-8"))
    return res


def pct_fraude(silver: pd.DataFrame) -> float:
    return float(silver["is_fraud"].mean() * 100)


# ---------------------------------------------------------------------------
# Airflow (corridas del DAG)
# ---------------------------------------------------------------------------


def url_airflow() -> str:
    """Conexión a la base de datos de Airflow. En Docker apunta al servicio
    postgres; para pruebas se puede usar sqlite:///ruta/airflow.db."""
    return os.environ.get(
        "AIRFLOW_DB_URL", "postgresql://airflow:airflow@postgres:5432/airflow"
    )


def _consulta(url: str, sql: str, params: tuple) -> pd.DataFrame:
    if url.startswith("sqlite:///"):
        con = sqlite3.connect(url.removeprefix("sqlite:///"))
        marca = "?"
    else:
        import psycopg2

        con = psycopg2.connect(url, connect_timeout=3)
        marca = "%s"
    try:
        cur = con.cursor()
        cur.execute(sql.replace("{p}", marca), params)
        columnas = [c[0] for c in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=columnas)
    finally:
        con.close()


def _a_hora_local(serie: pd.Series) -> pd.Series:
    fechas = pd.to_datetime(serie, utc=True, errors="coerce", format="mixed")
    return fechas.dt.tz_convert(ZONA_HORARIA)


def corridas(url: str, limite: int = 30) -> pd.DataFrame:
    """Últimas corridas del DAG, de la más vieja a la más nueva."""
    df = _consulta(
        url,
        "SELECT run_id, run_type, state, start_date, end_date FROM dag_run "
        "WHERE dag_id = {p} ORDER BY id DESC LIMIT {p}",
        (DAG_ID, limite),
    )
    if df.empty:
        return df
    df["inicio"] = _a_hora_local(df["start_date"])
    df["fin"] = _a_hora_local(df["end_date"])
    df["minutos"] = (df["fin"] - df["inicio"]).dt.total_seconds() / 60
    df["tipo"] = df["run_type"].map({"manual": "Manual", "scheduled": "Programada"})
    df["tipo"] = df["tipo"].fillna(df["run_type"])
    return df.sort_values("inicio").reset_index(drop=True)


def resumen_corridas(df: pd.DataFrame) -> dict:
    """KPI tasa_exito_ingesta del catálogo: corridas exitosas / terminadas."""
    terminadas = df[df["state"].isin(["success", "failed"])]
    exitosas = int((terminadas["state"] == "success").sum())
    total = len(terminadas)
    ultima = df.iloc[-1] if len(df) else None
    return {
        "exitosas": exitosas,
        "terminadas": total,
        "tasa_exito": (exitosas / total * 100) if total else None,
        "ultima": ultima,
        "duracion_tipica": (
            float(terminadas.loc[terminadas["state"] == "success", "minutos"].median())
            if exitosas
            else None
        ),
    }


def pasos_de_corrida(url: str, run_id: str) -> pd.DataFrame:
    """Estado y duración de los 10 pasos de una corrida, en el orden de PASOS
    (los que todavía no existen en la corrida aparecen como pendientes)."""
    df = _consulta(
        url,
        "SELECT task_id, state, duration, try_number FROM task_instance "
        "WHERE dag_id = {p} AND run_id = {p}",
        (DAG_ID, run_id),
    )
    base = pd.DataFrame(PASOS, columns=["task_id", "nombre", "etapa"])
    base["orden"] = range(len(base))
    res = base.merge(df, on="task_id", how="left")
    res["state"] = res["state"].fillna("pendiente")
    return res.sort_values("orden").reset_index(drop=True)
