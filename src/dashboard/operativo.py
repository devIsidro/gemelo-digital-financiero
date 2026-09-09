"""
Dashboard operativo / de observabilidad (Fase 3 / Semana 12).

Panel para el equipo de Data Engineering: muestra el estado de salud del
pipeline Bronze -> Silver (volumen de datos, filas descartadas, resultado
de las validaciones de calidad). Es la "Vista 1" del diagrama de flujo de
datos del proyecto -- distinto del dashboard ejecutivo de KPIs financieros
que se construye en la Fase 4 sobre la Capa Gold.

Corre con: streamlit run src/dashboard/operativo.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.spark.validate_silver import validar_silver  # noqa: E402

BRONZE_PATH = "data/bronze/transactions"
SILVER_PATH = "data/silver/transactions"

st.set_page_config(
    page_title="Gemelo Digital Financiero — Panel Operativo",
    layout="wide",
)


@st.cache_data(ttl=60)
def cargar_datos():
    bronze = pd.read_parquet(BRONZE_PATH)
    silver = pd.read_parquet(SILVER_PATH)
    return bronze, silver


@st.cache_data(ttl=60)
def cargar_validacion():
    return validar_silver(SILVER_PATH)


st.title("Panel Operativo — Gemelo Digital Financiero")
st.caption(
    "Vista técnica del pipeline de datos (Bronze → Silver). "
    "Fase 3 / Semana 12 — equipo de Data Engineering."
)

try:
    bronze_df, silver_df = cargar_datos()
    error_carga = None
except Exception as e:  # noqa: BLE001
    bronze_df, silver_df = None, None
    error_carga = str(e)

if error_carga:
    st.error(
        f"No se pudieron cargar los datos de Bronze/Silver. "
        f"¿Ya corriste el pipeline en Airflow al menos una vez? Detalle: {error_carga}"
    )
    st.stop()

# --- Fila de métricas principales ---
col1, col2, col3, col4 = st.columns(4)

filas_bronze = len(bronze_df)
filas_silver = len(silver_df)
descartadas = filas_bronze - filas_silver
pct_descartadas = (descartadas / filas_bronze * 100) if filas_bronze else 0

col1.metric("Filas en Bronze", f"{filas_bronze:,}")
col2.metric("Filas en Silver", f"{filas_silver:,}")
col3.metric(
    "Descartadas en limpieza",
    f"{descartadas:,}",
    delta=f"{pct_descartadas:.2f}% del total",
    delta_color="inverse",
)
tasa_exito = 100 - pct_descartadas
col4.metric("Tasa de éxito de ingesta", f"{tasa_exito:.2f}%")

st.divider()

# --- Validaciones de calidad ---
st.subheader("Validaciones de calidad de datos")

resultados = cargar_validacion()
todo_ok = all(r.ok for r in resultados)

if todo_ok:
    st.success("Todas las validaciones de calidad pasaron.")
else:
    fallidas = [r.nombre for r in resultados if not r.ok]
    st.error(f"Fallaron {len(fallidas)} validaciones: {', '.join(fallidas)}")

tabla_validacion = pd.DataFrame(
    [
        {"Regla": r.nombre, "Estado": "PASS" if r.ok else "FAIL", "Detalle": r.detalle}
        for r in resultados
    ]
)
st.dataframe(tabla_validacion, use_container_width=True, hide_index=True)

st.divider()

# --- Distribución por categoría y por día ---
col_izq, col_der = st.columns(2)

with col_izq:
    st.subheader("Transacciones por categoría (Silver)")
    por_categoria = silver_df["category"].value_counts().reset_index()
    por_categoria.columns = ["category", "count"]
    st.bar_chart(por_categoria.set_index("category"))

with col_der:
    st.subheader("Volumen de transacciones por día")
    por_dia = (
        silver_df.groupby(["year", "month", "day"])
        .size()
        .reset_index(name="transacciones")
    )
    por_dia["fecha"] = pd.to_datetime(por_dia[["year", "month", "day"]])
    por_dia = por_dia.sort_values("fecha")
    st.line_chart(por_dia.set_index("fecha")["transacciones"])

st.divider()

# --- Indicador de fraude (observabilidad de negocio, no solo técnica) ---
st.subheader("Indicador de fraude en el lote actual")
pct_fraude = silver_df["is_fraud"].mean() * 100
st.metric("% de transacciones marcadas como fraude", f"{pct_fraude:.3f}%")

st.caption(
    "Panel generado a partir de los datos actuales en `data/bronze` y `data/silver`. "
    "Se actualiza automáticamente cada vez que corre el DAG `bronze_ingest`."
)
