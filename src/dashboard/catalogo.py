"""
Catálogo de datos del Lakehouse (Panel Operativo).

Para cada tabla de Bronze, Silver y Gold dice qué contiene, de dónde viene,
qué paso del DAG la genera, qué significa cada columna y si guarda datos
personales. El estado (filas, columnas y última actualización) se lee de los
archivos Parquet, así que siempre está al día; si aparece una columna que no
está documentada aquí, el panel lo marca.

El proyecto nació de un problema de adopción del catálogo corporativo: la
gente consultaba fuentes no oficiales porque no encontraba ni entendía los
datos. Este catálogo es la versión del proyecto de esa solución.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from src.dashboard.datos import ZONA_HORARIA, Rutas


@dataclass(frozen=True)
class Tabla:
    nombre: str
    capa: str
    ruta: str  # relativa a la carpeta data/
    descripcion: str
    origen: str
    paso: str
    llave: str
    columnas: dict
    personales: frozenset = field(default_factory=frozenset)
    patrones: tuple = ()  # (regex, descripción) para columnas repetitivas


_PERCENTIL = {"p5": "caso malo (percentil 5)", "p50": "caso típico (mediana)"}
_PERCENTIL["p95"] = "caso bueno (percentil 95)"

CATALOGO = (
    Tabla(
        "transactions",
        "Bronze",
        "bronze/transactions",
        "Transacciones con tarjeta tal cual vienen del CSV de Kaggle (histórico auditable).",
        "Kaggle: Credit Card Transactions Fraud Detection (fraudTest.csv)",
        "ingest_source",
        "trans_num",
        {
            "trans_date_trans_time": "fecha y hora de la transacción",
            "cc_num": "número de tarjeta (identifica al cliente)",
            "merchant": "nombre del comercio",
            "category": "categoría del comercio (en inglés)",
            "amt": "monto de la compra (USD)",
            "first": "nombre del titular",
            "last": "apellido del titular",
            "gender": "género del titular",
            "street": "calle del domicilio",
            "city": "ciudad del domicilio",
            "state": "estado del domicilio",
            "zip": "código postal",
            "lat": "latitud del domicilio",
            "long": "longitud del domicilio",
            "city_pop": "población de la ciudad",
            "job": "ocupación del titular",
            "dob": "fecha de nacimiento",
            "trans_num": "identificador único de la transacción",
            "unix_time": "fecha y hora en formato Unix",
            "merch_lat": "latitud del comercio",
            "merch_long": "longitud del comercio",
            "is_fraud": "1 si la transacción fue fraude",
            "year": "año (partición)",
            "month": "mes (partición)",
            "day": "día (partición)",
        },
        frozenset(
            {
                "first",
                "last",
                "gender",
                "street",
                "city",
                "state",
                "zip",
                "lat",
                "long",
                "job",
                "dob",
            }
        ),
    ),
    Tabla(
        "loan_default",
        "Bronze",
        "bronze/loan_default",
        "Préstamos tal cual vienen del CSV de Kaggle.",
        "Kaggle: Loan Default Prediction (Loan_default.csv)",
        "ingest_loan_default",
        "LoanID",
        {
            "LoanID": "identificador del préstamo",
            "Age": "edad del solicitante",
            "Income": "ingreso anual declarado (USD)",
            "LoanAmount": "monto del préstamo (USD)",
            "CreditScore": "score crediticio (300 a 850)",
            "MonthsEmployed": "meses en su empleo actual",
            "NumCreditLines": "líneas de crédito abiertas",
            "InterestRate": "tasa de interés anual (%)",
            "LoanTerm": "plazo en meses",
            "DTIRatio": "DTI que trae el dataset (no se usa: no está documentado)",
            "Education": "nivel de estudios",
            "EmploymentType": "tipo de empleo",
            "MaritalStatus": "estado civil",
            "HasMortgage": "si tiene hipoteca",
            "HasDependents": "si tiene dependientes",
            "LoanPurpose": "para qué es el préstamo",
            "HasCoSigner": "si tiene aval",
            "Default": "1 si cayó en impago",
        },
    ),
    Tabla(
        "transactions",
        "Silver",
        "silver/transactions",
        "Transacciones limpias: sin duplicados ni montos inválidos, tipos correctos y sin "
        "datos personales.",
        "Bronze transactions",
        "clean_silver + validate_quality",
        "trans_num",
        {
            "trans_num": "identificador único de la transacción",
            "cc_num": "número de tarjeta (identifica al cliente)",
            "amt": "monto de la compra (USD)",
            "category": "categoría del comercio (en minúsculas)",
            "merchant": "nombre del comercio",
            "trans_date_trans_time": "fecha y hora de la transacción",
            "is_fraud": "1 si la transacción fue fraude",
            "city_pop": "población de la ciudad (sin el domicilio)",
            "year": "año (partición)",
            "month": "mes (partición)",
            "day": "día (partición)",
        },
    ),
    Tabla(
        "loan_default",
        "Silver",
        "silver/loan_default",
        "Préstamos validados, con nombres en minúsculas y tipos correctos.",
        "Bronze loan_default",
        "clean_silver_loans",
        "loan_id",
        {
            "loan_id": "identificador del préstamo",
            "age": "edad del solicitante",
            "income_anual": "ingreso anual declarado (USD)",
            "loan_amount": "monto del préstamo (USD)",
            "credit_score": "score crediticio (300 a 850)",
            "months_employed": "meses en su empleo actual",
            "num_credit_lines": "líneas de crédito abiertas",
            "interest_rate": "tasa de interés anual (%)",
            "loan_term": "plazo en meses",
            "dti_ratio": "DTI que trae el dataset (no se usa)",
            "education": "nivel de estudios",
            "employment_type": "tipo de empleo",
            "marital_status": "estado civil",
            "has_mortgage": "si tiene hipoteca",
            "has_dependents": "si tiene dependientes",
            "has_cosigner": "si tiene aval",
            "loan_purpose": "para qué es el préstamo",
            "is_default": "1 si cayó en impago (etiqueta del modelo)",
        },
    ),
    Tabla(
        "perfil_cliente",
        "Gold",
        "gold/perfil_cliente",
        "Gasto real de cada cliente a partir de sus transacciones.",
        "Silver transactions",
        "build_gold_perfil",
        "cc_num",
        {
            "cc_num": "cliente",
            "gasto_total": "gasto total del histórico (USD)",
            "num_transacciones": "número de compras",
            "pct_transacciones_fraude": "porcentaje de sus compras marcadas como fraude",
            "primera_transaccion": "fecha de su primera compra",
            "ultima_transaccion": "fecha de su última compra",
            "city_pop": "población de su ciudad",
            "meses_activo": "meses entre su primera y su última compra",
            "gasto_promedio_mensual": "gasto total entre meses activo (USD)",
            "categoria_principal": "categoría donde más gasta",
        },
    ),
    Tabla(
        "perfil_financiero_simulado",
        "Gold",
        "gold/perfil_financiero_simulado",
        "Ingreso mensual simulado de forma controlada (Opción A).",
        "Silver transactions",
        "simular_ingreso_credito",
        "cc_num",
        {
            "cc_num": "cliente",
            "ingreso_mensual_simulado": "ingreso mensual SIMULADO (USD), para ahorro y flujo",
        },
    ),
    Tabla(
        "asignacion_prestamos",
        "Gold",
        "gold/asignacion_prestamos",
        "Llave sintética: un préstamo distinto por cliente, asignado por parecido.",
        "Gold perfil_cliente + Silver loan_default",
        "asignar_prestamos",
        "cc_num",
        {
            "cc_num": "cliente",
            "loan_id": "préstamo asignado",
            "percentil_gasto": "lugar del cliente por gasto (0 a 1)",
            "percentil_ingreso": "lugar del préstamo por ingreso del titular (0 a 1)",
        },
    ),
    Tabla(
        "kpis_cliente",
        "Gold",
        "gold/kpis_cliente",
        "Una fila por cliente con sus KPIs financieros y su préstamo asignado.",
        "Gold perfil_cliente + perfil_financiero_simulado + asignacion_prestamos",
        "calcular_kpis",
        "cc_num",
        {
            "cc_num": "cliente",
            "gasto_total": "gasto total del histórico (USD)",
            "num_transacciones": "número de compras",
            "pct_transacciones_fraude": "porcentaje de sus compras marcadas como fraude",
            "primera_transaccion": "fecha de su primera compra",
            "ultima_transaccion": "fecha de su última compra",
            "city_pop": "población de su ciudad",
            "meses_activo": "meses entre su primera y su última compra",
            "gasto_promedio_mensual": "gasto promedio al mes (USD)",
            "categoria_principal": "categoría donde más gasta",
            "ingreso_mensual_simulado": "ingreso mensual simulado (USD)",
            "loan_id": "préstamo asignado",
            "ingreso_mensual_fuente": "ingreso mensual del préstamo (USD), para el DTI",
            "score_credito_fuente": "score del préstamo (de la fuente, no calculado)",
            "monto_prestamo_fuente": "monto del préstamo (USD)",
            "tasa_interes_anual_fuente": "tasa de interés anual del préstamo (%)",
            "plazo_meses_fuente": "plazo del préstamo en meses",
            "pago_mensual_prestamo": "pago mensual del préstamo (USD)",
            "flujo_efectivo_mensual": "KPI: ingreso simulado menos gasto promedio (USD)",
            "capacidad_ahorro": "KPI: flujo de efectivo entre ingreso simulado",
            "ratio_endeudamiento": "KPI: DTI = pago mensual / ingreso mensual del préstamo",
        },
    ),
    Tabla(
        "prob_impago_cliente",
        "Gold",
        "gold/prob_impago_cliente",
        "Probabilidad de impago de cada cliente según el modelo.",
        "Silver loan_default + Gold asignacion_prestamos",
        "modelo_prob_impago",
        "cc_num",
        {
            "cc_num": "cliente",
            "loan_id": "préstamo asignado",
            "prob_impago": "KPI: probabilidad de no pagar el préstamo (0 a 1)",
        },
    ),
    Tabla(
        "simulacion_insumos",
        "Gold",
        "gold/simulacion_insumos.parquet",
        "Lo que recibe el simulador Monte Carlo de cada cliente.",
        "Gold kpis_cliente + Silver transactions",
        "simulador_monte_carlo",
        "cc_num",
        {
            "cc_num": "cliente",
            "ingreso": "ingreso mensual (USD)",
            "gasto_base": "gasto mensual típico (USD)",
            "cv_gasto": "qué tanto varía su gasto de un mes a otro",
        },
    ),
    Tabla(
        "simulacion_resumen",
        "Gold",
        "gold/simulacion_resumen.parquet",
        "Resultado del simulador: 1,000 futuros a 36 meses por cliente y escenario.",
        "Gold simulacion_insumos",
        "simulador_monte_carlo",
        "cc_num + escenario",
        {
            "cc_num": "cliente",
            "escenario": "base, credito_hipotecario, cambio_empleo o negocio",
            "prob_deficit_12m": "probabilidad de quedar en números rojos en 12 meses",
            "prob_deficit": "probabilidad de quedar en números rojos en 36 meses",
            "capacidad_ahorro_12m_p50": "capacidad de ahorro típica del primer año",
            "meta_fondo_emergencia": "6 meses de gasto (USD)",
            "meses_meta_p50": "meses típicos para juntar el fondo de emergencia",
            "prob_meta_36m": "probabilidad de juntarlo en 36 meses",
            "prob_recuperar_inversion": "probabilidad de recuperar la inversión (negocio)",
            "meses_recuperacion_p50": "meses típicos para recuperarla (negocio)",
        },
        patrones=(
            (
                r"(flujo|ahorro)_m(\d+)_(p5|p50|p95)",
                lambda m: (
                    ("flujo de efectivo" if m[1] == "flujo" else "ahorro acumulado")
                    + f" del mes {m[2]}, {_PERCENTIL[m[3]]} (USD)"
                ),
            ),
        ),
    ),
)

CAPAS = ("Bronze", "Silver", "Gold")


def descripcion(tabla: Tabla, columna: str) -> str:
    if columna in tabla.columnas:
        return tabla.columnas[columna]
    for patron, texto in tabla.patrones:
        m = re.fullmatch(patron, columna)
        if m:
            return texto(m)
    return ""


def _archivos(ruta: Path) -> list:
    if not ruta.exists():
        return []
    return [ruta] if ruta.is_file() else list(ruta.rglob("*.parquet"))


def estado(tabla: Tabla, rutas: Rutas) -> dict:
    """Filas, columnas y última actualización leídas de los archivos."""
    ruta = rutas.base / tabla.ruta
    archivos = _archivos(ruta)
    if not archivos:
        return {"existe": False, "filas": None, "columnas": [], "actualizada": None}
    datos = ds.dataset(str(ruta), format="parquet", partitioning="hive")
    actualizada = pd.Timestamp(max(a.stat().st_mtime for a in archivos), unit="s")
    return {
        "existe": True,
        "filas": datos.count_rows(),
        "columnas": list(datos.schema.names),
        "actualizada": actualizada.tz_localize("UTC").tz_convert(ZONA_HORARIA),
    }


def resumen(rutas: Rutas) -> pd.DataFrame:
    """Una fila por tabla, para la vista general del catálogo."""
    filas = []
    for t in CATALOGO:
        e = estado(t, rutas)
        sin_doc = [c for c in e["columnas"] if not descripcion(t, c)]
        filas.append(
            {
                "capa": t.capa,
                "tabla": t.nombre,
                "descripcion": t.descripcion,
                "filas": e["filas"],
                "columnas": len(e["columnas"]) or None,
                "actualizada": e["actualizada"],
                "paso": t.paso,
                "personales": bool(t.personales),
                "sin_documentar": sin_doc,
                "existe": e["existe"],
            }
        )
    return pd.DataFrame(filas)


def columnas(tabla: Tabla, rutas: Rutas) -> pd.DataFrame:
    """Diccionario de una tabla: las columnas reales con su descripción."""
    reales = estado(tabla, rutas)["columnas"] or list(tabla.columnas)
    return pd.DataFrame(
        {
            "columna": reales,
            "descripcion": [descripcion(tabla, c) or "Sin documentar" for c in reales],
            "dato_personal": ["Sí" if c in tabla.personales else "" for c in reales],
        }
    )
