"""
Explicación de la probabilidad de impago de un cliente (Fase 5).

El modelo es una regresión logística sobre variables estandarizadas, así que
la probabilidad de cada préstamo se puede desarmar en el aporte de cada
variable:

    aporte_i = coeficiente_i × (valor_i − media_i) / desviación_i
    prob     = 1 / (1 + e^−(intercepto + Σ aporte_i))

Aporte positivo = esa variable sube el riesgo de ese cliente frente al
préstamo promedio; negativo = lo baja. Todo sale del JSON de métricas que
escribe modelo_impago.py, sin volver a entrenar ni usar Spark.

También concentra lo que comparten el reporte de revisión y el dashboard
ejecutivo: los niveles de riesgo y los nombres en español de las variables.
"""

import numpy as np
import pandas as pd

# Niveles de riesgo según prob_impago. PROPUESTA pendiente de aprobar con
# Eduardo (docs/fase5_revision_distribuciones.md, sección 4): si se cambian,
# basta con cambiar estos cortes y se actualizan el dashboard y el reporte.
CORTES_RIESGO = (0.10, 0.30)
NIVELES = ("Bajo", "Medio", "Alto")
RANGOS = ("< 10%", "10% – 30%", "≥ 30%")

# Nombres en español de las variables del modelo.
NOMBRES = {
    "age": "Edad",
    "interest_rate": "Tasa de interés",
    "months_employed": "Meses empleado",
    "dti": "DTI",
    "ingreso_mensual_fuente": "Ingreso (fuente)",
    "loan_amount": "Monto del préstamo",
    "has_cosigner": "Tiene aval",
    "loan_term": "Plazo",
    "has_dependents": "Tiene dependientes",
    "credit_score": "Score de crédito",
    "num_credit_lines": "Líneas de crédito",
    "has_mortgage": "Tiene hipoteca",
    "employment_type": "Empleo",
    "marital_status": "Estado civil",
    "loan_purpose": "Propósito",
    "education": "Escolaridad",
}
VALORES = {
    "Full-time": "tiempo completo",
    "Part-time": "medio tiempo",
    "Self-employed": "independiente",
    "Unemployed": "desempleado",
    "Married": "casado",
    "Single": "soltero",
    "Divorced": "divorciado",
    "Home": "vivienda",
    "Auto": "auto",
    "Business": "negocio",
    "Education": "educación",
    "Other": "otro",
    "High School": "preparatoria",
    "Bachelor's": "licenciatura",
    "Master's": "maestría",
    "PhD": "doctorado",
}


def nivel_riesgo(prob: pd.Series) -> pd.Series:
    """Bajo / Medio / Alto según CORTES_RIESGO (el corte cuenta para el nivel
    de arriba: 30% exacto es Alto)."""
    return pd.cut(
        prob,
        [-np.inf, *CORTES_RIESGO, np.inf],
        right=False,
        labels=list(NIVELES),
    )


def preparar_prestamos(loans: pd.DataFrame) -> pd.DataFrame:
    """Versión pandas de modelo_impago.preparar_variables: agrega el ingreso
    mensual de la fuente y el DTI calculado (misma fórmula que el KPI)."""
    loans = loans.copy()
    r = loans["interest_rate"] / 100 / 12
    n = loans["loan_term"]
    pago = loans["loan_amount"] * r / (1 - (1 + r) ** (-n))
    loans["ingreso_mensual_fuente"] = loans["income_anual"] / 12
    loans["dti"] = pago / loans["ingreso_mensual_fuente"]
    return loans


def nombre_variable(v: str, fila: pd.Series) -> str:
    """Nombre legible con el valor del cliente, p. ej. 'Edad: 23'."""
    if "=" in v:
        col, val = v.split("=", 1)
        estado = "sí" if fila[col] == val else "no"
        return f"{NOMBRES[col]} {VALORES.get(val, val)}: {estado}"
    valor = fila[v]
    if v in ("has_cosigner", "has_dependents", "has_mortgage"):
        texto = "sí" if valor == 1 else "no"
    elif v == "interest_rate":
        texto = f"{valor:.1f}%"
    elif v == "dti":
        texto = f"{valor:.2f}"
    elif v in ("loan_amount", "ingreso_mensual_fuente"):
        texto = f"{valor:,.0f} USD"
    elif v == "loan_term":
        texto = f"{valor:.0f} meses"
    else:
        texto = f"{valor:,.0f}"
    return f"{NOMBRES[v]}: {texto}"


def frase_factor(v: str, fila: pd.Series) -> str:
    """El factor en lenguaje de negocio, para alertas y el perfil del
    cliente (p. ej. 'Su pago mensual es 12.3 veces su ingreso')."""
    if "=" in v:
        # Se muestra el valor propio del cliente (p. ej. "Empleo: medio tiempo").
        col = v.split("=", 1)[0]
        propio = fila[col]
        if col == "employment_type" and propio == "Unemployed":
            return "Está desempleado"
        return f"{NOMBRES[col]}: {VALORES.get(propio, propio)}"
    x = fila[v]
    frases = {
        "dti": (
            f"Su pago mensual es {x:.1f} veces su ingreso"
            if x >= 1
            else f"Su pago mensual es el {x:.0%} de su ingreso"
        ),
        "months_employed": f"Lleva {x:.0f} meses en su empleo",
        "age": f"Tiene {x:.0f} años",
        "interest_rate": f"Tasa de interés de {x:.1f}%",
        "ingreso_mensual_fuente": f"Ingreso de {x:,.0f} USD al mes",
        "loan_amount": f"Préstamo de {x:,.0f} USD",
        "credit_score": f"Score de crédito de {x:.0f}",
        "loan_term": f"Plazo de {x:.0f} meses",
        "has_cosigner": "Sin aval" if x == 0 else "Tiene aval",
        "has_dependents": "Sin dependientes" if x == 0 else "Tiene dependientes",
        "has_mortgage": "Sin hipoteca" if x == 0 else "Tiene hipoteca",
        "num_credit_lines": (
            "1 línea de crédito" if x == 1 else f"{x:.0f} líneas de crédito"
        ),
    }
    return frases.get(v, nombre_variable(v, fila))


def matriz_variables(datos: pd.DataFrame, nombres: list) -> np.ndarray:
    """Arma las columnas en el orden del modelo. Las categóricas vienen como
    "education=PhD" y se vuelven 1/0."""
    columnas = []
    for nombre in nombres:
        if "=" in nombre:
            columna, valor = nombre.split("=", 1)
            columnas.append((datos[columna] == valor).astype(float).to_numpy())
        else:
            columnas.append(datos[nombre].astype(float).to_numpy())
    return np.column_stack(columnas)


def explicar(datos: pd.DataFrame, metricas: dict) -> tuple:
    """Devuelve (probabilidad, aportes) para cada renglón de `datos`.

    `datos` necesita las columnas del préstamo ya preparadas como en
    modelo_impago.preparar_variables (incluye ingreso_mensual_fuente y dti).
    `aportes` es un DataFrame con una columna por variable.
    """
    coef = metricas["coeficientes"]
    nombres = [c["variable"] for c in coef]
    pesos = np.array([c["coeficiente"] for c in coef])
    medias = np.array([c["media"] for c in coef])
    desviaciones = np.array([c["desviacion"] for c in coef])

    x = matriz_variables(datos, nombres)
    # Spark deja en 0 las variables sin variación (desviación 0).
    con_variacion = desviaciones > 0
    z = np.where(
        con_variacion, (x - medias) / np.where(con_variacion, desviaciones, 1), 0.0
    )
    aportes = z * pesos
    logit = metricas["intercepto"] + aportes.sum(axis=1)
    prob = 1 / (1 + np.exp(-logit))
    return prob, pd.DataFrame(aportes, columns=nombres, index=datos.index)
