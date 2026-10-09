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
"""

import numpy as np
import pandas as pd


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
