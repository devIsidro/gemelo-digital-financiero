"""
Modelo de probabilidad de impago (KPI prob_impago) — Fase 5.

Qué hace, en una sola corrida:
  1. Entrena una regresión logística (Spark MLlib) con los 255,347 préstamos
     de Loan Default (Silver), usando su etiqueta real de impago.
  2. La evalúa con un 20% de préstamos que el modelo no vio al entrenar.
  3. Le calcula la probabilidad de impago a cada cliente de tarjeta, con las
     características del préstamo que le asignó la llave sintética.

Decisiones (ver docs/fase5_modelo_impago.md):
  - Regresión logística en vez de gradient boosting (XGBoost/LightGBM, lo que
    decía el catálogo de Fase 1): con estos datos los dos dan prácticamente
    el mismo AUC (~0.75), y la logística es explicable — cada variable tiene
    un coeficiente que dice cuánto sube o baja el riesgo.
  - Variables: solo las del préstamo (fuente Loan Default). El ingreso es
    ingreso_mensual_fuente, NUNCA el ingreso simulado (decisión con Eduardo).
    El DTI es el mismo que el KPI ratio_endeudamiento (misma función). No se
    usa el DTIRatio del dataset.
  - El modelo se reentrena en cada corrida del DAG (tarda segundos). No se
    guarda en disco: así se evita el error de escritura de Spark en
    Docker/Windows con muchos archivos chiquitos. Sus coeficientes y métricas
    sí se guardan en un JSON para poder revisarlos.

IMPORTANTE: el préstamo asignado no es de esa persona (Opción A), así que
prob_impago es la probabilidad de un perfil de préstamo parecido, no la de un
cliente real.
"""

import json
from pathlib import Path

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.spark.gold_kpis import pago_mensual

SEMILLA = 42
PROPORCION_PRUEBA = 0.2

VARIABLES_NUMERICAS = [
    "age",
    "ingreso_mensual_fuente",
    "loan_amount",
    "credit_score",
    "months_employed",
    "num_credit_lines",
    "interest_rate",
    "loan_term",
    "dti",
    "has_mortgage",
    "has_dependents",
    "has_cosigner",
]
VARIABLES_CATEGORICAS = [
    "education",
    "employment_type",
    "marital_status",
    "loan_purpose",
]


def build_spark(app_name: str = "modelo_impago") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


def preparar_variables(loans_df: DataFrame) -> DataFrame:
    """Agrega las variables derivadas: ingreso mensual de la fuente y DTI.

    El DTI se calcula exactamente igual que el KPI ratio_endeudamiento
    (pago_mensual de gold_kpis.py / ingreso mensual de la fuente).
    """
    ingreso = F.col("income_anual") / F.lit(12.0)
    pago = pago_mensual(
        F.col("loan_amount"), F.col("interest_rate"), F.col("loan_term")
    )
    return loans_df.withColumn("ingreso_mensual_fuente", ingreso).withColumn(
        "dti", pago / ingreso
    )


def build_pipeline() -> Pipeline:
    """Categóricas -> one-hot; todo junto -> estandarizado -> logística.

    Se estandariza para que los coeficientes sean comparables entre sí:
    cada uno dice cuánto cambia el riesgo al subir esa variable una
    desviación estándar.
    """
    indexadores = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in VARIABLES_CATEGORICAS
    ]
    one_hot = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in VARIABLES_CATEGORICAS],
        outputCols=[f"{c}_oh" for c in VARIABLES_CATEGORICAS],
        handleInvalid="keep",
    )
    ensamblador = VectorAssembler(
        inputCols=VARIABLES_NUMERICAS + [f"{c}_oh" for c in VARIABLES_CATEGORICAS],
        outputCol="variables",
    )
    escalador = StandardScaler(
        inputCol="variables", outputCol="variables_std", withMean=True, withStd=True
    )
    logistica = LogisticRegression(
        featuresCol="variables_std", labelCol="is_default", maxIter=100
    )
    return Pipeline(stages=[*indexadores, one_hot, ensamblador, escalador, logistica])


def entrenar(loans_df: DataFrame) -> tuple:
    """Entrena con el 80% de los préstamos y evalúa con el 20% restante.

    Devuelve (modelo, métricas).
    """
    datos = preparar_variables(loans_df)
    entrenamiento, prueba = datos.randomSplit(
        [1 - PROPORCION_PRUEBA, PROPORCION_PRUEBA], seed=SEMILLA
    )

    modelo = build_pipeline().fit(entrenamiento)

    evaluador = BinaryClassificationEvaluator(
        labelCol="is_default",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    pred_prueba = modelo.transform(prueba)
    pred_entrenamiento = modelo.transform(entrenamiento)

    pred_prueba = pred_prueba.withColumn(
        "prob", vector_to_array("probability").getItem(1)
    )
    resumen = pred_prueba.agg(
        F.avg("is_default").alias("tasa_real"), F.avg("prob").alias("tasa_predicha")
    ).collect()[0]

    metricas = {
        "algoritmo": "Regresión logística (Spark MLlib)",
        "prestamos_entrenamiento": entrenamiento.count(),
        "prestamos_prueba": prueba.count(),
        "auc_prueba": round(evaluador.evaluate(pred_prueba), 4),
        "auc_entrenamiento": round(evaluador.evaluate(pred_entrenamiento), 4),
        "tasa_impago_real_prueba": round(resumen["tasa_real"], 4),
        "tasa_impago_predicha_prueba": round(resumen["tasa_predicha"], 4),
        "coeficientes": coeficientes(modelo, prueba.limit(1)),
    }
    return modelo, metricas


def coeficientes(modelo: PipelineModel, ejemplo_df: DataFrame) -> list:
    """Coeficiente de cada variable (estandarizada), de mayor a menor efecto.

    Positivo = sube el riesgo de impago; negativo = lo baja. Los nombres se
    leen de los metadatos que Spark guarda en el vector de variables, para
    no depender del orden en que el one-hot acomoda las categorías.
    """
    atributos = (
        modelo.transform(ejemplo_df).schema["variables"].metadata["ml_attr"]["attrs"]
    )
    nombres = {}
    for grupo in atributos.values():
        for a in grupo:
            # "education_oh_PhD" -> "education=PhD"
            nombres[a["idx"]] = a["name"].replace("_oh_", "=")
    valores = modelo.stages[-1].coefficients.toArray().tolist()
    pares = [
        {"variable": nombres.get(i, f"variable_{i}"), "coeficiente": round(v, 4)}
        for i, v in enumerate(valores)
        # La columna para categorías desconocidas siempre vale 0 al entrenar,
        # así que su coeficiente es 0 y no aporta nada al reporte.
        if not nombres.get(i, "").endswith("=__unknown")
    ]
    return sorted(pares, key=lambda p: abs(p["coeficiente"]), reverse=True)


def calcular_prob_impago(
    modelo: PipelineModel, asignacion_df: DataFrame, loans_df: DataFrame
) -> DataFrame:
    """prob_impago por cliente, con las características de su préstamo asignado."""
    datos = preparar_variables(
        asignacion_df.select("cc_num", "loan_id").join(
            loans_df, on="loan_id", how="inner"
        )
    )
    return modelo.transform(datos).select(
        "cc_num",
        "loan_id",
        F.round(vector_to_array("probability").getItem(1), 4).alias("prob_impago"),
    )


def run_modelo_job(
    loans_path: str = "data/silver/loan_default",
    asignacion_path: str = "data/gold/asignacion_prestamos",
    salida_path: str = "data/gold/prob_impago_cliente",
    metricas_path: str = "data/gold/modelo_impago_metricas.json",
) -> dict:
    """Entrena, evalúa, calcula prob_impago por cliente y escribe resultados.

    Falla a propósito si el modelo sale peor que una regla aleatoria
    (AUC < 0.6) o si algún cliente se queda sin probabilidad.
    """
    spark = build_spark()
    try:
        loans_df = spark.read.parquet(loans_path)
        asignacion_df = spark.read.parquet(asignacion_path)

        modelo, metricas = entrenar(loans_df)
        if metricas["auc_prueba"] < 0.6:
            raise ValueError(
                f"El modelo de impago salió con AUC {metricas['auc_prueba']}; "
                "algo cambió en los datos de préstamos."
            )

        prob_df = calcular_prob_impago(modelo, asignacion_df, loans_df)
        clientes = asignacion_df.count()
        con_prob = prob_df.filter(F.col("prob_impago").isNotNull()).count()
        if con_prob != clientes:
            raise ValueError(
                f"{clientes} clientes con préstamo asignado pero solo {con_prob} "
                "con probabilidad de impago."
            )

        # coalesce(1): tabla chica, un solo archivo (ver silver_transactions.py).
        prob_df.coalesce(1).write.mode("overwrite").parquet(salida_path)

        resumen = (
            spark.read.parquet(salida_path)
            .agg(
                F.expr("percentile_approx(prob_impago, 0.5)").alias("mediana"),
                F.min("prob_impago").alias("minima"),
                F.max("prob_impago").alias("maxima"),
            )
            .collect()[0]
        )
        metricas["clientes"] = {
            "total": clientes,
            "prob_impago_mediana": resumen["mediana"],
            "prob_impago_minima": resumen["minima"],
            "prob_impago_maxima": resumen["maxima"],
        }

        Path(metricas_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metricas_path).write_text(
            json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return metricas
    finally:
        spark.stop()


if __name__ == "__main__":
    m = run_modelo_job()
    print(
        f"Modelo de impago: AUC {m['auc_prueba']} en prueba "
        f"({m['prestamos_prueba']:,} préstamos) | {m['clientes']['total']} clientes, "
        f"prob_impago mediana {m['clientes']['prob_impago_mediana']:.1%}"
    )
