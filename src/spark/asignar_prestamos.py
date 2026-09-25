"""
Job de PySpark: llave sintética entre clientes de tarjeta y préstamos del
dataset Loan Default (Fase 4 — Opción A).

Los dos datasets no comparten clientes. Aquí se le asigna a cada cliente
(cc_num) un préstamo (loan_id) con una regla fija y documentada — no al
azar — en docs/fase4_llave_sintetica_prestamos.md:

  "Por parecido": se ordena a los clientes por su gasto mensual real con
  tarjeta y a los préstamos por el ingreso de su titular. El cliente que
  está en el percentil p de gasto recibe el préstamo que está en el
  percentil p de ingreso. Así, quien más gasta recibe a alguien que más
  gana, y los datos unidos quedan coherentes.

Propiedades (verificadas en tests/spark/test_asignar_prestamos.py):
  - Reproducible: mismos datos de entrada -> misma asignación siempre.
  - Uno a uno: ningún préstamo se asigna a dos clientes.
  - Todos los clientes reciben un préstamo.

IMPORTANTE: el préstamo asignado NO es de esa persona. Es una construcción
documentada para poder calcular los KPIs de riesgo.
"""

import sys

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def build_spark(app_name: str = "asignar_prestamos") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


def asignar_prestamos(perfil_df: DataFrame, loans_df: DataFrame) -> DataFrame:
    """Devuelve una fila por cliente: cc_num, loan_id y los percentiles usados.

    perfil_df: gold_perfil_cliente (necesita cc_num, gasto_promedio_mensual).
    loans_df: Silver de préstamos (necesita loan_id, income_anual).
    """
    num_clientes = perfil_df.count()
    num_prestamos = loans_df.count()
    if num_prestamos < num_clientes:
        raise ValueError(
            f"Hay {num_clientes} clientes pero solo {num_prestamos} préstamos; "
            "no alcanza para una asignación uno a uno."
        )

    # El segundo criterio de orden (cc_num / loan_id) rompe empates para que
    # el resultado sea siempre el mismo.
    orden_clientes = Window.orderBy("gasto_promedio_mensual", "cc_num")
    clientes = perfil_df.select(
        "cc_num",
        F.row_number().over(orden_clientes).alias("posicion_gasto"),
    ).withColumn(
        # Percentil del cliente, a la mitad de su "escalón": con 4 clientes
        # quedan en 0.125, 0.375, 0.625, 0.875 (nunca en 0 ni en 1).
        "percentil_gasto",
        (F.col("posicion_gasto") - F.lit(0.5)) / F.lit(float(num_clientes)),
    )
    clientes = clientes.withColumn(
        "posicion_prestamo",
        F.floor(F.col("percentil_gasto") * F.lit(num_prestamos)).cast("long") + 1,
    )

    orden_prestamos = Window.orderBy("income_anual", "loan_id")
    prestamos = loans_df.select(
        "loan_id",
        F.row_number().over(orden_prestamos).cast("long").alias("posicion_prestamo"),
    )

    return clientes.join(prestamos, on="posicion_prestamo", how="inner").select(
        "cc_num",
        "loan_id",
        F.round("percentil_gasto", 4).alias("percentil_gasto"),
        F.round(
            (F.col("posicion_prestamo") - F.lit(0.5)) / F.lit(float(num_prestamos)), 4
        ).alias("percentil_ingreso"),
    )


def run_asignacion_job(
    perfil_path: str = "data/gold/perfil_cliente",
    loans_path: str = "data/silver/loan_default",
    salida_path: str = "data/gold/asignacion_prestamos",
) -> dict:
    """Lee el perfil de clientes y los préstamos, asigna y escribe."""
    spark = build_spark()
    try:
        perfil_df = spark.read.parquet(perfil_path)
        loans_df = spark.read.parquet(loans_path)

        asignacion = asignar_prestamos(perfil_df, loans_df)
        num_asignados = asignacion.count()
        clientes = perfil_df.count()
        prestamos_distintos = asignacion.select("loan_id").distinct().count()

        if not (num_asignados == clientes == prestamos_distintos):
            raise ValueError(
                f"Asignación incompleta: {clientes} clientes, {num_asignados} "
                f"asignados, {prestamos_distintos} préstamos distintos."
            )

        # coalesce(1): tabla chica, un solo archivo (ver silver_transactions.py).
        asignacion.coalesce(1).write.mode("overwrite").parquet(salida_path)
        return {"clientes_asignados": num_asignados}
    finally:
        spark.stop()


if __name__ == "__main__":
    perfil_arg = sys.argv[1] if len(sys.argv) > 1 else "data/gold/perfil_cliente"
    loans_arg = sys.argv[2] if len(sys.argv) > 2 else "data/silver/loan_default"
    salida_arg = sys.argv[3] if len(sys.argv) > 3 else "data/gold/asignacion_prestamos"

    r = run_asignacion_job(perfil_arg, loans_arg, salida_arg)
    print(
        f"Llave sintética: {r['clientes_asignados']:,} clientes con préstamo "
        f"asignado en {salida_arg}/ (asignación SIMULADA, Opción A)"
    )
