# Extiende la imagen oficial de Airflow para poder correr el job de
# PySpark de la Capa Silver (Fase 3) dentro del mismo contenedor.
FROM apache/airflow:2.10.2-python3.11

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jdk-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java

USER airflow
RUN pip install --no-cache-dir pyspark==4.2.0 pyarrow pandas
