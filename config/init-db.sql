-- Se ejecuta automáticamente al primer arranque del contenedor de Postgres.
-- Crea la base de datos analítica (Capa Gold) separada de la metadata de Airflow.
-- Por ahora Gold se guarda en Parquet (./data/gold); esta base queda lista para cargarlo.
CREATE DATABASE gemelo_gold;
