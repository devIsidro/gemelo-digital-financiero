-- Se ejecuta automáticamente al primer arranque del contenedor de Postgres.
-- Crea la base de datos analítica (Capa Gold) separada de la metadata de Airflow.
CREATE DATABASE gemelo_gold;
