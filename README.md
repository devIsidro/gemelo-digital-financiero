# Gemelo Digital Financiero — BBVA · Tecmilenio

Plataforma de Ingeniería de Datos (arquitectura Lakehouse Bronze-Silver-Gold) que sirve de motor
fundacional para construir un "Gemelo Digital Financiero": un sistema capaz de analizar el
comportamiento financiero de un usuario, evaluar su perfil de riesgo, simular escenarios futuros
y responder preguntas en lenguaje natural mediante IA generativa.

Proyecto individual — Path de Data Engineering, Tecmilenio en colaboración con BBVA.
Uso académico con datos públicos de Kaggle; ningún dato pertenece a una persona real.

## Arquitectura

```
Fuentes de datos -> Ingesta automatizada -> Bronze -> Silver -> Gold -> Modelos predictivos
                                                                      -> Asistente IA
                                                                      -> Dashboard ejecutivo
                                                                      -> Observabilidad
```

| Capa   | Propósito                                                 | En este proyecto |
|--------|-----------------------------------------------------------|------------------|
| Bronze | Captura y preserva datos crudos (histórico auditable)     | Transacciones con tarjeta y préstamos, tal cual vienen de Kaggle |
| Silver | Limpieza, normalización, deduplicación, reglas de negocio | 555,719 transacciones y 255,347 préstamos validados |
| Gold   | KPIs financieros, métricas de riesgo, datasets para IA    | 924 clientes con KPIs, préstamo asignado y probabilidad de impago |

Diagramas: `docs/diagrama_infraestructura.png` y `docs/diagrama_flujo_datos.png`.

### Fuentes de datos

| Dataset (Kaggle) | Archivo esperado | Para qué se usa |
|---|---|---|
| Credit Card Transactions Fraud Detection | `data/raw/fraudTest.csv` | Gasto real de cada cliente |
| Loan Default Prediction | `data/raw/Loan_default.csv` | Préstamos, score, DTI y modelo de impago |

Los datasets no se suben al repositorio (pesan mucho y están en `.gitignore`): hay que
descargarlos de Kaggle y ponerlos en `data/raw/`.

## Pipeline (DAG `bronze_ingest` de Airflow)

Corre todos los días (`@daily`) o a mano desde la interfaz de Airflow. Tiene 10 pasos:

| Rama | Pasos |
|---|---|
| Transacciones | `ingest_source` → `clean_silver` → `validate_quality` → `build_gold_perfil` → `simular_ingreso_credito` |
| Préstamos | `ingest_loan_default` → `clean_silver_loans` |
| Gold y modelo | `asignar_prestamos` → `calcular_kpis`, y `modelo_prob_impago` |

Las transformaciones corren con PySpark 4.2 dentro del contenedor de Airflow
(`Dockerfile`). Una corrida completa tarda alrededor de 5 minutos.

## Dashboards

| | Panel Operativo | Dashboard Ejecutivo |
|---|---|---|
| Dirección | http://localhost:8501 | http://localhost:8502 |
| Para | Equipo de Data Engineering | Negocio: dirección, riesgos, producto |
| Responde | ¿La plataforma funciona bien? | ¿Cómo están los clientes? |
| Muestra | Estado de los 10 pasos, historial de corridas de Airflow, tasa de éxito, duración por paso, reglas de calidad | Portafolio por nivel de riesgo, ingresos vs gastos, mapa de calor de compras, consumo esencial vs discrecional, alertas tempranas y perfil 360 de cada cliente |
| Código | `src/dashboard/operativo.py` | `src/dashboard/ejecutivo.py` |

Ambos usan la Guía de Estilo de BBVA (`src/dashboard/estilo.py`) y leen la capa Gold de `./data`,
así que primero hay que correr el DAG completo.

## Cómo levantar el entorno local

Requiere Docker Desktop (o Docker Engine + Compose plugin) corriendo.

```bash
cp .env.example .env        # ajusta variables si es necesario
docker compose up -d --build
```

Luego, en Airflow, activa y corre el DAG `bronze_ingest`. Servicios disponibles:

- Airflow UI: http://localhost:8080 (usuario/clave definidos en `.env`)
- Panel Operativo: http://localhost:8501
- Dashboard Ejecutivo: http://localhost:8502
- MinIO Console: http://localhost:9001 (almacenamiento tipo S3 simulado)
- PostgreSQL: `localhost:5432` (metadata de Airflow + base analítica)

Si cambias el código de un dashboard, reconstrúyelo con
`docker compose up -d --build dashboard dashboard-ejecutivo`.

## Pruebas y calidad de código

```bash
pip install -r requirements.txt
python -m pytest tests            # pruebas unitarias (Spark, modelo, dashboards, DAG)
flake8 src dags tests --max-line-length=100
black --check src dags tests
```

El linter (flake8 + black) corre automático en cada Pull Request (`.github/workflows/lint.yml`).

## Estructura del repositorio

```
.
├── dags/                  # DAG de Apache Airflow (orquestación)
├── src/
│   ├── bronze_ingest/     # Ingesta cruda (Bronze) y data profiling
│   ├── spark/             # Jobs de PySpark (Silver y Gold) y validaciones de calidad
│   ├── ml/                # Modelo de probabilidad de impago y su explicación
│   ├── dashboard/         # Panel Operativo y Dashboard Ejecutivo (Streamlit)
│   └── reportes/          # Gráficas de los documentos de revisión
├── config/                # Catálogo de KPIs y scripts de base de datos
├── docs/                  # Decisiones, diseño y revisiones de cada fase
├── tests/                 # Pruebas unitarias (pytest)
├── Dockerfile             # Imagen de Airflow con PySpark
├── Dockerfile.dashboard   # Imagen de los dashboards
├── docker-compose.yml     # Infraestructura local reproducible
└── .github/workflows/     # CI: linters automáticos en cada Pull Request
```

## Estado del proyecto

| Fase | Estado | Documentación |
|---|---|---|
| 1. Kick off y diseño | Completa | `config/kpis.yaml`, diagramas en `docs/` |
| 2. Ingesta y capa Bronze | Completa | `docs/entregables/` (diccionario de datos) |
| 3. Silver, calidad y panel operativo | Completa | `src/spark/validate_silver.py`, `src/dashboard/operativo.py` |
| 4. Capa Gold, KPIs y dashboard ejecutivo | Completa (v1) | `docs/fase4_diseno_capa_gold.md`, `docs/fase4_datos_simulados_ingreso_credito.md`, `docs/fase4_llave_sintetica_prestamos.md`, `docs/fase4_decisiones_kpis.md`, `docs/fase4_dashboard_ejecutivo.md` |
| 5. IA y simulación | En curso: modelo de impago listo; siguen el simulador Monte Carlo y el asistente de IA | `docs/fase5_modelo_impago.md`, `docs/fase5_revision_distribuciones.md` |
| 6. Observabilidad y hardening | Pendiente | — |
| 7. Cierre ejecutivo | Pendiente | — |

Decisiones clave tomadas con BBVA (detalle en `docs/`):

- **Opción A:** a cada cliente de transacciones se le asigna un ingreso simulado y un préstamo
  del dataset Loan Default con una llave sintética "por parecido", documentada y reproducible.
- **Dos ingresos separados:** el simulado para ahorro y flujo; el de la fuente para DTI y riesgo.
- **DTI propio:** pago mensual del préstamo / ingreso mensual, con fórmula documentada.
- **Modelo explicable:** regresión logística (AUC 0.75, bien calibrada) que permite decir por qué
  cada cliente tiene su probabilidad de impago.

## Estrategia de ramas (GitFlow)

- `main` — versión estable, lista para demo/entrega
- `develop` — integración de features en curso
- `feature/<nombre>` — una rama por funcionalidad (ej. `feature/bronze-ingest`, `feature/kpi-riesgo`)
- `hotfix/<nombre>` — correcciones urgentes sobre `main`

Flujo típico: crear `feature/...` desde `develop` → Pull Request hacia `develop` (el linter de CI
corre automático) → al cerrar una fase, merge de `develop` a `main`.
