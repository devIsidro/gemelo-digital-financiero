# Gemelo Digital Financiero — BBVA · Tecmilenio

Plataforma de Ingeniería de Datos (arquitectura Lakehouse Bronze-Silver-Gold) que sirve de motor
fundacional para construir un "Gemelo Digital Financiero": un sistema capaz de analizar el
comportamiento financiero de un usuario, evaluar su perfil de riesgo, simular escenarios futuros
y responder preguntas en lenguaje natural mediante IA generativa.

Proyecto individual — Path de Data Engineering, Tecmilenio en colaboración con BBVA.

## Arquitectura

```
Fuentes de datos -> Ingesta automatizada -> Bronze -> Silver -> Gold -> Modelos predictivos
                                                                      -> Asistente IA
                                                                      -> Dashboard ejecutivo
                                                                      -> Observabilidad
```

| Capa   | Propósito                                              |
|--------|---------------------------------------------------------|
| Bronze | Captura y preserva datos crudos (histórico auditable)   |
| Silver | Limpieza, normalización, deduplicación, reglas de negocio |
| Gold   | KPIs financieros, métricas de riesgo, datasets para IA  |

## Estructura del repositorio

```
.
├── dags/                  # DAGs de Apache Airflow (orquestación)
├── src/
│   ├── bronze_ingest/     # Scripts de ingesta cruda (Bronze)
│   ├── spark/             # Jobs de PySpark (transformaciones Silver/Gold)
│   └── ml/                # Entrenamiento e inferencia de modelos (riesgo, simulación)
├── config/                # Esquemas, parámetros y variables de configuración
├── tests/                 # Pruebas unitarias (pytest)
├── docker-compose.yml     # Infraestructura local reproducible
└── .github/workflows/     # CI: linters automáticos en cada Pull Request
```

## Estrategia de ramas (GitFlow)

- `main` — versión estable, lista para demo/entrega
- `develop` — integración de features en curso
- `feature/<nombre>` — una rama por funcionalidad (ej. `feature/bronze-ingest`, `feature/kpi-riesgo`)
- `hotfix/<nombre>` — correcciones urgentes sobre `main`

Flujo típico: crear `feature/...` desde `develop` → Pull Request hacia `develop` (el linter de CI
corre automático) → al cerrar una fase, merge de `develop` a `main`.

## Cómo levantar el entorno local

Requiere Docker Desktop (o Docker Engine + Compose plugin) corriendo.

```bash
cp .env.example .env        # ajusta variables si es necesario
docker compose up --build   # levanta Postgres, MinIO y Airflow (LocalExecutor)
```

Servicios disponibles:

- Airflow UI: http://localhost:8080 (usuario/clave definidos en `.env`)
- MinIO Console: http://localhost:9001 (almacenamiento tipo S3 simulado para Bronze/Silver/Gold)
- PostgreSQL: `localhost:5432` (metadata de Airflow + base analítica)

## Estado del proyecto

Ver `docs/` y el Canvas del proyecto para el detalle de alcance, KPIs y cronograma.
