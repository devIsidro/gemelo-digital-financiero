# Fase 4 — Dashboard Ejecutivo v1

Entregable de la semana 16 del plan ("Dashboard ejecutivo v1 — publicar
indicadores"), basado en la sección 12 del documento del proyecto
("Dashboard Ejecutivo de Negocio e Impacto Financiero").

- Código: `src/dashboard/ejecutivo.py` (vista) y
  `src/dashboard/datos_ejecutivo.py` (cálculos, con pruebas en
  `tests/dashboard/test_datos_ejecutivo.py`).
- Se abre en **http://localhost:8502** (servicio `dashboard-ejecutivo` de
  `docker-compose.yml`). Usa la misma imagen y el mismo estilo BBVA que el
  Panel Operativo (http://localhost:8501).

## Para quién y qué responde

| | Panel Operativo | Dashboard Ejecutivo |
|---|---|---|
| Para | Equipo de Data Engineering | Negocio: dirección, riesgos, producto |
| Pregunta | ¿La plataforma funciona bien? | ¿Cómo están los clientes? |
| Lee de | Bronze, Silver, Gold y Airflow | Gold (y Silver para el histórico) |

## Contenido

**Pestaña Portafolio** (panel conectado a Gold con métricas agregadas del
portafolio). Filtro por segmento de riesgo (todos, bajo, medio, alto):

- KPIs: clientes, gasto mensual típico, capacidad de ahorro, endeudamiento
  (DTI) y probabilidad de impago (medianas).
- Niveles de riesgo, con la tasa de impago real de préstamos parecidos en
  el dataset para validar los cortes.
- Ingresos vs gastos por mes (módulo histórico).
- Mapa de calor de compras por día y hora (módulo histórico).
- Consumo por categoría, clasificado en esencial y discrecional (módulo
  histórico).
- Alertas tempranas: clientes en riesgo alto con el factor que más sube su
  riesgo (módulo predictivo).

**Pestaña Perfil 360 del cliente** (vista orientada al cliente final):

- KPIs del cliente comparados con la mediana del portafolio.
- Sus ingresos vs gastos por mes y en qué gasta.
- Su préstamo asignado y por qué el modelo le da esa probabilidad de
  impago (aporte de cada factor, `src/ml/explicacion.py`).

## Decisiones de la v1

- **Niveles de riesgo:** bajo < 10%, medio 10% – 30%, alto ≥ 30%. Es la
  propuesta de `docs/fase5_revision_distribuciones.md`, pendiente de validar
  con Eduardo. Se cambian en un solo lugar (`CORTES_RIESGO` en
  `src/ml/explicacion.py`).
- **Clientes anónimos:** se muestran como "Cliente 0001", … (orden del
  número de tarjeta). El número de tarjeta nunca aparece.
- **Ingresos vs gastos:** el gasto es real (Silver) y el ingreso es el
  simulado (Opción A), por eso es constante cada mes. Se muestra el
  promedio por cliente. Junio no aparece porque el dataset empieza el
  21 de junio y el mes saldría con gasto falsamente bajo.
- **Clasificación de consumo:** regla fija por categoría. Esencial:
  supermercado, gasolina y transporte, hogar, salud y ejercicio, niños y
  mascotas. Discrecional: el resto. El dataset no trae comercios reales ni
  cargos recurrentes, así que no se pueden detectar suscripciones.
- **Gasto con fraude:** se incluye, igual que en los KPIs de Gold, para que
  las cifras cuadren entre la capa Gold y el dashboard.
- **Explicación del riesgo:** necesita las métricas del modelo de octubre
  de 2026 (intercepto, medias y desviaciones). Si el DAG no se ha vuelto a
  correr, el dashboard funciona igual pero sin esa parte y lo avisa.

## Pendiente (siguientes fases del plan)

- Proyecciones del simulador Monte Carlo (semana 19).
- Chat con el asistente de IA dentro del dashboard (semana 20).
- Validar con Eduardo los cortes de riesgo y el contenido de la v1.
