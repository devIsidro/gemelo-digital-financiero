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
  propuesta de `docs/fase5_revision_distribuciones.md`, aprobada por
  Eduardo el 9 oct 2026. Si hubiera que cambiarlos, se cambian en un solo
  lugar (`CORTES_RIESGO` en `src/ml/explicacion.py`).
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

## Simulador Monte Carlo (agregado en la Fase 5)

- **Portafolio:** porcentaje de clientes en riesgo de números rojos en cada
  escenario (sin cambios, crédito hipotecario, cambio de empleo y abrir un
  negocio), con el ahorro a 12 meses y la capacidad de ahorro del año 1.
- **Perfil 360:** simulador interactivo: se elige el escenario, se mueven
  sus parámetros y se ven al momento las bandas del ahorro a 36 meses, el
  flujo de efectivo proyectado y el riesgo frente a no hacer cambios.

Detalle en `docs/fase5_simulador_monte_carlo.md`.

## Reporte ejecutivo en PDF (agregado después del cierre)

Botón «Descargar reporte (PDF)» arriba a la derecha de la pestaña Portafolio.
Genera un reporte de dos páginas del segmento elegido (todos, bajo, medio o
alto), pensado para compartirlo con un comité sin abrir el dashboard:

- Página 1: KPIs del segmento, hallazgos clave, niveles de riesgo del
  portafolio con el impago real de préstamos parecidos y el impacto de cada
  escenario del simulador.
- Página 2: alertas tempranas (los 15 clientes con mayor probabilidad de
  impago y lo que más sube su riesgo) y cómo leer el reporte.

Decisiones:

- **Mismas cifras que la pantalla:** el PDF usa las funciones de
  `datos_ejecutivo.py` (`resumen_portafolio`, `tabla_niveles`, `alertas` y
  `escenarios_portafolio`), no hace cálculos propios.
- **Hallazgos calculados, no escritos a mano:** las frases se arman con los
  datos de cada corrida (`hallazgos()` en `src/reportes/reporte_ejecutivo.py`).
- **fpdf2:** genera el PDF dibujando texto y rectángulos, sin navegador ni
  librerías del sistema, así que cabe en la imagen liviana del dashboard.
  Usa las fuentes estándar del PDF, por eso los símbolos fuera de Latin-1
  (≥, −, →) se cambian por su equivalente (>=, -, ->).
- **Mismo cuidado con los datos personales:** solo aparecen los
  identificadores anónimos («Cliente 0001»), nunca el número de tarjeta.

Pruebas en `tests/reportes/test_reporte_ejecutivo.py`. Después de actualizar
el código hay que reconstruir la imagen:
`docker compose up -d --build dashboard dashboard-ejecutivo`.

## Asistente de IA (agregado en la Fase 5)

Pestaña «Asistente IA»: chat sobre un cliente en lenguaje natural, con Llama 3
en Ollama y Text-to-SQL sobre Gold. Detalle en `docs/fase5_asistente_ia.md`.

La v1 se presentó a Eduardo el 9 oct 2026 y dio su visto bueno.
