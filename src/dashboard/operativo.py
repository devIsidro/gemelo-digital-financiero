"""
Panel Operativo del Gemelo Digital Financiero (Fase 3, rediseñado en Fase 4
con la Guía de Estilo de BBVA).

Vista para el equipo de Data Engineering: muestra la salud del pipeline
completo, de los datos crudos a los KPIs y el modelo de impago:

  1. Estado general y cifras principales.
  2. Los 10 pasos del DAG bronze_ingest en su última corrida e historial de
     corridas (de aquí sale el KPI tasa_exito_ingesta del catálogo).
  3. Validaciones de calidad de transacciones y préstamos.
  4. Volumen de transacciones por día y por categoría.
  5. Salud de la capa Gold y del modelo de impago.

Es distinto del Dashboard Ejecutivo (src/dashboard/ejecutivo.py), que es para negocio.

Corre con: streamlit run src/dashboard/operativo.py
"""

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dashboard import datos  # noqa: E402
from src.dashboard import estilo as e  # noqa: E402

RUTAS = datos.Rutas(Path("data"))

st.set_page_config(
    page_title="Panel Operativo — Gemelo Digital Financiero",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.html(e.CSS)


# ---------------------------------------------------------------------------
# Carga de datos (en caché 60 s para que el panel responda rápido)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60, show_spinner=False)
def cargar_conteos():
    return datos.conteos(RUTAS)


@st.cache_data(ttl=60, show_spinner=False)
def cargar_transacciones():
    silver = datos.transacciones_silver(RUTAS)
    return (
        datos.volumen_por_dia(silver),
        datos.volumen_por_categoria(silver),
        datos.pct_fraude(silver),
    )


@st.cache_data(ttl=60, show_spinner=False)
def cargar_validaciones():
    tx = datos.validar_transacciones(RUTAS)
    loans = (
        datos.validar_prestamos(datos.prestamos_silver(RUTAS))
        if RUTAS.silver_loans.exists()
        else None
    )
    return tx, loans


@st.cache_data(ttl=60, show_spinner=False)
def cargar_gold():
    return datos.resumen_gold(RUTAS)


@st.cache_data(ttl=30, show_spinner=False)
def cargar_airflow():
    """(corridas, pasos de la última corrida, error). Si Airflow no responde,
    regresa el error y el resto del panel sigue funcionando."""
    try:
        url = datos.url_airflow()
        corridas = datos.corridas(url)
        pasos = (
            datos.pasos_de_corrida(url, corridas.iloc[-1]["run_id"])
            if len(corridas)
            else None
        )
        return corridas, pasos, None
    except Exception as exc:  # noqa: BLE001
        return None, None, str(exc).splitlines()[0][:160]


def html(contenido: str):
    st.markdown(contenido, unsafe_allow_html=True)


def con_tema(chart):
    t = e.tema_altair()["config"]
    return (
        chart.configure(font=t["font"], background=t["background"])
        .configure_view(stroke=None)
        .configure_axis(**t["axis"])
        .configure_legend(**t["legend"])
    )


MESES = (
    "['ene','feb','mar','abr','may','jun','jul','ago','sep','oct','nov','dic']"
    "[month(datum.value)] + ' ' + year(datum.value)"
)


def nombre_estado(estado: str) -> str:
    return e.ESTADOS.get(estado, (estado,))[0]


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

try:
    conteos = cargar_conteos()
    por_dia, por_categoria, pct_fraude = cargar_transacciones()
    val_tx, val_loans = cargar_validaciones()
    error_datos = None
except Exception as exc:  # noqa: BLE001
    error_datos = str(exc).splitlines()[0][:200]

if error_datos:
    html(
        e.aviso(
            "<b>No se pudieron leer los datos de Bronze/Silver.</b> "
            "¿Ya corriste el DAG <code>bronze_ingest</code> al menos una vez? "
            f"Detalle: {error_datos}"
        )
    )
    st.stop()

corridas, pasos, error_airflow = cargar_airflow()
gold = cargar_gold()

# ---------------------------------------------------------------------------
# Estado general
# ---------------------------------------------------------------------------

hay_corridas = corridas is not None and len(corridas) > 0
resumen = datos.resumen_corridas(corridas) if hay_corridas else None
ultima = resumen["ultima"] if resumen else None
validaciones_ok = all(r.ok for r in val_tx) and (
    val_loans is None or all(r.ok for r in val_loans)
)
ultima_terminada = None
if hay_corridas:
    terminadas = corridas[corridas["state"].isin(["success", "failed"])]
    ultima_terminada = terminadas.iloc[-1] if len(terminadas) else None

motivos = []
if not validaciones_ok:
    motivos.append("Hay validaciones de calidad que no pasaron.")
if ultima_terminada is not None and ultima_terminada["state"] == "failed":
    motivos.append(
        "La última corrida terminada del DAG falló. Revisa abajo el paso marcado "
        "como «Falló» y su registro en Airflow."
    )

if motivos:
    estado_general = e.pill("Requiere atención", e.MANDARIN, "▲")
elif error_airflow:
    estado_general = e.pill("Datos sanos · sin conexión a Airflow", e.CANARY, "●")
elif ultima is not None and ultima["state"] in ("running", "queued"):
    estado_general = e.pill("Pipeline sano · corrida en curso", e.LIME, "✓")
else:
    estado_general = e.pill("Pipeline sano", e.LIME, "✓")

if ultima is not None and pd.notna(ultima["inicio"]):
    meta = f"Última corrida: {ultima['inicio']:%d/%m/%Y %H:%M} h · {ultima['tipo']}"
else:
    meta = "Última corrida: sin datos de Airflow"
ahora = pd.Timestamp.now(tz=datos.ZONA_HORARIA)

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------

izq, der = st.columns([3, 2], vertical_alignment="bottom")
with izq:
    html(
        '<div class="gd-eyebrow">Panel operativo · Data Engineering</div>'
        '<div class="gd-h1">Gemelo Digital Financiero</div>'
        '<p class="gd-sub">Salud del pipeline de datos: de los datos crudos a los '
        "KPIs financieros y el modelo de riesgo.</p>"
    )
with der:
    html(
        '<div class="gd-header-der">'
        '<span class="gd-tag">Uso académico · datos simulados</span>'
        f"{estado_general}"
        f'<div class="gd-meta">{meta}<br>Actualizado: {ahora:%H:%M} h</div></div>'
    )

for m in motivos:
    html(e.aviso(f"<b>Atención:</b> {m}"))

st.write("")

# ---------------------------------------------------------------------------
# Cifras principales
# ---------------------------------------------------------------------------

c = conteos
desc_tx = (c["bronze_tx"] or 0) - (c["silver_tx"] or 0)
desc_loans = (c["bronze_loans"] or 0) - (c["silver_loans"] or 0)

if resumen and resumen["tasa_exito"] is not None:
    tasa_valor = f"{resumen['tasa_exito']:.0f}%"
    tasa_nota = f"{resumen['exitosas']} de {resumen['terminadas']} corridas terminadas"
    tasa_acento = e.LIME if resumen["tasa_exito"] >= 90 else e.MANDARIN
else:
    tasa_valor, tasa_nota, tasa_acento = "—", "Sin conexión a Airflow", e.GRIS_3

if resumen and resumen["duracion_tipica"] is not None:
    dur_valor = f"{resumen['duracion_tipica']:.1f} min"
    dur_nota = "Mediana de las corridas exitosas"
else:
    dur_valor, dur_nota = "—", "Sin conexión a Airflow"

tarjetas = [
    e.kpi(
        "Transacciones en Silver",
        e.formato_entero(c["silver_tx"]),
        f"De {e.formato_entero(c['bronze_tx'])} en Bronze · {desc_tx:,} descartadas",
        "datos",
    ),
    e.kpi(
        "Préstamos en Silver",
        e.formato_entero(c["silver_loans"]),
        f"De {e.formato_entero(c['bronze_loans'])} en Bronze · "
        f"{desc_loans:,} descartados",
        "prestamo",
    ),
    e.kpi("Tasa de éxito de corridas", tasa_valor, tasa_nota, "flujo", tasa_acento),
    e.kpi("Duración típica del pipeline", dur_valor, dur_nota, "reloj"),
    e.kpi(
        "Clientes con perfil completo",
        e.formato_entero(c["clientes_gold"]),
        "Con KPIs, préstamo asignado y riesgo",
        "personas",
    ),
]
for i, (col, tarjeta) in enumerate(zip(st.columns(5, gap="medium"), tarjetas)):
    with col:
        with st.container(key=f"caja_kpi_{i}"):
            html(tarjeta)

st.write("")

# ---------------------------------------------------------------------------
# Pipeline: última corrida e historial
# ---------------------------------------------------------------------------

col_pipe, col_hist = st.columns([3, 2], gap="medium")

with col_pipe:
    with st.container(key="caja_pipeline"):
        if pasos is not None and ultima is not None:
            desc = (
                f"Última corrida ({ultima['inicio']:%d/%m %H:%M} h, "
                f"{nombre_estado(ultima['state']).lower()}). Las transacciones y los "
                "préstamos se procesan en paralelo y se unen en Gold."
            )
        else:
            desc = "Los 10 pasos del DAG bronze_ingest."
        html(e.titulo_caja("Pipeline de datos", desc, "flujo"))
        if pasos is None:
            html(
                e.aviso(
                    "No se pudo conectar a la base de datos de Airflow. "
                    f"<span style='color:{e.GRIS_1}'>({error_airflow})</span>",
                    info=True,
                )
            )
        else:
            numero = 1
            bloques = []
            for etapa in ["Transacciones", "Préstamos", "Gold y modelo"]:
                filas = pasos[pasos["etapa"] == etapa]
                cajas = []
                for _, p in filas.iterrows():
                    intentos = p.get("try_number")
                    reintento = ""
                    if pd.notna(intentos) and intentos > 1:
                        n = int(intentos) - 1
                        reintento = f" · {n} reintento{'s' if n > 1 else ''}"
                    cajas.append(
                        '<div class="gd-paso">'
                        f'<div class="gd-paso-num">PASO {numero}</div>'
                        f'<div class="gd-paso-nombre">{p["nombre"]}</div>'
                        f'<div class="gd-paso-pie">{e.chip_estado(p["state"])}'
                        f'<span class="gd-dur">{e.formato_duracion(p["duration"])}'
                        f"{reintento}</span></div></div>"
                    )
                    numero += 1
                bloques.append(
                    f'<div class="gd-carril"><div class="gd-carril-nombre">{etapa}</div>'
                    f'<div class="gd-pasos">{"".join(cajas)}</div></div>'
                )
            html("".join(bloques))

with col_hist:
    with st.container(key="caja_historial"):
        html(
            e.titulo_caja(
                "Historial de corridas",
                "Duración de cada corrida del DAG (últimas 30), según los registros "
                "de Airflow. De aquí sale la tasa de éxito.",
                "reloj",
            )
        )
        if not hay_corridas:
            html(e.aviso("Sin datos de Airflow todavía.", info=True))
        else:
            h = corridas.dropna(subset=["inicio"]).copy()
            h["estado"] = h["state"].map(nombre_estado)
            h["fecha"] = h["inicio"].dt.strftime("%d/%m %H:%M")
            h["duracion"] = h["minutos"].map(
                lambda m: "en curso" if pd.isna(m) else f"{m:.1f} min"
            )
            h["min_barra"] = h["minutos"].fillna(0.15)
            ancho_barra = max(6, min(26, 360 // max(len(h), 1)))
            graf = (
                alt.Chart(h)
                .mark_bar(
                    cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=ancho_barra
                )
                .encode(
                    x=alt.X(
                        "fecha:N",
                        sort=None,
                        title=None,
                        axis=alt.Axis(labelAngle=-45, labelOverlap=True),
                    ),
                    y=alt.Y("min_barra:Q", title="Minutos"),
                    color=alt.Color(
                        "estado:N",
                        scale=alt.Scale(
                            domain=["Exitosa", "Falló", "En curso"],
                            range=[e.ELECTRIC, e.MANDARIN, e.SERENE],
                        ),
                        title=None,
                    ),
                    tooltip=[
                        alt.Tooltip("fecha:N", title="Inicio"),
                        alt.Tooltip("tipo:N", title="Tipo"),
                        alt.Tooltip("estado:N", title="Estado"),
                        alt.Tooltip("duracion:N", title="Duración"),
                    ],
                )
                .properties(height=250)
            )
            st.altair_chart(con_tema(graf), width="stretch", theme=None)

st.write("")

# ---------------------------------------------------------------------------
# Calidad de datos y duración por paso
# ---------------------------------------------------------------------------

col_cal, col_dur = st.columns([1, 1], gap="medium")

with col_cal:
    with st.container(key="caja_calidad"):
        reglas = list(val_tx) + list(val_loans or [])
        html(
            e.titulo_caja(
                "Calidad de datos",
                f"{sum(r.ok for r in reglas)} de {len(reglas)} reglas de negocio se "
                "cumplen en la capa Silver.",
                "calidad",
            )
        )
        bloque = '<div class="gd-subtitulo">Transacciones</div>'
        bloque += "".join(e.validacion(r.nombre, r.ok, r.detalle) for r in val_tx)
        if val_loans is not None:
            bloque += (
                '<div class="gd-subtitulo" style="margin-top:14px">Préstamos</div>'
            )
            bloque += "".join(
                e.validacion(r.nombre, r.ok, r.detalle) for r in val_loans
            )
        html(bloque)

with col_dur:
    with st.container(key="caja_duracion"):
        html(
            e.titulo_caja(
                "Duración por paso",
                "Tiempo de cada paso en la última corrida. Los más pesados son los "
                "de Spark: limpieza en Silver y construcción de Gold.",
                "barras",
            )
        )
        if pasos is None or pasos["duration"].isna().all():
            html(e.aviso("Sin datos de duración todavía.", info=True))
        else:
            d = pasos.dropna(subset=["duration"]).copy()
            d["paso"] = [f"{i + 1}. {n}" for i, n in zip(d["orden"], d["nombre"])]
            d["grupo"] = d["etapa"].map(
                lambda x: "Gold y modelo" if x == "Gold y modelo" else "Bronze y Silver"
            )
            d["texto"] = d["duration"].map(e.formato_duracion)
            base = alt.Chart(d).encode(
                y=alt.Y("paso:N", sort=list(d["paso"]), title=None),
                x=alt.X(
                    "duration:Q",
                    title="Segundos",
                    # Aire a la derecha para que la etiqueta del paso más largo
                    # no se corte.
                    scale=alt.Scale(domain=[0, float(d["duration"].max()) * 1.25]),
                ),
            )
            barras = base.mark_bar(
                cornerRadiusTopRight=4, cornerRadiusBottomRight=4
            ).encode(
                color=alt.Color(
                    "grupo:N",
                    scale=alt.Scale(
                        domain=["Bronze y Silver", "Gold y modelo"],
                        range=[e.ELECTRIC, e.SERENE],
                    ),
                    title=None,
                ),
                tooltip=[
                    alt.Tooltip("paso:N", title="Paso"),
                    alt.Tooltip("texto:N", title="Duración"),
                ],
            )
            etiquetas = base.mark_text(
                align="left", dx=6, font=e.FUENTE_GRAFICAS, fontSize=11, color=e.GRIS_1
            ).encode(text="texto:N")
            st.altair_chart(
                con_tema((barras + etiquetas).properties(height=34 * len(d) + 20)),
                width="stretch",
                theme=None,
            )

st.write("")

# ---------------------------------------------------------------------------
# Volumen de transacciones
# ---------------------------------------------------------------------------

col_dia, col_cat = st.columns([3, 2], gap="medium")

with col_dia:
    with st.container(key="caja_dia"):
        html(
            e.titulo_caja(
                "Transacciones por día",
                f"{e.formato_entero(por_dia['transacciones'].sum())} transacciones en "
                f"Silver, del {por_dia['fecha'].min():%d/%m/%Y} al "
                f"{por_dia['fecha'].max():%d/%m/%Y}.",
                "barras",
            )
        )
        area = (
            alt.Chart(por_dia)
            .mark_area(
                line={"color": e.ELECTRIC, "strokeWidth": 2},
                color=alt.Gradient(
                    gradient="linear",
                    stops=[
                        alt.GradientStop(color="#FFFFFF", offset=0),
                        alt.GradientStop(color=e.SERENE, offset=1),
                    ],
                    x1=1,
                    x2=1,
                    y1=1,
                    y2=0,
                ),
            )
            .encode(
                x=alt.X("fecha:T", title=None, axis=alt.Axis(labelExpr=MESES)),
                y=alt.Y("transacciones:Q", title="Transacciones"),
                tooltip=[
                    alt.Tooltip("fecha:T", title="Día", format="%d/%m/%Y"),
                    alt.Tooltip("transacciones:Q", title="Transacciones", format=","),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(con_tema(area), width="stretch", theme=None)

with col_cat:
    with st.container(key="caja_categoria"):
        html(
            e.titulo_caja(
                "Transacciones por categoría",
                "Categorías del dataset original de Kaggle, traducidas al español.",
                "capas",
            )
        )
        cat = (
            alt.Chart(por_categoria)
            .mark_bar(
                cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color=e.ELECTRIC
            )
            .encode(
                y=alt.Y("categoria:N", sort="-x", title=None),
                x=alt.X("transacciones:Q", title="Transacciones"),
                tooltip=[
                    alt.Tooltip("categoria:N", title="Categoría"),
                    alt.Tooltip("transacciones:Q", title="Transacciones", format=","),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(con_tema(cat), width="stretch", theme=None)

st.write("")

# ---------------------------------------------------------------------------
# Capa Gold y modelo de impago
# ---------------------------------------------------------------------------

with st.container(key="caja_gold"):
    html(
        e.titulo_caja(
            "Capa Gold y modelo de impago",
            "Revisión de que la capa Gold está completa y de que el modelo de riesgo "
            "se comporta como se espera.",
            "modelo",
        )
    )
    g = gold or {}
    modelo = g.get("modelo") or {}
    completos = g.get("clientes") is not None and (
        g.get("prestamos_unicos") == g.get("clientes") == g.get("prestamos_asignados")
    )
    auc = modelo.get("auc_prueba")
    real = modelo.get("tasa_impago_real_prueba")
    pred = modelo.get("tasa_impago_predicha_prueba")
    hay_calibracion = real is not None and pred is not None

    mini = [
        e.kpi(
            "Préstamos asignados",
            e.formato_entero(g.get("prestamos_unicos")),
            (
                "Uno distinto por cliente (llave sintética)"
                if completos
                else "Revisar: no cuadra con el número de clientes"
            ),
            "prestamo",
            e.LIME if completos else e.MANDARIN,
        ),
        e.kpi(
            "AUC del modelo",
            f"{auc:.2f}" if auc else "—",
            (
                "En préstamos que el modelo no vio (0.5 = adivinar)"
                if auc
                else "Todavía no hay métricas"
            ),
            "modelo",
            (e.LIME if auc >= 0.7 else e.MANDARIN) if auc else e.GRIS_3,
        ),
        e.kpi(
            "Calibración",
            f"{pred:.1%}" if hay_calibracion else "—",
            (
                f"Impago promedio que predice el modelo; el real es {real:.1%}"
                if hay_calibracion
                else "Todavía no hay métricas"
            ),
            "calidad",
            e.LIME if hay_calibracion and abs(pred - real) < 0.01 else e.GRIS_3,
        ),
        e.kpi(
            "Clientes con riesgo alto",
            e.formato_entero(g.get("clientes_riesgo_alto")),
            "Probabilidad de impago de 30% o más",
            "alerta",
            e.MANDARIN,
        ),
        e.kpi(
            "Transacciones con fraude",
            f"{pct_fraude:.2f}%",
            "Marcadas como fraude en el dataset",
            "alerta",
            e.PURPLE,
        ),
    ]
    for col, tarjeta in zip(st.columns(5, gap="medium"), mini):
        with col:
            html(tarjeta)

html(
    '<p class="gd-pie">Fuentes: Credit Card Transactions Fraud Detection y Loan Default '
    "Prediction (Kaggle). El ingreso simulado y el préstamo asignado a cada cliente son "
    "construcciones documentadas (Opción A), no datos reales de ninguna persona. Montos "
    "en USD. Los datos se refrescan cada minuto al recargar la página.</p>"
)
