"""
Dashboard Ejecutivo del Gemelo Digital Financiero — v1 (Fase 4).

Para tomadores de decisiones de negocio (sección 12 del documento del
proyecto). Lee solo de la capa Gold (y de Silver para el histórico):

  Portafolio   Salud financiera de todos los clientes o de un segmento de
               riesgo: KPIs, niveles de riesgo, ingresos vs gastos por mes,
               mapa de calor de compras, consumo esencial vs discrecional y
               alertas tempranas de impago.
  Perfil 360   Un cliente: sus KPIs, su histórico, en qué gasta, su préstamo
               y por qué el modelo le da esa probabilidad de impago.

El simulador Monte Carlo (src/ml/simulador.py) aparece en las dos vistas:
impacto de cada escenario en el portafolio y simulación interactiva de un
cliente. Pendiente para la siguiente fase: chat con el asistente de IA.

Corre con: streamlit run src/dashboard/ejecutivo.py
"""

import math
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dashboard import datos  # noqa: E402
from src.dashboard import datos_ejecutivo as de  # noqa: E402
from src.dashboard import estilo as e  # noqa: E402
from src.ml import simulador as sm  # noqa: E402
from src.ml.explicacion import CORTES_RIESGO, NIVELES, RANGOS  # noqa: E402

RUTAS = datos.Rutas(Path("data"))

st.set_page_config(
    page_title="Dashboard Ejecutivo — Gemelo Digital Financiero",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.html(e.CSS)

COLOR_NIVEL = {"Bajo": e.LIME, "Medio": e.CANARY, "Alto": e.MANDARIN}
SIMBOLO_NIVEL = {"Bajo": "●", "Medio": "▲", "Alto": "▲"}


# ---------------------------------------------------------------------------
# Datos (en caché: se leen una vez y se reutilizan entre interacciones)
# ---------------------------------------------------------------------------


@st.cache_resource(ttl=600, show_spinner="Cargando la capa Gold…")
def cargar_base():
    clientes = de.cargar_clientes(RUTAS)
    metricas = de.cargar_metricas(RUTAS)
    aportes = de.explicar_clientes(clientes, metricas)
    tx = de.cargar_transacciones(RUTAS)
    impago = de.impago_real_por_nivel(pd.read_parquet(RUTAS.silver_loans), metricas)
    return clientes, metricas, aportes, tx, impago


@st.cache_data(ttl=600, show_spinner=False)
def agregados(segmento: str):
    clientes, _, _, tx, _ = cargar_base()
    sub = clientes if segmento == "Todos" else clientes[clientes["nivel"] == segmento]
    tx_sub = tx[tx["cc_num"].isin(sub["cc_num"])]
    return (
        de.ingresos_vs_gastos(tx, sub),
        de.mapa_calor(tx_sub),
        de.consumo_por_categoria(tx_sub),
    )


@st.cache_data(ttl=600, show_spinner=False)
def agregados_cliente(cc_num: int):
    clientes, _, _, tx, _ = cargar_base()
    uno = clientes[clientes["cc_num"] == cc_num]
    tx_uno = tx[tx["cc_num"] == cc_num]
    return de.ingresos_vs_gastos(tx, uno), de.consumo_por_categoria(tx_uno), tx_uno


@st.cache_resource(ttl=600, show_spinner=False)
def cargar_sim():
    return de.cargar_simulacion(RUTAS)


@st.cache_data(ttl=600, show_spinner="Simulando 1,000 futuros posibles…")
def simular(
    cc_num, ingreso, gasto_base, cv, escenario, supuestos: tuple, parametros: tuple
):
    """Corre el motor para un cliente (escenario elegido y base para comparar)."""
    sup = sm.Supuestos(**dict(supuestos))
    p = dict(parametros)
    par = sm.ParametrosEscenarios(
        credito_hipotecario=sm.CreditoHipotecario(**dict(p["credito_hipotecario"])),
        cambio_empleo=sm.CambioEmpleo(**dict(p["cambio_empleo"])),
        negocio=sm.Negocio(**dict(p["negocio"])),
    )
    sim = sm.simular_cliente(
        ingreso,
        gasto_base,
        cv,
        escenario,
        sup,
        par,
        sm.semilla_cliente(cc_num, escenario, sup.semilla),
    )
    base = sm.simular_cliente(
        ingreso,
        gasto_base,
        cv,
        "base",
        sup,
        par,
        sm.semilla_cliente(cc_num, "base", sup.semilla),
    )
    return (
        sm.resumir(sim, gasto_base, sup),
        sm.bandas(sim),
        sim["info"],
        sm.resumir(base, gasto_base, sup),
        sm.bandas(base),
    )


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


def dinero(x) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"−${-x:,.0f}" if x < 0 else f"${x:,.0f}"


def no_deseleccionar(key: str, defecto: str):
    """Los botones segmentados se desmarcan si se vuelve a picar la opción
    activa; esto la deja marcada."""
    if st.session_state.get(key) is None:
        st.session_state[key] = st.session_state.get(f"{key}_previo", defecto)
    st.session_state[f"{key}_previo"] = st.session_state[key]


def pct(x, dec=0) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"−{-x * 100:.{dec}f}%" if x < 0 else f"{x * 100:.{dec}f}%"


def chip_nivel(nivel: str, extra: str = "") -> str:
    texto = f"Riesgo {nivel.lower()}" + (f" · {extra}" if extra else "")
    return e.pill(texto, COLOR_NIVEL[nivel], SIMBOLO_NIVEL[nivel])


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------


def grafica_ingresos_gastos(ivg: pd.DataFrame):
    largo = ivg.melt(
        id_vars=["mes", "orden", "ahorro"],
        value_vars=["ingreso", "gasto"],
        var_name="serie",
        value_name="monto",
    )
    largo["serie"] = largo["serie"].map(
        {"ingreso": "Ingreso (simulado)", "gasto": "Gasto (real)"}
    )
    orden = list(ivg.sort_values("orden")["mes"])
    chart = (
        alt.Chart(largo)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=18)
        .encode(
            x=alt.X("mes:N", sort=orden, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("serie:N", sort=["Ingreso (simulado)", "Gasto (real)"]),
            y=alt.Y(
                "monto:Q", title="USD al mes por cliente", axis=alt.Axis(format=",.0f")
            ),
            color=alt.Color(
                "serie:N",
                scale=alt.Scale(
                    domain=["Ingreso (simulado)", "Gasto (real)"],
                    range=[e.SERENE, e.ELECTRIC],
                ),
                title=None,
            ),
            tooltip=[
                alt.Tooltip("mes:N", title="Mes"),
                alt.Tooltip("serie:N", title="Concepto"),
                alt.Tooltip("monto:Q", title="USD", format=",.0f"),
                alt.Tooltip("ahorro:Q", title="Ahorro del mes", format=".0%"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(con_tema(chart), width="stretch", theme=None)


def grafica_calor(calor: pd.DataFrame):
    chart = (
        alt.Chart(calor)
        .mark_rect(cornerRadius=2)
        .encode(
            x=alt.X("hora:O", title="Hora del día", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("dia_nombre:O", sort=de.DIAS, title=None),
            color=alt.Color(
                "pct:Q",
                scale=alt.Scale(range=["#EEF6FF", e.SERENE, e.ELECTRIC]),
                legend=alt.Legend(title="% de compras", format=".1%", orient="right"),
            ),
            tooltip=[
                alt.Tooltip("dia_nombre:N", title="Día"),
                alt.Tooltip("hora:O", title="Hora"),
                alt.Tooltip("transacciones:Q", title="Compras", format=","),
                alt.Tooltip("pct:Q", title="% del total", format=".2%"),
            ],
        )
        .properties(height=250)
    )
    st.altair_chart(con_tema(chart), width="stretch", theme=None)


def grafica_consumo(consumo: pd.DataFrame, alto: int = None):
    base = alt.Chart(consumo).encode(
        y=alt.Y(
            "categoria:N",
            sort=list(consumo.sort_values("pct", ascending=False)["categoria"]),
            title=None,
        ),
        x=alt.X(
            "pct:Q",
            title="% del gasto",
            axis=alt.Axis(format=".0%"),
            scale=alt.Scale(domain=[0, float(consumo["pct"].max()) * 1.25]),
        ),
    )
    barras = base.mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        color=alt.Color(
            "tipo:N",
            scale=alt.Scale(
                domain=["Esencial", "Discrecional"], range=[e.ELECTRIC, e.SERENE]
            ),
            title=None,
        ),
        tooltip=[
            alt.Tooltip("categoria:N", title="Categoría"),
            alt.Tooltip("tipo:N", title="Tipo"),
            alt.Tooltip("gasto:Q", title="Gasto (USD)", format=",.0f"),
            alt.Tooltip("pct:Q", title="% del gasto", format=".1%"),
        ],
    )
    texto = base.mark_text(
        align="left", dx=5, font=e.FUENTE_GRAFICAS, fontSize=11, color=e.GRIS_1
    ).encode(text=alt.Text("pct:Q", format=".0%"))
    alto = alto or 22 * len(consumo) + 30
    st.altair_chart(
        con_tema((barras + texto).properties(height=alto)), width="stretch", theme=None
    )


def grafica_factores(f: pd.DataFrame):
    limite = float(f["aporte"].abs().max()) * 1.35
    base = alt.Chart(f).encode(
        y=alt.Y(
            "factor:N",
            sort=list(f["factor"]),
            title=None,
            axis=alt.Axis(labelLimit=320, labelFontSize=12),
        ),
        x=alt.X(
            "aporte:Q",
            title="← baja el riesgo · sube el riesgo →",
            scale=alt.Scale(domain=[-limite, limite]),
        ),
    )
    barras = base.mark_bar(cornerRadius=4, size=18).encode(
        color=alt.Color(
            "efecto:N",
            scale=alt.Scale(
                domain=["Sube el riesgo", "Baja el riesgo"],
                range=[e.MANDARIN, e.ELECTRIC],
            ),
            title=None,
        ),
        tooltip=[
            alt.Tooltip("factor:N", title="Factor"),
            alt.Tooltip("efecto:N", title="Efecto"),
            alt.Tooltip("aporte:Q", title="Aporte", format="+.2f"),
        ],
    )
    f = f.assign(pos=f["aporte"] > 0)
    textos = [
        alt.Chart(f[f["pos"] == lado])
        .mark_text(
            align="left" if lado else "right",
            dx=5 if lado else -5,
            font=e.FUENTE_GRAFICAS,
            fontSize=11,
            color=e.GRIS_1,
        )
        .encode(
            y=alt.Y("factor:N", sort=list(f["factor"])),
            x="aporte:Q",
            text=alt.Text("aporte:Q", format="+.2f"),
        )
        for lado in (True, False)
    ]
    regla = (
        alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=e.GRIS_1).encode(x="x:Q")
    )
    capas = alt.layer(barras, regla, *textos).properties(height=34 * len(f) + 30)
    st.altair_chart(
        con_tema(capas),
        width="stretch",
        theme=None,
    )


def grafica_abanico(b: pd.DataFrame, base: pd.DataFrame = None):
    """Bandas de confianza del ahorro acumulado (abanico) y la mediana."""
    d = b.copy()
    capas = [
        alt.Chart(d)
        .mark_area(color=e.SERENE, opacity=0.45)
        .encode(
            x=alt.X(
                "mes:Q",
                title="Meses desde hoy",
                scale=alt.Scale(domain=[1, int(d["mes"].max())], nice=False),
                axis=alt.Axis(values=list(range(0, int(d["mes"].max()) + 1, 6))),
            ),
            y=alt.Y("p5:Q", title="Ahorro acumulado (USD)", axis=alt.Axis(format="~s")),
            y2="p95:Q",
            tooltip=[
                alt.Tooltip("mes:Q", title="Mes"),
                alt.Tooltip("p5:Q", title="Escenario malo (P5)", format=",.0f"),
                alt.Tooltip("p50:Q", title="Mediana", format=",.0f"),
                alt.Tooltip("p95:Q", title="Escenario bueno (P95)", format=",.0f"),
            ],
        ),
        alt.Chart(d)
        .mark_area(color=e.SERENE, opacity=0.75)
        .encode(x="mes:Q", y="p25:Q", y2="p75:Q"),
        alt.Chart(d)
        .mark_line(color=e.ELECTRIC, strokeWidth=2.5)
        .encode(x="mes:Q", y="p50:Q"),
    ]
    if base is not None:
        capas.append(
            alt.Chart(base)
            .mark_line(color=e.GRIS_1, strokeDash=[5, 4], strokeWidth=1.6)
            .encode(x="mes:Q", y="p50:Q")
        )
    if d["p5"].min() < 0:
        capas.append(
            alt.Chart(pd.DataFrame({"y": [0]}))
            .mark_rule(color=e.MANDARIN, strokeWidth=1.5)
            .encode(y="y:Q")
        )
    st.altair_chart(
        con_tema(alt.layer(*capas).properties(height=320)), width="stretch", theme=None
    )


def grafica_escenarios(t: pd.DataFrame):
    d = t.reset_index()
    base = alt.Chart(d).encode(
        y=alt.Y("nombre:N", sort=list(d["nombre"]), title=None),
        x=alt.X(
            "en_riesgo:Q",
            title="% de clientes en riesgo de números rojos",
            axis=alt.Axis(format=".0%"),
            scale=alt.Scale(
                domain=[0, math.ceil(float(d["en_riesgo"].max()) * 13) / 10 or 0.1]
            ),
        ),
    )
    barras = base.mark_bar(
        cornerRadiusTopRight=4, cornerRadiusBottomRight=4, size=22
    ).encode(
        color=alt.condition(
            alt.datum.escenario == "base", alt.value(e.SERENE), alt.value(e.ELECTRIC)
        ),
        tooltip=[
            alt.Tooltip("nombre:N", title="Escenario"),
            alt.Tooltip("en_riesgo:Q", title="En riesgo", format=".1%"),
            alt.Tooltip(
                "ahorro_12m:Q", title="Ahorro mediano a 12 meses", format=",.0f"
            ),
        ],
    )
    texto = base.mark_text(
        align="left", dx=6, font=e.FUENTE_GRAFICAS, fontSize=12, color=e.GRIS_1
    ).encode(text=alt.Text("en_riesgo:Q", format=".1%"))
    st.altair_chart(
        con_tema((barras + texto).properties(height=190)), width="stretch", theme=None
    )


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

try:
    clientes, metricas, aportes, tx, impago_real = cargar_base()
    error_datos = None
except Exception as exc:  # noqa: BLE001
    error_datos = str(exc).splitlines()[0][:200]

if error_datos:
    html(
        e.aviso(
            "<b>No se pudo leer la capa Gold.</b> ¿Ya corriste el DAG "
            f"<code>bronze_ingest</code> completo? Detalle: {error_datos}"
        )
    )
    st.stop()

completos = de.meses_completos(tx)
periodo = (
    f"{de.nombre_mes(completos[0])} – {de.nombre_mes(completos[-1])}"
    if completos
    else "—"
)
ahora = pd.Timestamp.now(tz=datos.ZONA_HORARIA)

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------

izq, der = st.columns([3, 2], vertical_alignment="bottom")
with izq:
    html(
        '<div class="gd-eyebrow">Dashboard ejecutivo · Negocio</div>'
        '<div class="gd-h1">Gemelo Digital Financiero</div>'
        '<p class="gd-sub">Salud financiera del portafolio de clientes y de cada '
        "cliente, a partir de la capa Gold.</p>"
    )
with der:
    html(
        '<div class="gd-header-der">'
        '<span class="gd-tag">Uso académico · datos simulados</span>'
        f'<div class="gd-meta">{len(clientes):,} clientes · histórico {periodo}<br>'
        f"Actualizado: {ahora:%H:%M} h</div></div>"
    )

st.write("")
tab_portafolio, tab_cliente = st.tabs(["Portafolio", "Perfil 360 del cliente"])

# ===========================================================================
# PORTAFOLIO
# ===========================================================================

with tab_portafolio:
    f1, f2 = st.columns([1, 6], vertical_alignment="center")
    with f1:
        html('<div class="gd-filtro">Ver segmento de riesgo:</div>')
    with f2:
        segmento = st.segmented_control(
            "Segmento de riesgo",
            ["Todos", *NIVELES],
            default="Todos",
            key="segmento",
            label_visibility="collapsed",
            on_change=no_deseleccionar,
            args=("segmento", "Todos"),
        )
    segmento = segmento or "Todos"
    sub = clientes if segmento == "Todos" else clientes[clientes["nivel"] == segmento]
    r = de.resumen_portafolio(sub)
    ivg, calor, consumo = agregados(segmento)
    etiqueta_seg = (
        "todo el portafolio" if segmento == "Todos" else f"riesgo {segmento.lower()}"
    )

    # ---- KPIs ----
    nivel_mediano = (
        de.nivel_riesgo(pd.Series([r["prob_impago"]])).iloc[0]
        if r["clientes"]
        else "Bajo"
    )
    tarjetas = [
        e.kpi(
            "Clientes",
            e.formato_entero(r["clientes"]),
            (
                "Con KPIs, préstamo asignado y riesgo"
                if segmento == "Todos"
                else f"{r['clientes'] / len(clientes):.0%} del portafolio"
            ),
            "personas",
        ),
        e.kpi(
            "Gasto mensual típico",
            dinero(r["gasto_mensual"]),
            "Mediana por cliente, USD. Gasto real con tarjeta",
            "carrito",
        ),
        e.kpi(
            "Capacidad de ahorro",
            pct(r["ahorro"]),
            f"Mediana. {r['ahorro_negativo']} clientes gastan más de lo que ganan",
            "ahorro",
            e.LIME if (r["ahorro"] or 0) >= 0.2 else e.MANDARIN,
        ),
        e.kpi(
            "Endeudamiento (DTI)",
            "—" if r["dti"] is None else f"{r['dti']:.2f}",
            f"Mediana. {r['dti_mayor_1']} clientes pagan más que su ingreso",
            "prestamo",
            e.LIME if (r["dti"] or 0) <= 0.43 else e.MANDARIN,
        ),
        e.kpi(
            "Probabilidad de impago",
            pct(r["prob_impago"], 1),
            f"Mediana. {r['riesgo_alto']} clientes en riesgo alto",
            "alerta",
            COLOR_NIVEL[nivel_mediano],
        ),
    ]
    for i, (col, t) in enumerate(zip(st.columns(5), tarjetas)):
        with col:
            with st.container(key=f"caja_pk{i}"):
                html(t)

    st.write("")

    # ---- Niveles de riesgo + ingresos vs gastos ----
    c1, c2 = st.columns([5, 7], gap="medium")
    tabla = de.tabla_niveles(clientes, impago_real)
    with c1:
        with st.container(key="caja_niveles"):
            alto = tabla.loc["Alto"]
            desc = f"El {alto['pct']:.0%} de los clientes está en riesgo alto." + (
                f" En préstamos con ese perfil, {alto['impago_real'] * 100:.0f} de cada "
                "100 no se pagaron."
                if impago_real
                else ""
            )
            html(e.titulo_caja("Niveles de riesgo", desc, "alerta"))
            filas = []
            for nivel, rango in zip(NIVELES, RANGOS):
                t = tabla.loc[nivel]
                activo = " activo" if segmento == nivel else ""
                real = (
                    f" · Impago real en préstamos parecidos: <b>{t['impago_real']:.1%}</b>"
                    if impago_real
                    else ""
                )
                filas.append(
                    f'<div class="gd-nivel{activo}"><div class="gd-nivel-top">'
                    f"{chip_nivel(nivel, rango)}"
                    f'<span class="gd-nivel-cifra">{int(t["clientes"]):,} '
                    f'<small>clientes · {t["pct"]:.0%}</small></span></div>'
                    f'<div class="gd-barra"><span style="width:{t["pct"] * 100:.1f}%;'
                    f'background:{COLOR_NIVEL[nivel]}"></span></div>'
                    '<div class="gd-nivel-det">Probabilidad promedio '
                    f'<b>{t["prob_promedio"]:.1%}</b>'
                    f' · DTI mediano <b>{t["dti_mediano"]:.2f}</b>'
                    f' · Ahorro mediano <b>{t["ahorro_mediano"]:.0%}</b>{real}</div></div>'
                )
            html("".join(filas))
            html(
                '<p class="gd-nota">Cortes en '
                f"{CORTES_RIESGO[0]:.0%} y {CORTES_RIESGO[1]:.0%}, validados con BBVA. "
                "El impago real se mide en los 255 mil préstamos del dataset Loan "
                "Default.</p>"
            )
    with c2:
        with st.container(key="caja_ivg"):
            if len(ivg):
                prom = ivg["gasto"].mean() / ivg["ingreso"].iloc[0]
                mayor = ivg.loc[ivg["gasto"].idxmax()]
                desc = (
                    f"En {etiqueta_seg}, cada cliente gasta en promedio {prom:.0%} de su "
                    f"ingreso. El mes de más gasto es {mayor['mes']}: "
                    f"{mayor['gasto'] / mayor['ingreso']:.0%} del ingreso."
                )
            else:
                desc = "Sin datos."
            html(e.titulo_caja("Ingresos vs gastos por mes", desc, "barras"))
            if len(ivg):
                grafica_ingresos_gastos(ivg)
            html(
                '<p class="gd-nota">Promedio por cliente. El gasto es real (transacciones '
                "con tarjeta); el ingreso es simulado (Opción A). No se muestra junio porque "
                "el dataset empieza el 21 de ese mes.</p>"
            )

    st.write("")

    # ---- Mapa de calor + consumo ----
    c1, c2 = st.columns([7, 5], gap="medium")
    with c1:
        with st.container(key="caja_calor"):
            if calor["transacciones"].sum():
                dia, hora = de.hora_pico(calor)
                tarde = calor.loc[calor["hora"] >= 12, "pct"].sum()
                desc = (
                    f"El {tarde:.0%} de las compras se hace de las 12 h en adelante. "
                    f"El momento de más compras es el {dia.lower()} a las {hora} h."
                )
            else:
                desc = "Sin datos."
            html(e.titulo_caja("¿Cuándo compran?", desc, "calendario"))
            grafica_calor(calor)
    with c2:
        with st.container(key="caja_consumo"):
            esencial = de.pct_esencial(consumo)
            desc = (
                f"El {esencial:.0%} del gasto es esencial (súper, gasolina, hogar, salud, "
                f"hijos y mascotas) y el {1 - esencial:.0%} es discrecional."
            )
            html(e.titulo_caja("¿En qué gastan?", desc, "carrito"))
            grafica_consumo(consumo)

    st.write("")

    # ---- Alertas tempranas ----
    with st.container(key="caja_alertas"):
        al = de.alertas(sub, aportes)
        if len(al):
            desc = (
                f"{len(al)} clientes con probabilidad de impago de "
                f"{CORTES_RIESGO[1]:.0%} o más. Se muestran los 10 más altos; consulta "
                "cualquiera en la pestaña «Perfil 360 del cliente»."
            )
        else:
            desc = "Este segmento no tiene clientes en riesgo alto."
        html(e.titulo_caja("Alertas tempranas de impago", desc, "alerta"))
        if len(al):
            filas = "".join(
                f'<tr><td class="gd-cliente">{a.cliente}</td>'
                f'<td>{chip_nivel("Alto", f"{a.prob_impago:.1%}")}</td>'
                f"<td>{a.factor}</td>"
                f'<td class="num">{a.dti:.2f}</td>'
                f'<td class="num">{a.capacidad_ahorro:.0%}</td>'
                f'<td class="num">{dinero(a.gasto_promedio_mensual)}</td></tr>'
                for a in al.head(10).itertuples()
            )
            html(
                '<table class="gd-tabla"><thead><tr><th>Cliente</th>'
                "<th>Probabilidad de impago</th><th>Lo que más sube su riesgo</th>"
                '<th class="num">DTI</th><th class="num">Capacidad de ahorro</th>'
                f'<th class="num">Gasto mensual</th></tr></thead><tbody>{filas}</tbody>'
                "</table>"
            )
            if aportes is None:
                html(
                    '<p class="gd-nota">Para ver «lo que más sube su riesgo», vuelve a '
                    "correr el DAG (el modelo guarda los datos para explicarlo).</p>"
                )

    st.write("")

    # ---- Simulador: escenarios en el portafolio ----
    simulacion = cargar_sim()
    with st.container(key="caja_esc"):
        if simulacion is None:
            html(
                e.titulo_caja(
                    "Escenarios a futuro (simulador Monte Carlo)", "", "modelo"
                )
            )
            html(
                e.aviso(
                    "Para ver los escenarios, vuelve a correr el DAG: el paso "
                    "<code>simulador_monte_carlo</code> los calcula.",
                    info=True,
                )
            )
        else:
            _, resumen_sim, _ = simulacion
            esc = de.escenarios_portafolio(resumen_sim, sub)
            b0 = esc.loc["base", "en_riesgo"]
            b1 = esc.loc["credito_hipotecario", "en_riesgo"]
            desc = (
                f"1,000 futuros posibles por cliente a 36 meses. Sin cambios, el {b0:.0%} de "
                f"{etiqueta_seg} quedaría en riesgo de números rojos; con un crédito "
                f"hipotecario de 3 años de ingreso, el {b1:.0%}."
            )
            html(
                e.titulo_caja(
                    "Escenarios a futuro (simulador Monte Carlo)", desc, "modelo"
                )
            )
            g1, g2 = st.columns([6, 6], gap="medium")
            with g1:
                grafica_escenarios(esc)
            with g2:
                filas = "".join(
                    f'<tr><td class="gd-cliente">{r.nombre}</td>'
                    f'<td class="num">{r.en_riesgo:.1%}</td>'
                    f'<td class="num">{dinero(r.ahorro_12m)}</td>'
                    f'<td class="num">{pct(r.capacidad_12m)}</td></tr>'
                    for r in esc.itertuples()
                )
                html(
                    '<table class="gd-tabla"><thead><tr><th>Escenario</th>'
                    '<th class="num">En riesgo</th><th class="num">Ahorro a 12 meses</th>'
                    '<th class="num">Capacidad de ahorro, año 1</th></tr></thead>'
                    f"<tbody>{filas}</tbody></table>"
                )
            html(
                '<p class="gd-nota">«En riesgo» = en al menos 10% de sus futuros posibles, su '
                "ahorro acumulado baja de cero. Medianas por cliente. Prueba cada escenario "
                "con un cliente en la pestaña «Perfil 360». Siguiente fase: el asistente de "
                "IA, para preguntar por un cliente en lenguaje natural.</p>"
            )

# ===========================================================================
# PERFIL 360
# ===========================================================================

with tab_cliente:
    por_riesgo = clientes.sort_values("prob_impago", ascending=False)
    opciones = list(clientes["cliente"])
    etiquetas = {
        fila.cliente: f"{fila.cliente} · riesgo {fila.nivel.lower()} ({fila.prob_impago:.1%})"
        for fila in clientes.itertuples()
    }
    s1, _ = st.columns([2, 3])
    with s1:
        elegido = st.selectbox(
            "Elige un cliente (escribe para buscar)",
            opciones,
            index=opciones.index(por_riesgo["cliente"].iloc[0]),
            format_func=lambda c: etiquetas[c],
            key="cliente",
        )
    idx = clientes.index[clientes["cliente"] == elegido][0]
    c = clientes.loc[idx]
    port = de.resumen_portafolio(clientes)
    ivg_c, consumo_c, tx_c = agregados_cliente(int(c["cc_num"]))

    # ---- Encabezado del cliente ----
    with st.container(key="caja_cli"):
        a, b = st.columns([3, 2], vertical_alignment="center")
        with a:
            principal = datos.CATEGORIAS.get(
                c["categoria_principal"], c["categoria_principal"]
            )
            html(
                '<div class="gd-eyebrow">Perfil 360</div>'
                f'<div class="gd-cliente-nombre">{c["cliente"]}</div>'
                f'<p class="gd-sub">Cliente desde {c["primera_transaccion"]:%d/%m/%Y} · '
                f'{int(c["num_transacciones"]):,} compras con tarjeta · gasta más en '
                f'{principal.lower()} · vive en una ciudad de {int(c["city_pop"]):,} '
                "habitantes</p>"
            )
        with b:
            texto_prob = f"{c['prob_impago']:.1%} de probabilidad de impago"
            html(
                f'<div class="gd-header-der">{chip_nivel(c["nivel"], texto_prob)}</div>'
            )

    st.write("")

    # ---- KPIs del cliente ----
    tarjetas = [
        e.kpi(
            "Ingreso mensual",
            dinero(c["ingreso_mensual_simulado"]),
            f"Simulado (Opción A). Mediana del portafolio: {dinero(port['ingreso_mensual'])}",
            "personas",
        ),
        e.kpi(
            "Gasto mensual",
            dinero(c["gasto_promedio_mensual"]),
            f"Real, con tarjeta. Mediana del portafolio: {dinero(port['gasto_mensual'])}",
            "carrito",
        ),
        e.kpi(
            "Capacidad de ahorro",
            pct(c["capacidad_ahorro"]),
            f"Mediana del portafolio: {pct(port['ahorro'])}",
            "ahorro",
            e.LIME if c["capacidad_ahorro"] >= 0.2 else e.MANDARIN,
        ),
        e.kpi(
            "Endeudamiento (DTI)",
            f"{c['dti']:.2f}",
            (
                "Su pago mensual supera su ingreso"
                if c["dti"] > 1
                else f"Mediana del portafolio: {port['dti']:.2f}"
            ),
            "prestamo",
            e.LIME if c["dti"] <= 0.43 else e.MANDARIN,
        ),
        e.kpi(
            "Probabilidad de impago",
            pct(c["prob_impago"], 1),
            f"Mediana del portafolio: {pct(port['prob_impago'], 1)}",
            "alerta",
            COLOR_NIVEL[c["nivel"]],
        ),
    ]
    for i, (col, t) in enumerate(zip(st.columns(5), tarjetas)):
        with col:
            with st.container(key=f"caja_ck{i}"):
                html(t)

    st.write("")

    # ---- Histórico del cliente ----
    c1, c2 = st.columns([7, 5], gap="medium")
    with c1:
        with st.container(key="caja_cli_ivg"):
            if len(ivg_c):
                mayor = ivg_c.loc[ivg_c["gasto"].idxmax()]
                negativos = int((ivg_c["ahorro"] < 0).sum())
                desc = (
                    f"Gastó en promedio {ivg_c['gasto'].mean() / ivg_c['ingreso'].iloc[0]:.0%} "
                    f"de su ingreso. Su mes de más gasto fue {mayor['mes']}"
                    + (
                        f"; en {negativos} mes(es) gastó más de lo que ganó."
                        if negativos
                        else "."
                    )
                )
            else:
                desc = "Sin datos."
            html(e.titulo_caja("Ingresos vs gastos por mes", desc, "barras"))
            if len(ivg_c):
                grafica_ingresos_gastos(ivg_c)
    with c2:
        with st.container(key="caja_cli_consumo"):
            esencial = de.pct_esencial(consumo_c)
            desc = (
                f"El {esencial:.0%} de su gasto es esencial y el {1 - esencial:.0%} "
                "discrecional."
            )
            html(e.titulo_caja("¿En qué gasta?", desc, "carrito"))
            grafica_consumo(consumo_c)

    st.write("")

    # ---- Préstamo + por qué ----
    c1, c2 = st.columns([5, 7], gap="medium")
    with c1:
        with st.container(key="caja_cli_prestamo"):
            html(
                e.titulo_caja(
                    "Su préstamo",
                    "El préstamo que la llave sintética le asignó (dataset Loan Default).",
                    "prestamo",
                )
            )
            datos_prestamo = [
                ("Monto", f"{c['monto_prestamo_fuente']:,.0f} USD"),
                ("Tasa de interés anual", f"{c['tasa_interes_anual_fuente']:.2f}%"),
                ("Plazo", f"{int(c['plazo_meses_fuente'])} meses"),
                ("Pago mensual", f"{c['pago_mensual_prestamo']:,.0f} USD"),
                (
                    "Ingreso mensual (fuente)",
                    f"{c['ingreso_mensual_fuente']:,.0f} USD",
                ),
                ("Endeudamiento (DTI)", f"{c['dti']:.2f}"),
                ("Score de crédito (fuente)", f"{int(c['score_credito_fuente'])}"),
            ]
            html(
                "".join(
                    f'<div class="gd-dato"><span>{k}</span><span>{v}</span></div>'
                    for k, v in datos_prestamo
                )
            )
            html(
                '<p class="gd-nota">El DTI y el riesgo usan el ingreso de la fuente; el '
                "ahorro usa el ingreso simulado. Son dos lecturas distintas del cliente "
                "(decisión 2, docs/fase4_decisiones_kpis.md).</p>"
            )
    with c2:
        with st.container(key="caja_cli_porque"):
            if aportes is not None:
                f = de.factores(aportes.loc[idx], c)
                sube = f[f["aporte"] > 0]
                baja = f[f["aporte"] < 0]
                partes = []
                if len(sube):
                    partes.append(
                        f"Lo que más sube su riesgo: {sube.iloc[0]['factor'].lower()}."
                    )
                if len(baja):
                    partes.append(
                        f"Lo que más lo baja: {baja.iloc[0]['factor'].lower()}."
                    )
                html(
                    e.titulo_caja(
                        "¿Por qué tiene este riesgo?", " ".join(partes), "modelo"
                    )
                )
                grafica_factores(f)
                html(
                    '<p class="gd-nota">Aporte de cada factor frente a un préstamo promedio '
                    "del dataset, según el modelo de regresión logística "
                    "(docs/fase5_revision_distribuciones.md).</p>"
                )
            else:
                html(e.titulo_caja("¿Por qué tiene este riesgo?", "", "modelo"))
                html(
                    e.aviso(
                        "Para ver la explicación, vuelve a correr el DAG: el paso "
                        "<code>modelo_prob_impago</code> guarda los datos que la hacen "
                        "posible.",
                        info=True,
                    )
                )

    st.write("")

    # ---- Simulador de escenarios ----
    with st.container(key="caja_cli_sim"):
        html(
            e.titulo_caja(
                "Simulador de escenarios (Monte Carlo)",
                "1,000 futuros posibles de su dinero en los próximos 36 meses, con su gasto "
                "real y su variación, la inflación y el riesgo de perder su ingreso.",
                "modelo",
            )
        )
        simulacion = cargar_sim()
        fila_sim = (
            None
            if simulacion is None
            else simulacion[0][simulacion[0]["cc_num"] == c["cc_num"]]
        )
        if fila_sim is None or fila_sim.empty:
            html(
                e.aviso(
                    "Para usar el simulador, vuelve a correr el DAG: el paso "
                    "<code>simulador_monte_carlo</code> prepara los datos de cada cliente.",
                    info=True,
                )
            )
        else:
            params_sim = simulacion[2]
            campos = sm.Supuestos.__dataclass_fields__
            sup_dict = {
                k: v for k, v in params_sim.get("supuestos", {}).items() if k in campos
            }
            nombres = {v: k for k, v in sm.NOMBRES_ESCENARIOS.items()}
            elegido_esc = (
                st.segmented_control(
                    "Escenario",
                    list(sm.NOMBRES_ESCENARIOS.values()),
                    default="Crédito hipotecario",
                    key="esc_sim",
                    label_visibility="collapsed",
                    on_change=no_deseleccionar,
                    args=("esc_sim", "Crédito hipotecario"),
                )
                or "Sin cambios"
            )
            esc = nombres[elegido_esc]
            ch, ce, ng = sm.CreditoHipotecario(), sm.CambioEmpleo(), sm.Negocio()
            if esc == "credito_hipotecario":
                k1, k2, k3 = st.columns(3)
                ch = sm.CreditoHipotecario(
                    anos_de_ingreso=k1.slider(
                        "Monto (años de ingreso)", 1.0, 5.0, 3.0, 0.5
                    ),
                    tasa_anual=k2.slider(
                        "Tasa de interés anual (%)", 5.0, 16.0, 10.0, 0.5
                    )
                    / 100,
                    plazo_meses=k3.slider("Plazo (años)", 5, 30, 20, 5) * 12,
                )
            elif esc == "cambio_empleo":
                k1, k2, k3 = st.columns(3)
                lo, hi = k2.slider("Cambio de sueldo (%)", -30, 50, (-10, 25), 5)
                ce = sm.CambioEmpleo(
                    mes_cambio=k1.slider("Mes del cambio", 1, 24, 6),
                    cambio_salario_min=lo / 100,
                    cambio_salario_max=hi / 100,
                    meses_sin_ingreso_promedio=k3.slider(
                        "Meses sin ingreso (promedio)", 0.0, 4.0, 1.0, 0.5
                    ),
                )
            elif esc == "negocio":
                k1, k2, k3 = st.columns(3)
                ng = sm.Negocio(
                    inversion_meses_ingreso=k1.slider(
                        "Inversión (meses de ingreso)", 1, 24, 6
                    ),
                    margen_medio=k2.slider(
                        "Utilidad esperada (% del ingreso)", -10, 60, 25, 5
                    )
                    / 100,
                    prob_cierre_24m=k3.slider(
                        "Probabilidad de cerrar en 2 años (%)", 0, 60, 30, 5
                    )
                    / 100,
                )
            ins = fila_sim.iloc[0]
            congelar = lambda d: tuple(sorted(d.items()))  # noqa: E731
            res, bnd, info, res_base, bnd_base = simular(
                int(c["cc_num"]),
                float(ins["ingreso"]),
                float(ins["gasto_base"]),
                float(ins["cv_gasto"]),
                esc,
                congelar(sup_dict),
                (
                    ("cambio_empleo", congelar(ce.__dict__)),
                    ("credito_hipotecario", congelar(ch.__dict__)),
                    ("negocio", congelar(ng.__dict__)),
                ),
            )

            # Indicadores
            riesgo = res["prob_deficit"]
            tarjetas = [
                (
                    "Riesgo de números rojos",
                    pct(riesgo),
                    "Sin cambios: " + pct(res_base["prob_deficit"]),
                    e.MANDARIN if riesgo >= de.UMBRAL_RIESGO_DEFICIT else e.LIME,
                ),
                (
                    "Ahorro en 12 meses",
                    dinero(res["ahorro_m12_p50"]),
                    f"90% entre {dinero(res['ahorro_m12_p5'])} y {dinero(res['ahorro_m12_p95'])}",
                    e.ELECTRIC,
                ),
                (
                    "Fondo de emergencia",
                    (
                        f"{res['meses_meta_p50']:.0f} meses"
                        if res["meses_meta_p50"]
                        else "> 36 meses"
                    ),
                    f"Para juntar {dinero(res['meta_fondo_emergencia'])} (6 meses de gasto)",
                    e.ELECTRIC,
                ),
            ]
            if esc == "credito_hipotecario":
                tarjetas.append(
                    (
                        "Pago del crédito",
                        dinero(info["pago_mensual"]),
                        f"{pct(info['pago_entre_ingreso'])} de su ingreso mensual",
                        (
                            e.LIME
                            if (info["pago_entre_ingreso"] or 1) <= 0.35
                            else e.MANDARIN
                        ),
                    )
                )
            elif esc == "negocio":
                tarjetas.append(
                    (
                        "Recupera la inversión",
                        pct(res["prob_recuperar_inversion"]),
                        f"De {dinero(info['inversion'])}, en 36 meses",
                        (
                            e.LIME
                            if (res["prob_recuperar_inversion"] or 0) >= 0.5
                            else e.MANDARIN
                        ),
                    )
                )
            else:
                tarjetas.append(
                    (
                        "Capacidad de ahorro, año 1",
                        pct(res["capacidad_ahorro_12m_p50"]),
                        "Mediana. Sin cambios: "
                        + pct(res_base["capacidad_ahorro_12m_p50"]),
                        e.ELECTRIC,
                    )
                )
            izq, der = st.columns([5, 7], gap="medium")
            with izq:
                cols = st.columns(2)
                for i, (et, val, nota, acento) in enumerate(tarjetas):
                    with cols[i % 2]:
                        html(
                            '<div class="gd-mini">'
                            f'<span class="gd-mini-label">{et}</span>'
                            f'<span class="gd-mini-valor">{val}</span>'
                            f'<span class="gd-kpi-marca" style="background:{acento}"></span>'
                            f'<span class="gd-mini-nota">{nota}</span></div>'
                        )
                filas = "".join(
                    f"<tr><td>Mes {h}</td>"
                    f'<td class="num">{dinero(res[f"flujo_m{h}_p50"])}</td>'
                    f'<td class="num">{dinero(res[f"flujo_m{h}_p5"])} a '
                    f'{dinero(res[f"flujo_m{h}_p95"])}</td></tr>'
                    for h in (1, 3, 6, 12)
                )
                html(
                    '<div class="gd-subtitulo" style="margin-top:6px">Flujo de efectivo '
                    "proyectado</div>"
                    '<table class="gd-tabla"><thead><tr><th>Horizonte</th>'
                    '<th class="num">Mediana</th><th class="num">90% de los casos</th>'
                    f"</tr></thead><tbody>{filas}</tbody></table>"
                )
            with der:
                grafica_abanico(bnd, None if esc == "base" else bnd_base)
                html(
                    '<p class="gd-nota"><b style="color:#001391">Línea azul</b>: mediana. '
                    "Banda oscura: 50% de los casos; banda clara: 90%. "
                    + (
                        "<b>Línea punteada</b>: mediana sin cambios. "
                        if esc != "base"
                        else ""
                    )
                    + "El pico de gasto de cada diciembre (meses 12, 24 y 36) sale de los datos "
                    "reales. Supuestos en docs/fase5_simulador_monte_carlo.md.</p>"
                )

html(
    '<p class="gd-pie">Fuentes: Credit Card Transactions Fraud Detection y Loan Default '
    "Prediction (Kaggle). El ingreso simulado y el préstamo asignado a cada cliente son "
    "construcciones documentadas (Opción A), no datos reales de ninguna persona; los "
    "clientes se muestran con un identificador anónimo. Montos en USD.</p>"
)
