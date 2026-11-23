"""
Reporte ejecutivo del portafolio en PDF.

Una o dos páginas con lo que un comité necesita ver sin abrir el dashboard:
KPIs del portafolio (o de un segmento de riesgo), niveles de riesgo,
hallazgos clave, impacto de los escenarios del simulador y la lista de
alertas tempranas de impago.

Los números salen de las mismas funciones que usa el Dashboard Ejecutivo
(src/dashboard/datos_ejecutivo.py), así que el PDF y la pantalla siempre
coinciden. El dashboard lo genera con el botón «Descargar reporte (PDF)».

Usa fpdf2 (sin navegador ni dependencias del sistema) y solo dibuja
rectángulos y texto, así que también corre en la imagen liviana del
dashboard.
"""

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from fpdf import FPDF

from src.dashboard import datos_ejecutivo as de
from src.ml.explicacion import CORTES_RIESGO, NIVELES

# Colores de la guía de estilo de BBVA (los mismos de src/dashboard/estilo.py).
ELECTRIC = (0, 19, 145)
MIDNIGHT = (6, 14, 70)
SERENE = (133, 200, 255)
SAND = (247, 248, 248)
GRIS = (98, 107, 120)
LINEA = (221, 225, 230)
BLANCO = (255, 255, 255)
COLOR_NIVEL = {
    "Bajo": (136, 231, 131),
    "Medio": (255, 231, 97),
    "Alto": (255, 181, 107),
}
RANGO_NIVEL = {
    "Bajo": f"menos de {CORTES_RIESGO[0]:.0%}",
    "Medio": f"{CORTES_RIESGO[0]:.0%} a {CORTES_RIESGO[1]:.0%}",
    "Alto": f"{CORTES_RIESGO[1]:.0%} o más",
}
MAX_ALERTAS = 15

# Las fuentes estándar del PDF solo traen Latin-1: se cambian los símbolos
# que no existen ahí por su equivalente más cercano.
REEMPLAZOS = {
    "−": "-",
    "–": "-",
    "—": "-",
    "≥": ">=",
    "≤": "<=",
    "→": "->",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "…": "...",
}


def latin(texto) -> str:
    texto = str(texto)
    for a, b in REEMPLAZOS.items():
        texto = texto.replace(a, b)
    return texto.encode("latin-1", "replace").decode("latin-1")


def dinero(x) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


def pct(x, dec=0) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{x * 100:.{dec}f}%"


# ---------------------------------------------------------------------------
# Datos del reporte
# ---------------------------------------------------------------------------


@dataclass
class DatosReporte:
    segmento: str
    total_clientes: int
    periodo: str
    resumen: dict
    niveles: pd.DataFrame
    alertas: pd.DataFrame
    escenarios: pd.DataFrame = None
    hallazgos: list = field(default_factory=list)

    @property
    def titulo_segmento(self) -> str:
        if self.segmento == "Todos":
            return "Todo el portafolio"
        return f"Segmento de riesgo {self.segmento.lower()}"


def hallazgos(r: dict, total: int, segmento: str, esc: pd.DataFrame = None) -> list:
    """Frases clave calculadas a partir de los datos (no se escriben a mano)."""
    n = r["clientes"]
    if not n:
        return ["El segmento no tiene clientes."]
    quien = "del portafolio" if segmento == "Todos" else "del segmento"
    frases = []
    if segmento == "Todos":
        frases.append(
            f"{r['riesgo_alto']:,} de {n:,} clientes ({r['riesgo_alto'] / n:.0%}) "
            f"están en riesgo alto de impago (probabilidad de {CORTES_RIESGO[1]:.0%} "
            "o más)."
        )
    else:
        frases.append(
            f"El segmento reúne {n:,} clientes, el {n / max(total, 1):.0%} del "
            "portafolio."
        )
    frases.append(
        f"{r['ahorro_negativo']:,} clientes {quien} gastan más de lo que ganan; la "
        f"capacidad de ahorro típica es de {pct(r['ahorro'])}."
    )
    frases.append(
        f"{r['dti_mayor_1']:,} clientes tienen un pago mensual de deuda mayor que su "
        f"ingreso (DTI mayor a 1). El DTI típico es {r['dti']:.2f}."
    )
    if esc is not None and {"base", "credito_hipotecario"} <= set(esc.index):
        b0 = esc.loc["base", "en_riesgo"]
        b1 = esc.loc["credito_hipotecario", "en_riesgo"]
        frases.append(
            "Con un crédito hipotecario, los clientes en riesgo de números rojos "
            f"pasarían de {b0:.0%} a {b1:.0%} en los próximos 36 meses."
        )
    return frases


def armar_datos(
    clientes: pd.DataFrame,
    aportes: pd.DataFrame = None,
    impago_real: dict = None,
    resumen_sim: pd.DataFrame = None,
    segmento: str = "Todos",
    periodo: str = "",
) -> DatosReporte:
    sub = clientes if segmento == "Todos" else clientes[clientes["nivel"] == segmento]
    r = de.resumen_portafolio(sub)
    esc = (
        None
        if resumen_sim is None or sub.empty
        else de.escenarios_portafolio(resumen_sim, sub)
    )
    return DatosReporte(
        segmento=segmento,
        total_clientes=len(clientes),
        periodo=periodo,
        resumen=r,
        niveles=de.tabla_niveles(clientes, impago_real),
        alertas=de.alertas(sub, aportes),
        escenarios=esc,
        hallazgos=hallazgos(r, len(clientes), segmento, esc),
    )


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


class _Pdf(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_draw_color(*LINEA)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.set_font("helvetica", "", 7.5)
        self.set_text_color(*GRIS)
        self.cell(
            0,
            7,
            latin(
                "Gemelo Digital Financiero · Uso académico con datos simulados: son "
                "estimaciones, no asesoría financiera"
            ),
        )
        self.set_x(self.l_margin)
        self.cell(0, 7, f"Página {self.page_no()} de {{nb}}", align="R")


def _titulo(pdf: FPDF, texto: str, nota: str = ""):
    if pdf.get_y() > pdf.h - 60:
        pdf.add_page()
    pdf.ln(4)
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(*MIDNIGHT)
    pdf.cell(0, 7, latin(texto), new_x="LMARGIN", new_y="NEXT")
    if nota:
        pdf.set_font("helvetica", "", 8.5)
        pdf.set_text_color(*GRIS)
        pdf.multi_cell(0, 4.2, latin(nota), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)


def _encabezado(pdf: FPDF, d: DatosReporte, generado: datetime):
    pdf.set_fill_color(*ELECTRIC)
    pdf.rect(0, 0, pdf.w, 34, style="F")
    pdf.set_xy(pdf.l_margin, 8)
    pdf.set_font("helvetica", "B", 8)
    pdf.set_text_color(*SERENE)
    pdf.cell(0, 4, "REPORTE EJECUTIVO DEL PORTAFOLIO", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "B", 19)
    pdf.set_text_color(*BLANCO)
    pdf.cell(0, 10, "Gemelo Digital Financiero", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 9)
    pdf.cell(
        0,
        5,
        latin(
            f"{d.titulo_segmento} · {d.resumen['clientes']:,} clientes"
            + (f" · histórico {d.periodo}" if d.periodo else "")
        ),
    )
    pdf.set_xy(pdf.l_margin, 21)
    pdf.cell(0, 5, latin(f"Generado el {generado:%d/%m/%Y %H:%M} h"), align="R")
    pdf.set_xy(pdf.l_margin, 26)
    pdf.cell(0, 5, "Fuente: capa Gold del pipeline", align="R")
    pdf.set_y(40)


def _kpis(pdf: FPDF, r: dict):
    tarjetas = [
        ("Clientes", f"{r['clientes']:,}", "Con KPIs, préstamo y riesgo"),
        ("Gasto mensual típico", dinero(r["gasto_mensual"]), "Mediana, USD"),
        (
            "Capacidad de ahorro",
            pct(r["ahorro"]),
            f"{r['ahorro_negativo']:,} gastan más de lo que ganan",
        ),
        (
            "Endeudamiento (DTI)",
            "-" if r["dti"] is None else f"{r['dti']:.2f}",
            f"{r['dti_mayor_1']:,} pagan más que su ingreso",
        ),
        (
            "Probabilidad de impago",
            pct(r["prob_impago"], 1),
            f"{r['riesgo_alto']:,} en riesgo alto",
        ),
    ]
    sep = 3
    ancho = (pdf.epw - sep * (len(tarjetas) - 1)) / len(tarjetas)
    y = pdf.get_y()
    for i, (etiqueta, valor, nota) in enumerate(tarjetas):
        x = pdf.l_margin + i * (ancho + sep)
        pdf.set_fill_color(*SAND)
        pdf.set_draw_color(*LINEA)
        pdf.rect(x, y, ancho, 25, style="DF")
        pdf.set_fill_color(*ELECTRIC)
        pdf.rect(x, y, 1.2, 25, style="F")
        pdf.set_xy(x + 3, y + 2.5)
        pdf.set_font("helvetica", "", 7.5)
        pdf.set_text_color(*GRIS)
        pdf.cell(ancho - 4, 4, latin(etiqueta))
        pdf.set_xy(x + 3, y + 7.5)
        pdf.set_font("helvetica", "B", 15)
        pdf.set_text_color(*MIDNIGHT)
        pdf.cell(ancho - 4, 8, latin(valor))
        pdf.set_xy(x + 3, y + 16.5)
        pdf.set_font("helvetica", "", 6.8)
        pdf.set_text_color(*GRIS)
        pdf.multi_cell(ancho - 5, 3.2, latin(nota), align="L")
    pdf.set_y(y + 28)


def _hallazgos(pdf: FPDF, frases: list):
    _titulo(pdf, "Hallazgos clave")
    for frase in frases:
        y = pdf.get_y()
        pdf.set_fill_color(*ELECTRIC)
        pdf.rect(pdf.l_margin + 1, y + 1.6, 1.8, 1.8, style="F")
        pdf.set_xy(pdf.l_margin + 5, y)
        pdf.set_font("helvetica", "", 9.5)
        pdf.set_text_color(*MIDNIGHT)
        pdf.multi_cell(pdf.epw - 5, 5, latin(frase), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(0.8)


def _niveles(pdf: FPDF, d: DatosReporte):
    t = d.niveles
    alto = t.loc["Alto"]
    nota = (
        "Todo el portafolio, por probabilidad de impago del modelo. "
        f"El {alto['pct']:.0%} de los clientes está en riesgo alto."
    )
    if "impago_real" in t and pd.notna(alto.get("impago_real")):
        nota += (
            f" En préstamos con ese perfil, {alto['impago_real'] * 100:.0f} de cada 100 "
            "no se pagaron (255 mil préstamos del dataset Loan Default)."
        )
    _titulo(pdf, "Niveles de riesgo de impago", nota)
    x0, ancho_etq, ancho_barra = pdf.l_margin, 46, 82
    for nivel in NIVELES:
        f = t.loc[nivel]
        y = pdf.get_y()
        pdf.set_fill_color(*COLOR_NIVEL[nivel])
        pdf.rect(x0, y + 1, 3, 3, style="F")
        pdf.set_xy(x0 + 5, y)
        pdf.set_font("helvetica", "B", 9.5)
        pdf.set_text_color(*MIDNIGHT)
        pdf.cell(ancho_etq - 5, 5, latin(f"Riesgo {nivel.lower()}"))
        pdf.set_xy(x0 + 5, y + 4.5)
        pdf.set_font("helvetica", "", 7.5)
        pdf.set_text_color(*GRIS)
        pdf.cell(ancho_etq - 5, 4, latin(RANGO_NIVEL[nivel]))
        xb = x0 + ancho_etq
        pdf.set_fill_color(*SAND)
        pdf.rect(xb, y + 1, ancho_barra, 5, style="F")
        pdf.set_fill_color(*COLOR_NIVEL[nivel])
        pdf.rect(xb, y + 1, max(ancho_barra * float(f["pct"]), 0.5), 5, style="F")
        pdf.set_xy(xb + ancho_barra + 3, y)
        pdf.set_font("helvetica", "B", 9.5)
        pdf.set_text_color(*MIDNIGHT)
        pdf.cell(28, 5, latin(f"{int(f['clientes']):,} · {f['pct']:.0%}"))
        pdf.set_xy(xb + ancho_barra + 3, y + 4.5)
        pdf.set_font("helvetica", "", 7.5)
        pdf.set_text_color(*GRIS)
        detalle = f"Prob. promedio {f['prob_promedio']:.1%}"
        if "impago_real" in t and pd.notna(f.get("impago_real")):
            detalle += f" · impago real {f['impago_real']:.1%}"
        pdf.cell(0, 4, latin(detalle))
        pdf.set_y(y + 11)


def _escenarios(pdf: FPDF, d: DatosReporte):
    esc = d.escenarios
    if esc is None:
        return
    _titulo(
        pdf,
        "Escenarios a futuro (simulador Monte Carlo)",
        "1,000 futuros posibles por cliente a 36 meses. «En riesgo» = en al menos 10% "
        "de sus futuros, su ahorro acumulado baja de cero.",
    )
    tope = max(float(esc["en_riesgo"].max()), 0.01)
    x0, ancho_etq, ancho_barra = pdf.l_margin, 46, 82
    for clave, f in esc.iterrows():
        y = pdf.get_y()
        pdf.set_xy(x0, y)
        pdf.set_font("helvetica", "B" if clave == "base" else "", 9)
        pdf.set_text_color(*MIDNIGHT)
        pdf.cell(ancho_etq, 5, latin(f["nombre"]))
        xb = x0 + ancho_etq
        pdf.set_fill_color(*SAND)
        pdf.rect(xb, y + 0.5, ancho_barra, 4.5, style="F")
        pdf.set_fill_color(*(SERENE if clave == "base" else ELECTRIC))
        largo = ancho_barra * float(f["en_riesgo"]) / tope
        pdf.rect(xb, y + 0.5, max(largo, 0.5), 4.5, style="F")
        pdf.set_xy(xb + ancho_barra + 3, y)
        pdf.set_font("helvetica", "B", 9)
        pdf.cell(16, 5, latin(pct(f["en_riesgo"], 1)))
        pdf.set_font("helvetica", "", 7.5)
        pdf.set_text_color(*GRIS)
        pdf.cell(0, 5, latin(f"ahorro a 12 meses {dinero(f['ahorro_12m'])}"))
        pdf.set_y(y + 7)


def _alertas(pdf: FPDF, d: DatosReporte):
    al = d.alertas
    if al.empty:
        _titulo(
            pdf,
            "Alertas tempranas de impago",
            "Este segmento no tiene clientes en riesgo alto.",
        )
        return
    mostrados = min(len(al), MAX_ALERTAS)
    # La lista va completa en su propia página para no partir la tabla.
    pdf.add_page()
    _titulo(
        pdf,
        "Alertas tempranas de impago",
        f"{len(al):,} clientes con probabilidad de impago de {CORTES_RIESGO[1]:.0%} o "
        f"más. Se muestran los {mostrados} más altos; el detalle de cada uno está en "
        "el Perfil 360 del dashboard.",
    )
    cols = [
        ("Cliente", 24, "L"),
        ("Prob. impago", 22, "R"),
        ("Lo que más sube su riesgo", 78, "L"),
        ("DTI", 16, "R"),
        ("Ahorro", 18, "R"),
        ("Gasto mensual", 0, "R"),
    ]
    ultimo = pdf.epw - sum(c[1] for c in cols)
    cols[-1] = (cols[-1][0], ultimo, "R")

    def encabezado():
        pdf.set_fill_color(*ELECTRIC)
        pdf.set_text_color(*BLANCO)
        pdf.set_font("helvetica", "B", 8)
        for nombre, ancho, alin in cols:
            pdf.cell(ancho, 7, latin(f" {nombre} "), fill=True, align=alin)
        pdf.ln(7)

    encabezado()
    pdf.set_font("helvetica", "", 8)
    for i, a in enumerate(al.head(MAX_ALERTAS).itertuples()):
        if pdf.get_y() > pdf.h - 22:
            pdf.add_page()
            encabezado()
            pdf.set_font("helvetica", "", 8)
        pdf.set_fill_color(*(SAND if i % 2 else BLANCO))
        pdf.set_text_color(*MIDNIGHT)
        valores = [
            a.cliente,
            pct(a.prob_impago, 1),
            a.factor,
            f"{a.dti:.2f}",
            pct(a.capacidad_ahorro),
            dinero(a.gasto_promedio_mensual),
        ]
        for (_, ancho, alin), v in zip(cols, valores):
            texto = latin(v)
            while pdf.get_string_width(f" {texto} ") > ancho - 1 and len(texto) > 4:
                texto = texto[:-4] + "..."
            pdf.cell(ancho, 6.5, f" {texto} ", fill=True, align=alin)
        pdf.ln(6.5)


def _notas(pdf: FPDF):
    _titulo(pdf, "Cómo leer este reporte")
    pdf.set_font("helvetica", "", 8)
    pdf.set_text_color(*GRIS)
    notas = [
        "Fuentes: Credit Card Transactions Fraud Detection y Loan Default Prediction "
        "(Kaggle). El ingreso y el préstamo asignado a cada cliente son construcciones "
        "documentadas; los clientes se muestran con un identificador anónimo.",
        "Los datos pasan por las capas Bronze, Silver y Gold y por reglas de calidad en "
        "cada corrida del pipeline. Los montos están en USD.",
        "La probabilidad de impago sale de un modelo de regresión logística explicable; "
        f"los cortes de riesgo ({CORTES_RIESGO[0]:.0%} y {CORTES_RIESGO[1]:.0%}) se "
        "validaron con BBVA.",
    ]
    for n in notas:
        pdf.multi_cell(0, 4, latin(n), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)


def generar_pdf(d: DatosReporte, generado: datetime = None) -> bytes:
    """El reporte como bytes, listo para descargar o guardar."""
    generado = generado or datetime.now()
    pdf = _Pdf(format="Letter")
    pdf.set_margins(14, 14, 14)
    pdf.set_auto_page_break(True, margin=16)
    pdf.set_title(latin(f"Reporte ejecutivo - {d.titulo_segmento}"))
    pdf.set_author("Gemelo Digital Financiero")
    pdf.add_page()
    _encabezado(pdf, d, generado)
    _kpis(pdf, d.resumen)
    _hallazgos(pdf, d.hallazgos)
    _niveles(pdf, d)
    _escenarios(pdf, d)
    _alertas(pdf, d)
    _notas(pdf)
    return bytes(pdf.output())


def nombre_archivo(d: DatosReporte, generado: datetime = None) -> str:
    generado = generado or datetime.now()
    seg = "portafolio" if d.segmento == "Todos" else f"riesgo_{d.segmento.lower()}"
    return f"reporte_ejecutivo_{seg}_{generado:%Y%m%d}.pdf"
