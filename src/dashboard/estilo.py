"""
Estilo visual del Panel Operativo, según la Guía de Estilo que compartió
BBVA para uso académico:

  - Colores primarios: Electric Blue #001391 (principal y cabeceras),
    Serene Blue #85C8FF (apoyo), Midnight #060E46 (texto) y Sand #F7F8F8
    (fondo).
  - Acentos (Lime, Mandarin, Canary, Ice, Purple) SOLO para destacar estados
    y métricas clave; nunca como fondo general.
  - Tipografía: Source Serif 4 para titulares y cifras destacadas; Lato para
    subtítulos, interfaz, tablas y notas.
  - Composición bento: cajas con bordes suaves, títulos alineados y margen
    de seguridad de 40 px.
  - Iconos lineales en Electric Blue con punto inferior.
"""

from html import escape

ELECTRIC = "#001391"
SERENE = "#85C8FF"
MIDNIGHT = "#060E46"
SAND = "#F7F8F8"
BLANCO = "#FFFFFF"
GRIS_1 = "#4B5468"  # texto secundario
GRIS_2 = "#A9B1BD"  # texto tenue / ejes
GRIS_3 = "#DDE1E6"  # bordes
GRIS_4 = "#EEF0F3"  # rejilla y fondos neutros

LIME = "#88E783"
MANDARIN = "#FFB56B"
CANARY = "#FFE761"
ICE = "#8BE1E9"
PURPLE = "#9694FF"

FUENTE_GRAFICAS = "Lato, 'Segoe UI', Arial, sans-serif"

ESTADOS = {
    # estado de Airflow -> (texto, color de acento, símbolo)
    "success": ("Exitosa", LIME, "✓"),
    "failed": ("Falló", MANDARIN, "✕"),
    "upstream_failed": ("No corrió", GRIS_4, "–"),
    "running": ("En curso", CANARY, "●"),
    "queued": ("En cola", CANARY, "●"),
    "up_for_retry": ("Reintentando", CANARY, "↻"),
    "scheduled": ("Programada", GRIS_4, "○"),
    "pendiente": ("Pendiente", GRIS_4, "○"),
    "skipped": ("Omitida", GRIS_4, "–"),
}

FUENTES = (
    "https://fonts.googleapis.com/css2?family=Lato:wght@400;700;900"
    "&family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap"
)

CSS = f"""
<style>
@import url('{FUENTES}');

:root {{
  --electric: {ELECTRIC}; --serene: {SERENE}; --midnight: {MIDNIGHT};
  --sand: {SAND}; --g1: {GRIS_1}; --g2: {GRIS_2}; --g3: {GRIS_3}; --g4: {GRIS_4};
}}
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
  background: var(--sand) !important;
  color: var(--midnight);
  font-family: 'Lato', 'Segoe UI', Arial, sans-serif;
}}
[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer {{
  display: none !important;
}}
[data-testid="stMainBlockContainer"] {{
  max-width: 1440px;
  padding: 40px 40px 48px 40px !important;
}}
p, li, span, div {{ font-family: 'Lato', 'Segoe UI', Arial, sans-serif; }}

/* ---- Cajas bento (contenedores con key="caja_...") ---- */
div[data-testid="stColumn"] div[data-testid="stLayoutWrapper"]:has(> div[class*="st-key-caja"]) {{
  height: 100%;
}}
div[class*="st-key-caja"] {{
  height: 100%;
  background: #FFFFFF;
  border: 1px solid var(--g3);
  border-radius: 16px;
  padding: 22px 24px 18px 24px;
  box-shadow: 0 1px 2px rgba(6, 14, 70, 0.04);
}}

/* ---- Encabezado ---- */
.gd-eyebrow {{
  font: 700 12px/1.2 'Lato', sans-serif; letter-spacing: .12em;
  color: var(--electric); text-transform: uppercase; margin-bottom: 6px;
}}
.gd-h1 {{
  font: 700 40px/1.1 'Source Serif 4', Georgia, serif; color: var(--midnight);
  margin: 0 0 8px 0;
}}
.gd-sub {{ font: 400 16px/1.5 'Lato', sans-serif; color: var(--g1); margin: 0; }}
.gd-header-der {{ display: flex; flex-direction: column; align-items: flex-end; gap: 10px; }}
.gd-tag {{
  display: inline-block; border: 1px solid var(--g3); background: #fff;
  border-radius: 8px; padding: 6px 12px; font: 700 12px 'Lato', sans-serif;
  color: var(--electric);
}}
.gd-pill {{
  display: inline-flex; align-items: center; gap: 8px; border-radius: 999px;
  padding: 7px 14px; font: 700 14px 'Lato', sans-serif; color: var(--midnight);
}}
.gd-meta {{ font: 400 13px 'Lato', sans-serif; color: var(--g1); text-align: right; }}

/* ---- Tarjetas de cifra ---- */
.gd-kpi {{ display: flex; flex-direction: column; gap: 6px; min-height: 132px; }}
.gd-kpi-top {{ display: flex; align-items: center; gap: 10px; }}
.gd-kpi-label {{ font: 700 13px/1.3 'Lato', sans-serif; color: var(--g1); }}
.gd-kpi-valor {{
  font: 700 36px/1.1 'Source Serif 4', Georgia, serif; color: var(--midnight);
  margin-top: 4px;
}}
.gd-kpi-nota {{ font: 400 12px/1.4 'Lato', sans-serif; color: var(--g1); }}
.gd-kpi-marca {{
  display: inline-block; width: 28px; height: 4px; border-radius: 2px; margin-top: 2px;
}}

/* ---- Títulos de caja ---- */
.gd-caja-titulo {{ display: flex; align-items: center; gap: 12px; margin-bottom: 4px; }}
.gd-caja-titulo h3 {{
  font: 700 18px/1.25 'Lato', sans-serif !important; color: var(--midnight) !important;
  margin: 0 !important; padding: 0 !important;
}}
.gd-caja-desc {{ font: 400 13px/1.45 'Lato', sans-serif; color: var(--g1); margin: 0 0 14px 0; }}

/* ---- Icono lineal con punto inferior ---- */
.gd-icono {{
  width: 36px; height: 36px; border-radius: 9px; background: var(--electric);
  display: inline-flex; align-items: center; justify-content: center; flex: none;
}}
.gd-icono.claro {{ background: var(--serene); }}

/* ---- Pipeline ---- */
.gd-carril {{ display: flex; flex-direction: column; gap: 6px; margin: 0 0 14px 0; }}
.gd-carril-nombre {{
  font: 700 11px/1.35 'Lato', sans-serif; color: var(--g1);
  text-transform: uppercase; letter-spacing: .05em;
}}
.gd-pasos {{ display: flex; gap: 8px; min-width: 0; }}
.gd-paso {{
  flex: 1 1 0; min-width: 0; max-width: 200px; border: 1px solid var(--g3);
  border-radius: 12px; padding: 10px 10px; background: #fff; position: relative;
}}
.gd-paso-num {{ font: 700 11px 'Lato', sans-serif; color: var(--electric); }}
.gd-paso-nombre {{
  font: 700 13px/1.3 'Lato', sans-serif; color: var(--midnight); margin: 2px 0 8px 0;
}}
.gd-paso-pie {{ display: flex; flex-direction: column; align-items: flex-start; gap: 4px; }}
.gd-chip {{
  display: inline-flex; align-items: center; gap: 4px; border-radius: 999px;
  padding: 2px 8px; font: 700 11px 'Lato', sans-serif; color: var(--midnight);
  white-space: nowrap;
}}
.gd-dur {{ font: 400 12px 'Lato', sans-serif; color: var(--g1); }}

/* ---- Validaciones ---- */
.gd-val {{
  display: flex; align-items: center; gap: 10px; padding: 9px 0;
  border-bottom: 1px solid var(--g4);
}}
.gd-val:last-child {{ border-bottom: none; }}
.gd-val-marca {{
  width: 22px; height: 22px; border-radius: 6px; flex: none; display: inline-flex;
  align-items: center; justify-content: center; font: 900 13px 'Lato', sans-serif;
  color: var(--midnight);
}}
.gd-val-nombre {{ font: 700 13px/1.3 'Lato', sans-serif; color: var(--midnight); flex: 1; }}
.gd-val-det {{ font: 400 12px 'Lato', sans-serif; color: var(--g1); text-align: right; }}
.gd-subtitulo {{
  font: 700 12px 'Lato', sans-serif; color: var(--electric); text-transform: uppercase;
  letter-spacing: .08em; margin: 4px 0 2px 0;
}}

/* ---- Avisos ---- */
.gd-aviso {{
  border-left: 4px solid var(--mandarin, {MANDARIN}); background: #FFF8F0;
  border-radius: 8px; padding: 12px 14px; font: 400 14px/1.45 'Lato', sans-serif;
  color: var(--midnight); margin-bottom: 8px;
}}
.gd-aviso.info {{ border-left-color: var(--serene); background: #F2F8FF; }}

.gd-pie {{ font: 400 12px/1.5 'Lato', sans-serif; color: var(--g1); margin-top: 8px; }}
</style>
"""

# Trazos de iconos lineales (24x24). Se dibujan en blanco sobre la caja azul,
# con el punto inferior característico de la guía.
_TRAZOS = {
    "datos": '<ellipse cx="12" cy="5" rx="7.5" ry="2.8"/>'
    '<path d="M4.5 5v5.5c0 1.6 3.4 2.8 7.5 2.8s7.5-1.2 7.5-2.8V5"/>'
    '<path d="M4.5 10.5V16c0 1.6 3.4 2.8 7.5 2.8s7.5-1.2 7.5-2.8v-5.5"/>',
    "calidad": '<path d="M12 2.5l7.5 2.8v5.6c0 4.3-3.2 7.9-7.5 8.6-4.3-.7-7.5-4.3-7.5-8.6V5.3z"/>'
    '<path d="M8.6 10.8l2.3 2.3 4.4-4.6"/>',
    "flujo": '<rect x="2.5" y="3" width="7" height="5.5" rx="1.5"/>'
    '<rect x="14.5" y="12" width="7" height="5.5" rx="1.5"/>'
    '<path d="M6 8.5v2.5a3 3 0 0 0 3 3h5.5"/>',
    "barras": '<path d="M5 18.5V11M10 18.5V5M15 18.5v-6M20 18.5V8"/>',
    "capas": '<path d="M12 3l9 4.8-9 4.8-9-4.8z"/><path d="M3 12.6l9 4.8 9-4.8"/>',
    "reloj": '<circle cx="12" cy="10.5" r="8"/><path d="M12 6v4.5l3 2"/>',
    "personas": '<circle cx="9" cy="7" r="3.2"/>'
    '<path d="M2.8 18c.6-3.2 3.1-5 6.2-5s5.6 1.8 6.2 5"/>'
    '<path d="M15.5 4a3.2 3.2 0 0 1 0 6.2M17.6 13.3c1.9.7 3.1 2.3 3.5 4.7"/>',
    "alerta": '<path d="M12 3l9.5 16H2.5z"/><path d="M12 9v4.5"/><path d="M12 16.2v.3"/>',
    "modelo": '<path d="M4 18.5l5-6 4 3.5 7-9"/><path d="M15 7h5v5"/>',
    "prestamo": '<rect x="2.5" y="5" width="19" height="12.5" rx="2"/>'
    '<path d="M2.5 9.5h19"/><path d="M6 14h4"/>',
}


def icono(nombre: str, claro: bool = False) -> str:
    color = MIDNIGHT if claro else "#FFFFFF"
    trazo = _TRAZOS[nombre]
    return (
        f'<span class="gd-icono{" claro" if claro else ""}">'
        f'<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="{color}" '
        f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{trazo}'
        f'<circle cx="12" cy="22.3" r="1.1" fill="{color}" stroke="none"/></svg></span>'
    )


def titulo_caja(texto: str, desc: str = "", ico: str = "", claro: bool = False) -> str:
    cabeza = (
        f'<div class="gd-caja-titulo">{icono(ico, claro) if ico else ""}'
        f"<h3>{escape(texto)}</h3></div>"
    )
    return cabeza + (f'<p class="gd-caja-desc">{escape(desc)}</p>' if desc else "")


def chip_estado(estado: str) -> str:
    texto, color, simbolo = ESTADOS.get(estado, (estado, GRIS_4, "○"))
    return f'<span class="gd-chip" style="background:{color}">{simbolo} {escape(texto)}</span>'


def pill(texto: str, color: str, simbolo: str) -> str:
    return f'<span class="gd-pill" style="background:{color}">{simbolo} {escape(texto)}</span>'


def kpi(etiqueta: str, valor: str, nota: str, ico: str, acento: str = ELECTRIC) -> str:
    return (
        f'<div class="gd-kpi"><div class="gd-kpi-top">{icono(ico)}'
        f'<span class="gd-kpi-label">{escape(etiqueta)}</span></div>'
        f'<div class="gd-kpi-valor">{escape(valor)}</div>'
        f'<span class="gd-kpi-marca" style="background:{acento}"></span>'
        f'<div class="gd-kpi-nota">{escape(nota)}</div></div>'
    )


def validacion(nombre: str, ok: bool, detalle: str) -> str:
    color, simbolo = (LIME, "✓") if ok else (MANDARIN, "✕")
    return (
        f'<div class="gd-val"><span class="gd-val-marca" style="background:{color}">'
        f"{simbolo}</span>"
        f'<span class="gd-val-nombre">{escape(nombre)}</span>'
        f'<span class="gd-val-det">{escape(detalle)}</span></div>'
    )


def aviso(texto: str, info: bool = False) -> str:
    return f'<div class="gd-aviso{" info" if info else ""}">{texto}</div>'


def formato_entero(n) -> str:
    return "—" if n is None else f"{int(n):,}"


def formato_duracion(segundos) -> str:
    if segundos is None or segundos != segundos:  # None o NaN
        return "—"
    segundos = float(segundos)
    if segundos < 60:
        return f"{segundos:.0f} s"
    return f"{int(segundos // 60)} min {int(round(segundos % 60)):02d} s"


def tema_altair() -> dict:
    """Configuración común de las gráficas: tipografía Lato, ejes y rejilla
    discretos, texto en tinta (nunca del color de la serie)."""
    return {
        "config": {
            "font": FUENTE_GRAFICAS,
            "background": "#FFFFFF",
            "view": {"stroke": None},
            "axis": {
                "labelColor": GRIS_1,
                "labelFont": FUENTE_GRAFICAS,
                "labelFontSize": 11,
                "titleColor": GRIS_1,
                "titleFont": FUENTE_GRAFICAS,
                "titleFontSize": 11,
                "titleFontWeight": "normal",
                "gridColor": GRIS_4,
                "domain": False,
                "tickColor": GRIS_3,
            },
            "legend": {
                "labelColor": MIDNIGHT,
                "labelFont": FUENTE_GRAFICAS,
                "labelFontSize": 12,
                "titleColor": GRIS_1,
                "titleFont": FUENTE_GRAFICAS,
                "orient": "top",
                "symbolType": "square",
            },
        }
    }
