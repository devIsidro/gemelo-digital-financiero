"""
Reporte para la revisión de distribuciones y casos representativos
(pedido de Eduardo, BBVA, antes de cerrar la parte de KPIs y modelo).

Genera las gráficas de docs/fase5_revision_distribuciones.md a partir de
la capa Gold, el Silver de préstamos y las métricas del modelo, e imprime
en pantalla las tablas que usa el documento.

No es parte del DAG: se corre a mano cuando cambian los datos.

    python -m src.reportes.revision_distribuciones

Necesita pandas, numpy y matplotlib (no Spark).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.ml.explicacion import explicar  # noqa: E402

# Colores de la Guía de Estilo de BBVA (los mismos de src/dashboard/estilo.py).
ELECTRIC = "#001391"
SERENE = "#85C8FF"
MIDNIGHT = "#060E46"
MANDARIN = "#FFB56B"
GRIS_1 = "#4B5468"
GRIS_3 = "#DDE1E6"

SALIDA = Path("docs/img/revision")

# Cortes de riesgo que se proponen (para decidir con Eduardo).
CORTES = [0.10, 0.30]
BANDAS = ["Bajo (< 10%)", "Medio (10% – 30%)", "Alto (≥ 30%)"]

NOMBRES = {
    "age": "Edad",
    "interest_rate": "Tasa de interés",
    "months_employed": "Meses empleado",
    "dti": "DTI",
    "ingreso_mensual_fuente": "Ingreso (fuente)",
    "loan_amount": "Monto del préstamo",
    "has_cosigner": "Tiene aval",
    "loan_term": "Plazo",
    "has_dependents": "Tiene dependientes",
    "credit_score": "Score de crédito",
    "num_credit_lines": "Líneas de crédito",
    "has_mortgage": "Tiene hipoteca",
    "employment_type": "Empleo",
    "marital_status": "Estado civil",
    "loan_purpose": "Propósito",
    "education": "Escolaridad",
}
VALORES = {
    "Full-time": "tiempo completo",
    "Part-time": "medio tiempo",
    "Self-employed": "independiente",
    "Unemployed": "desempleado",
    "Married": "casado",
    "Single": "soltero",
    "Divorced": "divorciado",
    "Home": "vivienda",
    "Auto": "auto",
    "Business": "negocio",
    "Education": "educación",
    "Other": "otro",
    "High School": "preparatoria",
    "Bachelor's": "licenciatura",
    "Master's": "maestría",
    "PhD": "doctorado",
}


def estilo():
    plt.rcParams.update(
        {
            "font.family": ["Lato", "DejaVu Sans"],
            "font.size": 10,
            "axes.edgecolor": GRIS_3,
            "axes.labelcolor": GRIS_1,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlecolor": MIDNIGHT,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#EEF0F3",
            "xtick.color": GRIS_1,
            "ytick.color": GRIS_1,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------


def cargar(base: Path = Path("data")) -> tuple:
    kpis = pd.read_parquet(base / "gold" / "kpis_cliente")
    prob = pd.read_parquet(base / "gold" / "prob_impago_cliente")
    loans = pd.read_parquet(base / "silver" / "loan_default")
    metricas = json.loads(
        (base / "gold" / "modelo_impago_metricas.json").read_text(encoding="utf-8")
    )
    clientes = kpis.merge(prob[["cc_num", "prob_impago"]], on="cc_num").merge(
        loans, on="loan_id", how="left"
    )
    clientes["dti"] = clientes["ratio_endeudamiento"]
    return clientes, preparar_prestamos(loans), metricas


def preparar_prestamos(loans: pd.DataFrame) -> pd.DataFrame:
    """Mismas variables derivadas que modelo_impago.preparar_variables."""
    loans = loans.copy()
    r = loans["interest_rate"] / 100 / 12
    n = loans["loan_term"]
    pago = loans["loan_amount"] * r / (1 - (1 + r) ** (-n))
    loans["ingreso_mensual_fuente"] = loans["income_anual"] / 12
    loans["dti"] = pago / loans["ingreso_mensual_fuente"]
    return loans


def casos_representativos(c: pd.DataFrame) -> pd.DataFrame:
    """Los mismos criterios de docs/fase4_decisiones_kpis.md, más el cliente
    con mayor riesgo."""

    def cercano(serie, valor):
        return (serie - valor).abs().idxmin()

    gasto = c["gasto_promedio_mensual"]
    dti_alto = c[c["dti"] > 1]
    filas = {
        "A: gasto bajo (p10)": cercano(gasto, gasto.quantile(0.10)),
        "B: típico (mediana)": cercano(gasto, gasto.median()),
        "C: gasto alto (p90)": cercano(gasto, gasto.quantile(0.90)),
        "D: ahorro negativo": c["capacidad_ahorro"].idxmin(),
        "E: DTI > 1": cercano(dti_alto["dti"], dti_alto["dti"].median()),
        "F: mayor riesgo": c["prob_impago"].idxmax(),
    }
    casos = c.loc[list(filas.values())].copy()
    casos.index = list(filas.keys())
    return casos


def nombre_variable(v: str, fila: pd.Series) -> str:
    """Nombre legible con el valor del cliente, p. ej. 'Edad: 23'."""
    if "=" in v:
        col, val = v.split("=", 1)
        propio = fila[col]
        estado = "sí" if propio == val else "no"
        return f"{NOMBRES[col]} {VALORES.get(val, val)}: {estado}"
    valor = fila[v]
    if v in ("has_cosigner", "has_dependents", "has_mortgage"):
        texto = "sí" if valor == 1 else "no"
    elif v == "interest_rate":
        texto = f"{valor:.1f}%"
    elif v == "dti":
        texto = f"{valor:.2f}"
    elif v in ("loan_amount", "ingreso_mensual_fuente"):
        texto = f"{valor:,.0f} USD"
    elif v == "loan_term":
        texto = f"{valor:.0f} meses"
    else:
        texto = f"{valor:,.0f}"
    return f"{NOMBRES[v]}: {texto}"


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------


def _mediana(ax, valor, texto):
    ax.axvline(valor, color=MIDNIGHT, lw=1.2, ls="--")
    ax.annotate(
        texto,
        xy=(valor, 1),
        xycoords=("data", "axes fraction"),
        xytext=(4, -2),
        textcoords="offset points",
        va="top",
        fontsize=9,
        color=MIDNIGHT,
    )


def grafica_distribuciones(c: pd.DataFrame, ruta: Path):
    fig, ejes = plt.subplots(2, 3, figsize=(13, 7))
    ejes = ejes.ravel()

    # 1. Los dos ingresos
    ax = ejes[0]
    bins = np.arange(0, 36_000, 1_000)
    ax.hist(c["ingreso_mensual_fuente"], bins=bins, color=ELECTRIC, label="Fuente")
    ax.hist(c["ingreso_mensual_simulado"], bins=bins, color=SERENE, label="Simulado")
    ax.set_title("Ingreso mensual: fuente vs simulado")
    ax.set_xlabel("USD al mes")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1000:.0f}k")

    # 2. Score
    ax = ejes[1]
    ax.hist(c["score_credito_fuente"], bins=np.arange(300, 851, 25), color=ELECTRIC)
    _mediana(ax, c["score_credito_fuente"].median(), "mediana")
    ax.set_title("Score de crédito (fuente)")
    ax.set_xlabel("Puntos")

    # 3. Pago mensual
    ax = ejes[2]
    ax.hist(
        c["pago_mensual_prestamo"], bins=np.arange(0, 24_000, 1_000), color=ELECTRIC
    )
    _mediana(ax, c["pago_mensual_prestamo"].median(), "mediana")
    ax.set_title("Pago mensual del préstamo")
    ax.set_xlabel("USD al mes")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1000:.0f}k")

    # 4. DTI (cola larga: los mayores a 4 se juntan en la última barra)
    ax = ejes[3]
    dti = c["dti"].clip(upper=4.0)
    ax.hist(dti, bins=np.arange(0, 4.01, 0.2), color=ELECTRIC)
    ax.axvspan(1, 4.05, color=MANDARIN, alpha=0.18, lw=0, zorder=0)
    ax.annotate(
        f"DTI > 1: {(c['dti'] > 1).sum()} clientes",
        xy=(2.5, 0.6),
        xycoords=("data", "axes fraction"),
        ha="center",
        fontsize=9,
        color=GRIS_1,
    )
    _mediana(ax, c["dti"].median(), "mediana")
    ax.set_title("DTI (pago mensual / ingreso de la fuente)")
    ax.set_xlabel("Veces el ingreso mensual (la última barra junta 4 o más)")

    # 5. Capacidad de ahorro
    ax = ejes[4]
    ax.hist(c["capacidad_ahorro"] * 100, bins=np.arange(-25, 100, 5), color=SERENE)
    _mediana(ax, c["capacidad_ahorro"].median() * 100, "mediana")
    ax.set_title("Capacidad de ahorro (ingreso simulado)")
    ax.set_xlabel("% del ingreso")

    # 6. Probabilidad de impago
    ax = ejes[5]
    ax.hist(c["prob_impago"] * 100, bins=np.arange(0, 90, 2.5), color=ELECTRIC)
    _mediana(ax, c["prob_impago"].median() * 100, "mediana")
    ax.set_title("Probabilidad de impago del modelo")
    ax.set_xlabel("%")

    for ax in ejes:
        ax.set_ylabel("Clientes")
    fig.tight_layout(h_pad=2.5, w_pad=2)
    fig.savefig(ruta)
    plt.close(fig)


def impago_por_decil(loans: pd.DataFrame) -> pd.DataFrame:
    calc = loans.groupby(pd.qcut(loans["dti"], 10, labels=False))["is_default"].mean()
    fuente = loans.groupby(pd.qcut(loans["dti_ratio"], 10, labels=False))[
        "is_default"
    ].mean()
    return pd.DataFrame({"calculado": calc, "fuente": fuente}) * 100


def grafica_dti_vs_impago(loans: pd.DataFrame, ruta: Path) -> pd.DataFrame:
    tabla = impago_por_decil(loans)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(1, 11)
    ancho = 0.4
    ax.bar(
        x - ancho / 2,
        tabla["calculado"],
        ancho,
        color=ELECTRIC,
        label="DTI calculado (el nuestro)",
    )
    ax.bar(
        x + ancho / 2,
        tabla["fuente"],
        ancho,
        color=SERENE,
        label="DTIRatio del dataset",
    )
    base = loans["is_default"].mean() * 100
    ax.axhline(base, color=GRIS_1, lw=1, ls="--")
    ax.annotate(
        f"promedio {base:.1f}%",
        xy=(0.55, base),
        xytext=(0, 4),
        textcoords="offset points",
        ha="left",
        fontsize=9,
        color=GRIS_1,
    )
    ax.set_xticks(x, [f"{i}" for i in x])
    ax.set_xlabel("Decil de DTI (1 = el 10% con menor DTI, 10 = el 10% con mayor DTI)")
    ax.set_ylabel("% de préstamos en impago")
    ax.legend(loc="upper left")
    fig.savefig(ruta)
    plt.close(fig)
    return tabla


def grafica_casos(casos: pd.DataFrame, aportes: pd.DataFrame, ruta: Path):
    fig, ejes = plt.subplots(2, 3, figsize=(14, 7.5))
    # Misma escala en los seis para poder compararlos.
    limite = float(aportes.abs().max().max()) * 1.3
    for ax, (nombre, fila) in zip(ejes.ravel(), casos.iterrows()):
        a = aportes.loc[nombre]
        top = a.reindex(a.abs().sort_values(ascending=False).index[:6])[::-1]
        etiquetas = [nombre_variable(v, fila) for v in top.index]
        colores = [MANDARIN if v > 0 else ELECTRIC for v in top]
        ax.barh(etiquetas, top, color=colores, height=0.6)
        for y, v in enumerate(top):
            ax.annotate(
                f"{v:+.2f}",
                xy=(v, y),
                xytext=(4 if v > 0 else -4, 0),
                textcoords="offset points",
                ha="left" if v > 0 else "right",
                va="center",
                fontsize=8.5,
                color=GRIS_1,
            )
        ax.axvline(0, color=GRIS_1, lw=0.8)
        ax.set_title(f"{nombre} · {fila['prob_impago']:.1%}")
        ax.set_xlim(-limite, limite)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=9)
    for ax in ejes[1]:
        ax.set_xlabel("← baja el riesgo      sube el riesgo →")
    fig.tight_layout(h_pad=2.5, w_pad=1.5)
    fig.savefig(ruta)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------


def banda(prob: pd.Series) -> pd.Series:
    return pd.cut(prob, [0, *CORTES, 1.0], right=False, labels=BANDAS)


def tabla_bandas(clientes: pd.DataFrame, loans: pd.DataFrame, metricas: dict):
    prob_loans, _ = explicar(loans, metricas)
    l_banda = banda(pd.Series(prob_loans, index=loans.index))
    validacion = loans.groupby(l_banda, observed=False)["is_default"].agg(
        ["size", "mean"]
    )
    c_banda = banda(clientes["prob_impago"]).value_counts().reindex(BANDAS)
    return pd.DataFrame(
        {
            "clientes": c_banda,
            "pct_clientes": c_banda / len(clientes) * 100,
            "prestamos": validacion["size"],
            "impago_real_pct": validacion["mean"] * 100,
        }
    )


def resumen(serie: pd.Series) -> list:
    return [serie.min(), *serie.quantile([0.10, 0.5, 0.90]).tolist(), serie.max()]


def main():
    estilo()
    SALIDA.mkdir(parents=True, exist_ok=True)
    clientes, loans, metricas = cargar()

    grafica_distribuciones(clientes, SALIDA / "distribuciones.png")
    deciles = grafica_dti_vs_impago(loans, SALIDA / "dti_vs_impago.png")

    casos = casos_representativos(clientes)
    prob, aportes = explicar(casos, metricas)
    aportes.index = casos.index
    grafica_casos(casos, aportes, SALIDA / "casos_aportes.png")

    print("== Diferencia máxima entre la explicación y prob_impago:")
    print(f"   {np.abs(prob - casos['prob_impago'].to_numpy()).max():.4f}")
    print("\n== Distribuciones (mín, p10, mediana, p90, máx)")
    for col in [
        "ingreso_mensual_fuente",
        "ingreso_mensual_simulado",
        "score_credito_fuente",
        "pago_mensual_prestamo",
        "dti",
        "capacidad_ahorro",
        "prob_impago",
    ]:
        print(f"   {col}: {[round(v, 4) for v in resumen(clientes[col])]}")
    print(f"   DTI > 1: {(clientes['dti'] > 1).sum()} clientes")
    print("\n== Impago real por decil de DTI (%)")
    print(deciles.round(1).T.to_string())
    print("\n== Bandas de riesgo propuestas")
    print(tabla_bandas(clientes, loans, metricas).round(1).to_string())
    print("\n== Casos representativos")
    columnas = [
        "gasto_promedio_mensual",
        "capacidad_ahorro",
        "ingreso_mensual_fuente",
        "loan_amount",
        "interest_rate",
        "loan_term",
        "dti",
        "credit_score",
        "age",
        "months_employed",
        "employment_type",
        "has_cosigner",
        "prob_impago",
    ]
    print(casos[columnas].T.to_string())
    print("\n== Aportes principales por caso")
    for nombre, fila in casos.iterrows():
        a = aportes.loc[nombre]
        top = a.reindex(a.abs().sort_values(ascending=False).index[:5])
        partes = [f"{nombre_variable(v, fila)} ({x:+.2f})" for v, x in top.items()]
        print(f"   {nombre}: " + "; ".join(partes))


if __name__ == "__main__":
    main()
