"""
Linaje de datos (Panel Operativo).

Para cada indicador que ve el negocio (dashboard ejecutivo, reporte PDF y
asistente de IA) dice de dónde sale: desde el archivo original de Kaggle,
pasando por Bronze, Silver y Gold, con el paso del DAG y el código que hace
cada transformación. Así cualquier número se puede seguir hasta su origen,
que es lo que pide una auditoría.

Las tablas son las mismas del catálogo de datos (catalogo.py): el panel
junta este mapa con el estado de cada tabla (filas y última actualización),
y una prueba revisa que todas las tablas del linaje existan en el catálogo.
"""

from dataclasses import dataclass

import pandas as pd

CAPAS = ("Fuente", "Bronze", "Silver", "Gold", "Uso")

# Colores por capa (guía de estilo de BBVA, src/dashboard/estilo.py).
COLOR_CAPA = {
    "Fuente": "#F7F8F8",
    "Bronze": "#FFE2C4",
    "Silver": "#E3E7EC",
    "Gold": "#FFF3B0",
    "Uso": "#D6ECFF",
}


@dataclass(frozen=True)
class Nodo:
    capa: str
    nombre: str  # para Bronze, Silver y Gold: el nombre de la tabla del catálogo


@dataclass(frozen=True)
class Arista:
    desde: str
    hacia: str
    paso: str  # paso del DAG (o "—" si no es del DAG)
    que_pasa: str
    codigo: str


@dataclass(frozen=True)
class Indicador:
    nombre: str
    tabla: str  # nodo Gold donde vive el indicador
    columna: str
    formula: str
    aristas: tuple


NODOS = {
    "csv_tx": Nodo("Fuente", "fraudTest.csv"),
    "csv_loans": Nodo("Fuente", "Loan_default.csv"),
    "bronze_tx": Nodo("Bronze", "transactions"),
    "bronze_loans": Nodo("Bronze", "loan_default"),
    "silver_tx": Nodo("Silver", "transactions"),
    "silver_loans": Nodo("Silver", "loan_default"),
    "perfil": Nodo("Gold", "perfil_cliente"),
    "simulado": Nodo("Gold", "perfil_financiero_simulado"),
    "asignacion": Nodo("Gold", "asignacion_prestamos"),
    "kpis": Nodo("Gold", "kpis_cliente"),
    "prob": Nodo("Gold", "prob_impago_cliente"),
    "sim_insumos": Nodo("Gold", "simulacion_insumos"),
    "sim_resumen": Nodo("Gold", "simulacion_resumen"),
    "uso": Nodo("Uso", "Dashboard ejecutivo, reporte PDF y asistente de IA"),
}

# ---------------------------------------------------------------------------
# Tramos que comparten varios indicadores
# ---------------------------------------------------------------------------

TRANSACCIONES = (
    Arista(
        "csv_tx",
        "bronze_tx",
        "ingest_source",
        "Se copia el CSV de Kaggle tal cual a Parquet, particionado por año, mes y "
        "día. Es el histórico auditable: no se modifica.",
        "src/bronze_ingest/ingest_fraud_transactions.py",
    ),
    Arista(
        "bronze_tx",
        "silver_tx",
        "clean_silver + validate_quality",
        "Quita duplicados (trans_num), montos de 0 o menos, fechas nulas y los datos "
        "personales. Después deben pasar 6 reglas de calidad; si una falla, el DAG "
        "se detiene y Gold no se construye.",
        "src/spark/silver_transactions.py · src/spark/validate_silver.py",
    ),
)

GASTO = TRANSACCIONES + (
    Arista(
        "silver_tx",
        "perfil",
        "build_gold_perfil",
        "Por cliente: gasto_promedio_mensual = gasto total ÷ meses activo (días entre "
        "su primera y su última compra ÷ 30.44).",
        "src/spark/gold_transactions.py",
    ),
    Arista(
        "perfil",
        "kpis",
        "calcular_kpis",
        "Se une con el resto de los datos del cliente por cc_num.",
        "src/spark/gold_kpis.py",
    ),
)

INGRESO_SIMULADO = (
    Arista(
        "silver_tx",
        "simulado",
        "simular_ingreso_credito",
        "Ingreso mensual simulado con una fórmula fija y documentada (Opción A), "
        "porque los datasets no comparten clientes.",
        "src/spark/simular_perfil_financiero.py",
    ),
    Arista(
        "simulado",
        "kpis",
        "calcular_kpis",
        "Se une por cc_num. Este ingreso solo se usa para ahorro y flujo.",
        "src/spark/gold_kpis.py",
    ),
)

PRESTAMOS = (
    Arista(
        "csv_loans",
        "bronze_loans",
        "ingest_loan_default",
        "Se copia el CSV de Kaggle tal cual a Parquet.",
        "src/bronze_ingest/ingest_loan_default.py",
    ),
    Arista(
        "bronze_loans",
        "silver_loans",
        "clean_silver_loans",
        "Nombres en minúsculas, tipos correctos y se descartan préstamos sin "
        "ingreso válido.",
        "src/spark/silver_loans.py",
    ),
)

ASIGNACION = (
    Arista(
        "silver_loans",
        "asignacion",
        "asignar_prestamos",
        "Llave sintética por parecido: el cliente en el percentil p de gasto recibe "
        "el préstamo en el percentil p de ingreso. Uno a uno y reproducible.",
        "src/spark/asignar_prestamos.py",
    ),
    Arista(
        "perfil",
        "asignacion",
        "asignar_prestamos",
        "Aporta el gasto mensual de cada cliente para ordenarlos.",
        "src/spark/asignar_prestamos.py",
    ),
)

USO = Arista(
    "",  # se completa con la tabla Gold de cada indicador
    "uso",
    "—",
    "Leen la columna tal cual: el dashboard solo agrega (medianas, conteos) y el "
    "nivel de riesgo se clasifica con los cortes de 10% y 30%.",
    "src/dashboard/datos_ejecutivo.py",
)


def _uso(tabla: str) -> Arista:
    return Arista(tabla, USO.hacia, USO.paso, USO.que_pasa, USO.codigo)


# ---------------------------------------------------------------------------
# Indicadores
# ---------------------------------------------------------------------------

INDICADORES = (
    Indicador(
        "Gasto mensual",
        "kpis",
        "gasto_promedio_mensual",
        "gasto total con tarjeta ÷ meses activo",
        GASTO + (_uso("kpis"),),
    ),
    Indicador(
        "Capacidad de ahorro",
        "kpis",
        "capacidad_ahorro",
        "(ingreso_mensual_simulado − gasto_promedio_mensual) ÷ ingreso_mensual_simulado",
        GASTO + INGRESO_SIMULADO + (_uso("kpis"),),
    ),
    Indicador(
        "Endeudamiento (DTI)",
        "kpis",
        "ratio_endeudamiento",
        "pago_mensual_prestamo ÷ ingreso_mensual_fuente (ingreso del préstamo ÷ 12)",
        GASTO[:3]
        + PRESTAMOS
        + ASIGNACION
        + (
            Arista(
                "asignacion",
                "kpis",
                "calcular_kpis",
                "Trae monto, tasa, plazo e ingreso del préstamo asignado y calcula el "
                "pago mensual con la fórmula de amortización.",
                "src/spark/gold_kpis.py",
            ),
            _uso("kpis"),
        ),
    ),
    Indicador(
        "Probabilidad de impago",
        "prob",
        "prob_impago",
        "regresión logística sobre las características del préstamo asignado",
        GASTO[:3]
        + PRESTAMOS
        + ASIGNACION
        + (
            Arista(
                "silver_loans",
                "prob",
                "modelo_prob_impago",
                "Entrena una regresión logística con los préstamos y su etiqueta real "
                "de impago (80% para entrenar, 20% para probar).",
                "src/ml/modelo_impago.py",
            ),
            Arista(
                "asignacion",
                "prob",
                "modelo_prob_impago",
                "Aplica el modelo al préstamo asignado a cada cliente.",
                "src/ml/modelo_impago.py",
            ),
            _uso("prob"),
        ),
    ),
    Indicador(
        "Riesgo de números rojos (simulador)",
        "sim_resumen",
        "prob_deficit",
        "% de los 1,000 futuros simulados en los que el ahorro acumulado baja de cero",
        GASTO
        + INGRESO_SIMULADO
        + (
            Arista(
                "kpis",
                "sim_insumos",
                "simulador_monte_carlo",
                "Toma ingreso y gasto de cada cliente.",
                "src/ml/simulador.py",
            ),
            Arista(
                "silver_tx",
                "sim_insumos",
                "simulador_monte_carlo",
                "Calcula cuánto varía el gasto de cada cliente mes a mes.",
                "src/ml/simulador.py",
            ),
            Arista(
                "sim_insumos",
                "sim_resumen",
                "simulador_monte_carlo",
                "1,000 futuros a 36 meses por cliente y escenario.",
                "src/ml/simulador.py",
            ),
            _uso("sim_resumen"),
        ),
    ),
)


def indicador(nombre: str) -> Indicador:
    return next(i for i in INDICADORES if i.nombre == nombre)


def nodos_de(ind: Indicador) -> list:
    """Los nodos que recorre el indicador, de la fuente al uso."""
    ids = {a.desde for a in ind.aristas} | {a.hacia for a in ind.aristas}
    return sorted(ids, key=lambda i: (CAPAS.index(NODOS[i].capa), i))


def _etiqueta(nodo_id: str, estado: dict) -> str:
    n = NODOS[nodo_id]
    nombre = n.nombre.replace(", reporte", ",\\nreporte")
    texto = f"{n.capa}\\n{nombre}"
    filas = (estado or {}).get(nodo_id)
    if filas is not None:
        texto += f"\\n{filas:,} filas"
    return texto


def grafo(ind: Indicador, estado: dict = None) -> str:
    """El linaje en lenguaje DOT (Graphviz), de izquierda a derecha.

    estado: {id del nodo: filas} para mostrar cuántas filas tiene cada tabla.
    """
    lineas = [
        "digraph linaje {",
        '  rankdir=LR; bgcolor="transparent"; nodesep=0.3; ranksep=0.45;',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", '
        'fontsize=13, color="#C9CED6", fontcolor="#060E46", margin="0.15,0.08"];',
        '  edge [color="#001391", arrowsize=0.7, fontname="Helvetica", fontsize=10, '
        'fontcolor="#5B6573"];',
    ]
    nodos = nodos_de(ind)
    for capa in CAPAS:
        ids = [i for i in nodos if NODOS[i].capa == capa]
        if not ids:
            continue
        lineas.append("  { rank=same; " + " ".join(f'"{i}";' for i in ids) + " }")
        for i in ids:
            extra = ""
            if i == ind.tabla:
                extra = ', penwidth=2.2, color="#001391"'
            lineas.append(
                f'  "{i}" [label="{_etiqueta(i, estado)}", '
                f'fillcolor="{COLOR_CAPA[capa]}"{extra}];'
            )
    vistas = set()
    for a in ind.aristas:
        if (a.desde, a.hacia) in vistas:
            continue
        vistas.add((a.desde, a.hacia))
        etiqueta = "" if a.paso == "—" else a.paso.replace(" + ", "\\n+ ")
        lineas.append(f'  "{a.desde}" -> "{a.hacia}" [label=" {etiqueta} "];')
    lineas.append("}")
    return "\n".join(lineas)


def _profundidad(ind: Indicador) -> dict:
    """Cuántos pasos hay desde la fuente hasta cada nodo (el camino más largo),
    para listar las transformaciones en el orden en que ocurren."""
    previos = {}
    for a in ind.aristas:
        previos.setdefault(a.hacia, set()).add(a.desde)
    prof = {}

    def calcular(n):
        if n not in prof:
            prof[n] = 1 + max((calcular(p) for p in previos.get(n, ())), default=-1)
        return prof[n]

    for n in nodos_de(ind):
        calcular(n)
    return prof


def tabla_pasos(ind: Indicador, estado: dict = None) -> pd.DataFrame:
    """Una fila por transformación, en el orden en que ocurren, para auditarla."""
    prof = _profundidad(ind)
    filas = []
    for a in sorted(
        ind.aristas,
        key=lambda a: (prof[a.hacia], prof[a.desde], a.desde),
    ):
        destino = NODOS[a.hacia]
        filas.append(
            {
                "De": f"{NODOS[a.desde].capa} · {NODOS[a.desde].nombre}",
                "A": f"{destino.capa} · {destino.nombre}",
                "Paso del DAG": a.paso,
                "Qué pasa": a.que_pasa,
                "Código": a.codigo,
                "Filas en destino": (estado or {}).get(a.hacia),
            }
        )
    return pd.DataFrame(filas)


def estado_desde_catalogo(resumen: pd.DataFrame) -> dict:
    """{id del nodo: filas} a partir del resumen del catálogo."""
    filas = {(r.capa, r.tabla): r.filas for r in resumen.itertuples()}
    res = {}
    for i, n in NODOS.items():
        valor = filas.get((n.capa, n.nombre))
        if valor is not None and pd.notna(valor):
            res[i] = int(valor)
    return res
