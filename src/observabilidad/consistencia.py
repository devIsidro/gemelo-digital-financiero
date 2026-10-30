"""
Consistencia de los datos entre capas (Fase 6).

La demo final pide «demostrar la consistencia de los datos en todas las capas
del Lakehouse». Aquí se revisa, con los mismos archivos que usa el pipeline,
que nada se pierda ni se invente al pasar de una capa a otra:

  Bronze -> Silver   ninguna fila nueva en Silver; lo descartado está acotado.
  Silver -> Gold     los mismos clientes y el mismo gasto, al centavo.
  Gold               cada cliente con su préstamo, su probabilidad de impago y
                     sus escenarios del simulador.

Cada revisión regresa un Chequeo con ok=True/False, o ok=None si la tabla
todavía no existe (por ejemplo, antes de la primera corrida del DAG).
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Máximo de filas que Silver puede descartar de Bronze (reglas de negocio y
# duplicados) antes de considerarlo un problema.
MAX_DESCARTE = 0.01
TOLERANCIA_USD = 0.01


@dataclass
class Chequeo:
    nombre: str
    ok: object  # True, False o None (sin datos)
    detalle: str

    def __post_init__(self):
        if self.ok is not None:
            self.ok = bool(self.ok)  # numpy.bool_ -> bool


def _leer(ruta: Path, columnas: list):
    ruta = Path(ruta)
    if not ruta.exists():
        return None
    return pd.read_parquet(ruta, columns=columnas)


def _sin_datos(nombre: str, que: str) -> Chequeo:
    return Chequeo(nombre, None, f"Todavía no existe {que}.")


def _bronze_a_silver(nombre, bronze, silver, llave_b, llave_s) -> Chequeo:
    if bronze is None or silver is None:
        return _sin_datos(nombre, "Bronze o Silver")
    ids_b = set(bronze[llave_b])
    nuevas = int((~silver[llave_s].isin(ids_b)).sum())
    duplicadas_silver = int(silver[llave_s].duplicated().sum())
    descartadas = len(bronze) - len(silver)
    pct = descartadas / len(bronze) if len(bronze) else 0.0
    ok = nuevas == 0 and duplicadas_silver == 0 and 0 <= pct <= MAX_DESCARTE
    detalle = (
        f"{len(bronze):,} en Bronze → {len(silver):,} en Silver "
        f"({descartadas:,} descartadas por reglas de negocio o duplicados)."
    )
    if nuevas:
        detalle += f" {nuevas:,} filas de Silver no existen en Bronze."
    if duplicadas_silver:
        detalle += f" {duplicadas_silver:,} llaves repetidas en Silver."
    if pct > MAX_DESCARTE:
        detalle += f" Se descartó el {pct:.1%} (máximo {MAX_DESCARTE:.0%})."
    return Chequeo(nombre, ok, detalle)


def transacciones_bronze_silver(rutas) -> Chequeo:
    return _bronze_a_silver(
        "Transacciones: Bronze → Silver",
        _leer(rutas.bronze_tx, ["trans_num"]),
        _leer(rutas.silver_tx, ["trans_num"]),
        "trans_num",
        "trans_num",
    )


def prestamos_bronze_silver(rutas) -> Chequeo:
    return _bronze_a_silver(
        "Préstamos: Bronze → Silver",
        _leer(rutas.bronze_loans, ["LoanID"]),
        _leer(rutas.silver_loans, ["loan_id"]),
        "LoanID",
        "loan_id",
    )


def clientes_silver_gold(rutas) -> Chequeo:
    nombre = "Clientes: Silver → Gold"
    silver = _leer(rutas.silver_tx, ["cc_num"])
    tablas = {
        "perfil": _leer(rutas.gold_perfil, ["cc_num"]),
        "KPIs": _leer(rutas.gold_kpis, ["cc_num"]),
        "probabilidad de impago": _leer(rutas.gold_prob, ["cc_num"]),
    }
    if silver is None or any(t is None for t in tablas.values()):
        return _sin_datos(nombre, "Silver o alguna tabla de Gold")
    esperados = set(silver["cc_num"])
    problemas = []
    for que, t in tablas.items():
        ids = set(t["cc_num"])
        if ids != esperados or t["cc_num"].duplicated().any():
            faltan = len(esperados - ids)
            sobran = len(ids - esperados)
            problemas.append(f"{que}: faltan {faltan:,}, sobran {sobran:,}")
    detalle = (
        f"Los {len(esperados):,} clientes de Silver están una vez en el perfil, "
        "en los KPIs y en la probabilidad de impago."
        if not problemas
        else "No cuadran los clientes. " + "; ".join(problemas) + "."
    )
    return Chequeo(nombre, not problemas, detalle)


def gasto_silver_gold(rutas) -> Chequeo:
    nombre = "Gasto: Silver → Gold"
    silver = _leer(rutas.silver_tx, ["cc_num", "amt"])
    gold = _leer(rutas.gold_perfil, ["cc_num", "gasto_total", "num_transacciones"])
    if silver is None or gold is None:
        return _sin_datos(nombre, "Silver o el perfil de Gold")
    s = silver.groupby("cc_num")["amt"].agg(["sum", "size"]).reset_index()
    t = s.merge(gold, on="cc_num", how="outer")
    mal_gasto = int(((t["sum"] - t["gasto_total"]).abs() > TOLERANCIA_USD).sum())
    mal_compras = int((t["size"] != t["num_transacciones"]).sum())
    total_s, total_g = float(silver["amt"].sum()), float(gold["gasto_total"].sum())
    ok = (
        mal_gasto == 0
        and mal_compras == 0
        and t[["sum", "gasto_total"]].notna().all().all()
    )
    detalle = (
        f"Gasto total: ${total_s:,.2f} en Silver y ${total_g:,.2f} en Gold; "
        f"{len(silver):,} compras en los dos."
    )
    if not ok:
        detalle += (
            f" {mal_gasto:,} clientes con gasto distinto y {mal_compras:,} con "
            "distinto número de compras."
        )
    return Chequeo(nombre, ok, detalle)


def prestamos_gold(rutas) -> Chequeo:
    nombre = "Préstamo asignado a cada cliente"
    kpis = _leer(rutas.gold_kpis, ["cc_num", "loan_id"])
    loans = _leer(rutas.silver_loans, ["loan_id"])
    if kpis is None or loans is None:
        return _sin_datos(nombre, "los KPIs de Gold o el Silver de préstamos")
    sin_prestamo = int(kpis["loan_id"].isna().sum())
    repetidos = int(kpis["loan_id"].dropna().duplicated().sum())
    inexistentes = int((~kpis["loan_id"].dropna().isin(set(loans["loan_id"]))).sum())
    ok = sin_prestamo == repetidos == inexistentes == 0
    detalle = (
        f"Los {len(kpis):,} clientes tienen un préstamo distinto que existe en "
        "Silver (llave sintética)."
        if ok
        else f"{sin_prestamo:,} sin préstamo, {repetidos:,} préstamos repetidos y "
        f"{inexistentes:,} que no existen en Silver."
    )
    return Chequeo(nombre, ok, detalle)


def probabilidad_gold(rutas) -> Chequeo:
    nombre = "Probabilidad de impago"
    prob = _leer(rutas.gold_prob, ["cc_num", "prob_impago"])
    if prob is None:
        return _sin_datos(nombre, "la probabilidad de impago")
    p = prob["prob_impago"]
    fuera = int((p.isna() | (p < 0) | (p > 1)).sum())
    ok = fuera == 0
    detalle = (
        f"Las {len(prob):,} probabilidades están entre 0 y 1 "
        f"(de {p.min():.1%} a {p.max():.1%})."
        if ok
        else f"{fuera:,} probabilidades vacías o fuera de 0 a 1."
    )
    return Chequeo(nombre, ok, detalle)


def simulacion_gold(rutas) -> Chequeo:
    nombre = "Simulador Monte Carlo"
    kpis = _leer(rutas.gold_kpis, ["cc_num"])
    insumos = _leer(rutas.simulacion_insumos, ["cc_num"])
    resumen = _leer(rutas.simulacion_resumen, ["cc_num", "escenario"])
    if kpis is None or insumos is None or resumen is None:
        return _sin_datos(nombre, "la simulación")
    escenarios = resumen["escenario"].nunique()
    por_cliente = resumen.groupby("cc_num")["escenario"].nunique()
    clientes = set(kpis["cc_num"])
    faltan = len(clientes - set(insumos["cc_num"])) + len(
        clientes - set(por_cliente.index)
    )
    incompletos = int((por_cliente < escenarios).sum())
    ok = faltan == 0 and incompletos == 0
    detalle = (
        f"Los {len(clientes):,} clientes tienen sus {escenarios} escenarios simulados."
        if ok
        else f"{faltan:,} clientes sin simular y {incompletos:,} con escenarios "
        "incompletos."
    )
    return Chequeo(nombre, ok, detalle)


REVISIONES = (
    transacciones_bronze_silver,
    prestamos_bronze_silver,
    clientes_silver_gold,
    gasto_silver_gold,
    prestamos_gold,
    probabilidad_gold,
    simulacion_gold,
)


def revisar_consistencia(rutas) -> list:
    """Todas las revisiones. Si una truena (archivo dañado, columna que ya no
    existe), se reporta como falla en lugar de tumbar el panel o el DAG."""
    chequeos = []
    for revision in REVISIONES:
        try:
            chequeos.append(revision(rutas))
        except Exception as err:  # noqa: BLE001
            nombre = revision.__name__.replace("_", " ")
            chequeos.append(Chequeo(nombre, False, f"No se pudo revisar: {err}"))
    return chequeos
