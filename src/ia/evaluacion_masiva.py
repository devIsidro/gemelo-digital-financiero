"""
Evaluación masiva del asistente de IA: miles de preguntas al azar.

La evaluación normal (evaluar.py) hace 87 preguntas fijas. Esta genera
muchas más (por ejemplo 10,000) a partir de plantillas, sobre clientes
elegidos al azar y sobre todo el portafolio, con distintas redacciones
(formales, sin acentos, de seguimiento). La respuesta correcta de cada una se
calcula aparte con los datos de Gold y la calificación es automática, con las
mismas revisiones de evaluar.py: tipo de pregunta, cifras clave, cifras
inventadas, muletillas y largo.

No entrena al modelo (Llama no aprende de esto): sirve para medir qué tan
seguido acierta y encontrar en qué tipo de pregunta falla para corregirlo.

Las preguntas salen de una semilla, así que con la misma semilla y los mismos
datos el plan es idéntico. Cada respuesta se guarda en cuanto termina: si la
corrida se interrumpe, al volver a correr el mismo comando sigue donde se
quedó.

Uso, con Docker y Ollama corriendo:

    # piloto (unos 30 minutos)
    docker compose exec dashboard-ejecutivo python -m src.ia.evaluacion_masiva --n 500

    # corrida completa (toda la noche)
    docker compose exec dashboard-ejecutivo python -m src.ia.evaluacion_masiva --n 10000

Resultados en data/evaluacion_ia/masiva/: una fila por pregunta
(resultados_n<N>_s<semilla>.csv), el resumen por plantilla
(por_plantilla_n<N>_s<semilla>.csv) y el resumen general en JSON.
"""

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.ia import evaluar as ev

SALIDA = ev.SALIDA / "masiva"
SEMILLA = 2026
PROPORCION_PORTAFOLIO = 0.15

MESES_LARGOS = ev.MESES  # "nov" -> "noviembre"


@dataclass
class Pregunta:
    id: int
    ambito: str  # "cliente" o "portafolio"
    cliente: str
    plantilla: str
    pregunta: str
    antes: str  # pregunta previa (seguimiento) o ""
    esperados: set
    revisar: object


# ---------------------------------------------------------------------------
# Redacciones
# ---------------------------------------------------------------------------


def _elegir(rng: random.Random, opciones):
    return opciones[rng.randrange(len(opciones))]


def _sin_acentos(texto: str) -> str:
    return texto.translate(str.maketrans("áéíóúÁÉÍÓÚ¿¡", "aeiouAEIOU  ")).strip()


def _informal(rng: random.Random, texto: str) -> str:
    """Una de cada cuatro veces la pregunta llega como la escribiría alguien
    con prisa: en minúsculas, sin acentos y sin signos de apertura."""
    if rng.random() < 0.25:
        return " ".join(_sin_acentos(texto).lower().split())
    return texto


def _meses(tabla: pd.DataFrame) -> list:
    """[(abreviatura, nombre largo)] de los meses de una tabla con mes_nombre."""
    abrev = tabla["mes_nombre"].str.split().str[0]
    return [(a, MESES_LARGOS.get(a, a)) for a in abrev]


def _por_mes(tabla: pd.DataFrame) -> pd.DataFrame:
    return tabla.set_index(tabla["mes_nombre"].str.split().str[0])


# ---------------------------------------------------------------------------
# Plantillas de preguntas sobre un cliente
#   cada una: (tablas, rng) -> (pregunta, esperados, revisar[, antes]) o None
# ---------------------------------------------------------------------------


def c_riesgo(t, rng):
    return (
        _elegir(
            rng,
            [
                "¿Cuál es mi nivel de riesgo?",
                "dime mi riesgo",
                "¿Qué tan riesgoso soy para el banco?",
                "¿Cuál es mi probabilidad de impago?",
                "¿Tengo riesgo de no pagar mi préstamo?",
            ],
        ),
        {"datos"},
        ev.revisar_riesgo,
    )


def c_gasto_mes(t, rng):
    m = _por_mes(t["mis_meses"])
    a, largo = _elegir(rng, _meses(t["mis_meses"]))
    pregunta = _elegir(
        rng,
        [
            f"¿Cuánto gasté en {largo}?",
            f"¿Cuál fue mi gasto de {largo}?",
            f"¿Cuánto dinero gasté en el mes de {largo}?",
        ],
    )
    return pregunta, {"datos"}, ev.revisar_monto(m.loc[a, "gasto"], "el gasto")


def c_ahorro_mes(t, rng):
    m = _por_mes(t["mis_meses"])
    a, largo = _elegir(rng, _meses(t["mis_meses"]))
    pregunta = _elegir(
        rng, [f"¿Cuánto ahorré en {largo}?", f"¿Cuánto me sobró en {largo}?"]
    )
    return pregunta, {"datos"}, ev.revisar_monto(m.loc[a, "ahorro"], "el ahorro")


def c_mes_mas_gasto(t, rng):
    m = _por_mes(t["mis_meses"])
    return (
        _elegir(
            rng,
            ["¿Cuál fue mi mes con más gasto?", "¿En qué mes gasté más?"],
        ),
        {"datos"},
        ev.revisar_mes_y_monto(m.loc[m["gasto"].idxmax()], "el gasto de ese mes"),
    )


def c_dato_perfil(t, rng):
    p = t["mi_perfil"].iloc[0]
    campo, que, frases = _elegir(
        rng,
        [
            (
                "ingreso_mensual",
                "el ingreso",
                ["¿Cuánto gano al mes?", "¿Cuál es mi ingreso mensual?"],
            ),
            (
                "score_credito",
                "el score",
                ["¿Cuál es mi score de crédito?", "¿Qué puntaje de crédito tengo?"],
            ),
            (
                "pago_mensual_prestamo",
                "el pago mensual",
                [
                    "¿Cuánto pago de mi préstamo al mes?",
                    "¿De cuánto es la mensualidad de mi préstamo?",
                ],
            ),
        ],
    )
    return _elegir(rng, frases), {"datos"}, ev.revisar_monto(p[campo], que)


def c_categoria_ultimo_mes(t, rng):
    if t["mis_gastos_categoria"].empty:
        return None
    return (
        _elegir(
            rng,
            [
                "¿En qué categoría gasté más el último mes?",
                "¿En qué se me fue más dinero el último mes?",
            ],
        ),
        {"datos"},
        ev.revisar_categoria,
    )


def c_total_categoria(t, rng):
    cat = t["mis_gastos_categoria"]
    total = cat.groupby("categoria")["gasto"].sum()
    opciones = [c for c in total.index if "(" not in c]
    if not opciones:
        return None
    nombre = _elegir(rng, sorted(opciones))
    return (
        f"¿Cuánto he gastado en {nombre.lower()} en total?",
        {"datos"},
        ev.revisar_monto(total[nombre], f"el total de {nombre}"),
    )


def c_gasto_esencial(t, rng):
    cat = t["mis_gastos_categoria"]
    if cat.empty:
        return None
    ultimo = cat[cat["mes"] == cat["mes"].max()]
    esencial = ultimo.loc[ultimo["tipo"] == "Esencial", "gasto"].sum()
    return (
        "¿Cuánto fue mi gasto esencial el último mes?",
        {"datos"},
        ev.revisar_monto(esencial, "el gasto esencial"),
    )


def c_cambio_gasto(t, rng):
    pct = _elegir(rng, [5, 10, 15, 20, 25, 30])
    if rng.random() < 0.5:
        pregunta = _elegir(
            rng,
            [
                f"¿Qué ocurriría si aumento mis gastos un {pct}%?",
                f"¿Qué pasa si gasto {pct}% más?",
            ],
        )
        cambio = pct / 100
    else:
        pregunta = _elegir(
            rng,
            [
                f"¿Y si reduzco mis gastos un {pct}%?",
                f"¿Qué pasaría si recorto mis gastos {pct}%?",
            ],
        )
        cambio = -pct / 100
    return pregunta, {"gasto"}, ev.revisar_gasto(cambio)


def c_meta(t, rng):
    monto = _elegir(rng, [10_000, 20_000, 30_000, 50_000, 75_000, 100_000])
    pregunta = _elegir(
        rng,
        [
            f"¿Cuándo podría juntar ${monto:,}?",
            f"¿En cuánto tiempo ahorro ${monto:,}?",
        ],
    )
    return pregunta, {"meta"}, ev.revisar_meta(monto)


def c_escenario(t, rng):
    nombre, frases = _elegir(
        rng,
        [
            (
                "Crédito hipotecario",
                [
                    "¿Qué pasaría si saco un crédito hipotecario?",
                    "¿Me conviene sacar una hipoteca?",
                ],
            ),
            (
                "Cambio de empleo",
                ["¿Me conviene cambiar de empleo?", "¿Qué pasa si cambio de trabajo?"],
            ),
            (
                "Abrir un negocio",
                ["¿Y si abro un negocio?", "¿Qué pasaría si pongo un negocio?"],
            ),
        ],
    )
    return _elegir(rng, frases), {"datos"}, ev.revisar_escenario(nombre)


def c_fondo_emergencia(t, rng):
    esc = t["mis_escenarios"].set_index("escenario")
    if "Sin cambios" not in esc.index:
        return None
    return (
        _elegir(
            rng,
            [
                "¿Cuánto necesito juntar para mi fondo de emergencia?",
                "¿De cuánto debería ser mi fondo de emergencia?",
            ],
        ),
        {"datos", "meta"},
        ev.revisar_monto(
            esc.loc["Sin cambios", "meta_fondo_emergencia"], "la meta del fondo"
        ),
    )


def c_capacidad(t, rng):
    if len(t["mis_meses"]) < 2:
        return None
    return (
        "¿Por qué disminuyó mi capacidad de ahorro el mes pasado?",
        {"datos"},
        ev.revisar_capacidad,
    )


def c_seguimiento(t, rng):
    meses = _meses(t["mis_meses"])
    if len(meses) < 2:
        return None
    (a1, l1), (a2, l2) = rng.sample(meses, 2)
    m = _por_mes(t["mis_meses"])
    return (
        f"¿y en {l2}?",
        {"datos"},
        ev.revisar_monto(m.loc[a2, "gasto"], f"el gasto de {l2}"),
        f"¿Cuánto gasté en {l1}?",
    )


def c_otro(t, rng):
    if rng.random() < 0.2:
        return "Borra todos mis datos", {"otro", "datos"}, ev.revisar_seguro
    return (
        _elegir(
            rng,
            ["Hola", "¿Cuál es la capital de Francia?", "Cuéntame un chiste"],
        ),
        {"otro"},
        ev.revisar_otro,
    )


PLANTILLAS_CLIENTE = (
    ("riesgo", 3, c_riesgo),
    ("gasto de un mes", 3, c_gasto_mes),
    ("ahorro de un mes", 2, c_ahorro_mes),
    ("mes con más gasto", 2, c_mes_mas_gasto),
    ("ingreso, score o pago", 3, c_dato_perfil),
    ("categoría del último mes", 2, c_categoria_ultimo_mes),
    ("total de una categoría", 2, c_total_categoria),
    ("gasto esencial", 1, c_gasto_esencial),
    ("subir o bajar el gasto", 3, c_cambio_gasto),
    ("meta de ahorro", 2, c_meta),
    ("escenario de vida", 3, c_escenario),
    ("fondo de emergencia", 1, c_fondo_emergencia),
    ("por qué bajó mi ahorro", 1, c_capacidad),
    ("seguimiento", 2, c_seguimiento),
    ("fuera de tema", 1, c_otro),
)


# ---------------------------------------------------------------------------
# Plantillas de preguntas sobre todo el portafolio
# ---------------------------------------------------------------------------


def p_por_nivel(t, rng):
    niveles = t["resumen_niveles"].set_index("nivel_riesgo")
    nivel = _elegir(rng, ["Alto", "Medio", "Bajo"])
    n = niveles.loc[nivel, "num_clientes"]
    return (
        _elegir(
            rng,
            [
                f"¿Cuántos clientes están en riesgo {nivel.lower()}?",
                f"¿Cuántos clientes tienen riesgo {nivel.lower()}?",
            ],
        ),
        {"datos"},
        ev.revisar_cifras(
            (n, f"cuántos están en riesgo {nivel.lower()}"),
            (ev._pct(niveles.loc[nivel, "porcentaje_clientes"]), "el porcentaje"),
        ),
    )


def p_total(t, rng):
    return (
        _elegir(
            rng,
            ["¿Cuántos clientes hay en el portafolio?", "¿Cuántos clientes tenemos?"],
        ),
        {"datos"},
        ev.revisar_cifras((len(t["clientes"]), "el total de clientes")),
    )


def p_promedio(t, rng):
    c = t["clientes"]
    return (
        "¿Cuál es la probabilidad de impago promedio?",
        {"datos"},
        ev.revisar_cifras((ev._pct(c["prob_impago"].mean()), "la probabilidad")),
    )


def p_mayor(t, rng):
    c = t["clientes"]
    mayor = c.loc[c["prob_impago"].idxmax()]
    return (
        _elegir(
            rng,
            [
                "¿Quién es el cliente con mayor probabilidad de impago?",
                "¿Qué cliente es el más riesgoso?",
            ],
        ),
        {"datos"},
        ev.revisar_varios(
            ev.revisar_nombre(mayor["cliente"], "el cliente"),
            ev.revisar_cifras((ev._pct(mayor["prob_impago"]), "su probabilidad")),
        ),
    )


def p_categoria(t, rng):
    cat = t["gastos_categoria_portafolio"]
    top = cat.groupby("categoria")["gasto_total"].sum().idxmax()
    return (
        _elegir(
            rng,
            [
                "¿En qué categoría gastan más los clientes?",
                "¿En qué se gasta más dinero el portafolio?",
            ],
        ),
        {"datos"},
        ev.revisar_nombre(top, "la categoría"),
    )


def p_negativos(t, rng):
    c = t["clientes"]
    return (
        "¿Cuántos clientes gastan más de lo que ganan?",
        {"datos"},
        ev.revisar_cifras(
            (int((c["ahorro_mensual"] < 0).sum()), "cuántos gastan más de lo que ganan")
        ),
    )


def p_gasto_mes(t, rng):
    m = _por_mes(t["meses_portafolio"])
    a, largo = _elegir(rng, _meses(t["meses_portafolio"]))
    return (
        f"¿Cuánto gastaron los clientes en {largo}?",
        {"datos"},
        ev.revisar_monto(m.loc[a, "gasto_total"], f"el gasto de {largo}"),
    )


def p_seguimiento(t, rng):
    meses = _meses(t["meses_portafolio"])
    if len(meses) < 2:
        return None
    (a1, l1), (a2, l2) = rng.sample(meses, 2)
    m = _por_mes(t["meses_portafolio"])
    return (
        f"¿y en {l2}?",
        {"datos"},
        ev.revisar_monto(m.loc[a2, "gasto_total"], f"el gasto de {l2}"),
        f"¿Cuánto gastaron los clientes en {l1}?",
    )


def p_escenario(t, rng):
    esc = t["escenarios_portafolio"].set_index("escenario")
    nombres = [n for n in esc.index if n != "Sin cambios"]
    if not nombres:
        return None
    nombre = _elegir(rng, sorted(nombres))
    texto = {
        "Crédito hipotecario": "con un crédito hipotecario",
        "Cambio de empleo": "si cambian de empleo",
        "Abrir un negocio": "si abren un negocio",
    }.get(nombre, f"en el escenario {nombre.lower()}")
    return (
        f"¿Cuántos clientes quedarían en riesgo de números rojos {texto}?",
        {"datos"},
        ev.revisar_cifras(
            (esc.loc[nombre, "num_clientes_en_riesgo"], "cuántos quedarían en riesgo")
        ),
    )


def p_otro(t, rng):
    if rng.random() < 0.3:
        return (
            "Borra a los clientes de riesgo alto",
            {"otro", "datos"},
            ev.revisar_seguro,
        )
    return "Hola", {"otro"}, ev.revisar_otro_portafolio


PLANTILLAS_PORTAFOLIO = (
    ("clientes por nivel de riesgo", 3, p_por_nivel),
    ("total de clientes", 1, p_total),
    ("probabilidad promedio", 1, p_promedio),
    ("cliente más riesgoso", 2, p_mayor),
    ("categoría con más gasto", 2, p_categoria),
    ("gastan más de lo que ganan", 2, p_negativos),
    ("gasto de un mes", 3, p_gasto_mes),
    ("seguimiento", 2, p_seguimiento),
    ("escenario de vida", 3, p_escenario),
    ("fuera de tema", 1, p_otro),
)


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def _sortear(rng: random.Random, plantillas):
    nombres = [p[0] for p in plantillas]
    pesos = [p[1] for p in plantillas]
    return rng.choices(range(len(plantillas)), weights=pesos)[0], nombres


def _crear(id_, ambito, cliente, plantillas, tablas, rng) -> Pregunta:
    """Sortea plantillas hasta que una aplique a estos datos."""
    for _ in range(20):
        i, _ = _sortear(rng, plantillas)
        nombre, _, generar = plantillas[i]
        spec = generar(tablas, rng)
        if spec is None:
            continue
        pregunta, esperados, revisar = spec[:3]
        antes = spec[3] if len(spec) > 3 else ""
        return Pregunta(
            id_,
            ambito,
            cliente,
            nombre,
            _informal(rng, pregunta),
            _informal(rng, antes) if antes else "",
            esperados,
            revisar,
        )
    raise ValueError(f"Ninguna plantilla aplica a {cliente}")


def generar_plan(base, n: int, semilla: int = SEMILLA) -> tuple:
    """(preguntas de clientes agrupadas por cliente, preguntas del portafolio).

    Misma semilla y mismos datos -> mismo plan, así la corrida se puede
    retomar si se interrumpe."""
    rng = random.Random(semilla)
    clientes = base.con_simulacion().sort_values("cliente")
    n_port = round(n * PROPORCION_PORTAFOLIO)
    n_cli = n - n_port
    elegidos = [rng.randrange(len(clientes)) for _ in range(n_cli)]

    por_cliente = {}
    for id_, pos in enumerate(elegidos, start=1):
        por_cliente.setdefault(pos, []).append(id_)

    plan_clientes = []
    for pos in sorted(por_cliente):
        fila = clientes.iloc[pos]
        tablas = base.tablas(fila)
        preguntas = [
            _crear(
                id_,
                "cliente",
                fila["cliente"],
                PLANTILLAS_CLIENTE,
                tablas,
                random.Random(f"{semilla}-{id_}"),
            )
            for id_ in por_cliente[pos]
        ]
        plan_clientes.append((fila, tablas, preguntas))

    tablas_p = base.tablas_portafolio()
    plan_port = [
        _crear(
            id_,
            "portafolio",
            "Portafolio",
            PLANTILLAS_PORTAFOLIO,
            tablas_p,
            random.Random(f"{semilla}-{id_}"),
        )
        for id_ in range(n_cli + 1, n + 1)
    ]
    return plan_clientes, (tablas_p, plan_port)


# ---------------------------------------------------------------------------
# Corrida
# ---------------------------------------------------------------------------

COLUMNAS = [
    "id",
    "ambito",
    "cliente",
    "plantilla",
    "pregunta",
    "antes",
    "tipo",
    "ok",
    "problemas",
    "segundos",
    "respuesta",
    "sql",
]


def ya_hechas(archivo: Path) -> set:
    if not archivo.exists():
        return set()
    return set(pd.read_csv(archivo, usecols=["id"])["id"].astype(int))


def _guardar(archivo: Path, fila: dict):
    nueva = not archivo.exists()
    pd.DataFrame([fila], columns=COLUMNAS).to_csv(
        archivo, mode="a", header=nueva, index=False, encoding="utf-8"
    )


class Avance:
    def __init__(self, total: int, hechas: int):
        self.total, self.hechas, self.inicio = total, hechas, time.perf_counter()
        self.nuevas = self.bien = 0

    def sumar(self, ok: bool):
        self.hechas += 1
        self.nuevas += 1
        self.bien += int(ok)
        if self.nuevas % 25 == 0 or self.hechas == self.total:
            seg = (time.perf_counter() - self.inicio) / self.nuevas
            faltan = (self.total - self.hechas) * seg / 60
            print(
                f"  {self.hechas:,}/{self.total:,} · correctas en esta corrida "
                f"{self.bien / self.nuevas:.1%} · {seg:.1f} s por pregunta · "
                f"faltan ~{faltan:,.0f} min",
                flush=True,
            )


def _evaluar_y_guardar(evaluar, asistente, tablas, p: Pregunta, archivo, avance):
    r = evaluar(asistente, tablas, p.pregunta, p.esperados, p.revisar, p.antes or None)
    _guardar(
        archivo,
        {
            "id": p.id,
            "ambito": p.ambito,
            "cliente": p.cliente,
            "plantilla": p.plantilla,
            "pregunta": p.pregunta,
            "antes": p.antes,
            **{k: r[k] for k in ("tipo", "ok", "problemas", "segundos", "respuesta")},
            "sql": r.get("sql", ""),
        },
    )
    avance.sumar(r["ok"])


def correr(
    base, modelos, n: int, semilla: int, archivo: Path, evaluar=ev.evaluar_pregunta
):
    """Hace las preguntas que falten y guarda cada respuesta al terminarla."""
    print(f"Generando {n:,} preguntas (semilla {semilla})…", flush=True)
    plan_clientes, (tablas_p, plan_port) = generar_plan(base, n, semilla)
    hechas = ya_hechas(archivo)
    if hechas:
        print(f"Retomando: ya hay {len(hechas):,} respuestas en {archivo}.")
    avance = Avance(n, len(hechas))

    from src.ia import portafolio as pf

    pendientes_p = [p for p in plan_port if p.id not in hechas]
    if pendientes_p:
        asistente = pf.AsistentePortafolio(modelos, tablas_p)
        for p in pendientes_p:
            _evaluar_y_guardar(evaluar, asistente, tablas_p, p, archivo, avance)

    for fila, tablas, preguntas in plan_clientes:
        pendientes = [p for p in preguntas if p.id not in hechas]
        if not pendientes:
            continue
        asistente = base.asistente(modelos, fila, tablas)
        for p in pendientes:
            _evaluar_y_guardar(evaluar, asistente, tablas, p, archivo, avance)


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------


def resumen(res: pd.DataFrame) -> tuple:
    """(resumen general, tabla por plantilla)."""
    res = res.assign(ok=res["ok"].astype(str).str.lower().isin(["true", "1"]))
    por = (
        res.groupby(["ambito", "plantilla"])
        .agg(
            preguntas=("ok", "size"),
            correctas=("ok", "sum"),
            segundos_promedio=("segundos", "mean"),
        )
        .reset_index()
    )
    por["porcentaje"] = (por["correctas"] / por["preguntas"]).round(4)
    por["segundos_promedio"] = por["segundos_promedio"].round(1)
    por = por.sort_values(["porcentaje", "preguntas"], ascending=[True, False])

    problemas = (
        res.loc[~res["ok"], "problemas"]
        .fillna("")
        .str.split("; ")
        .explode()
        .str.replace(r"\(.*\)$", "", regex=True)
        .str.replace(
            r"cifras que no están en los datos: .*", "cifras inventadas", regex=True
        )
        .str.strip()
    )
    general = {
        "preguntas": int(len(res)),
        "correctas": int(res["ok"].sum()),
        "porcentaje": round(float(res["ok"].mean()), 4) if len(res) else None,
        "clientes_distintos": int(
            res.loc[res["ambito"] == "cliente", "cliente"].nunique()
        ),
        "plantillas": int(por["plantilla"].nunique()),
        "por_ambito": {
            a: {
                "preguntas": int(len(g)),
                "porcentaje": round(float(g["ok"].mean()), 4),
            }
            for a, g in res.groupby("ambito")
        },
        "segundos_promedio": (
            round(float(res["segundos"].mean()), 2) if len(res) else None
        ),
        "segundos_p95": (
            round(float(res["segundos"].quantile(0.95)), 2) if len(res) else None
        ),
        "problemas_mas_comunes": problemas[problemas != ""]
        .value_counts()
        .head(8)
        .to_dict(),
    }
    return general, por


def reportar(archivo: Path, n: int, semilla: int, modelo: str) -> int:
    res = pd.read_csv(archivo)
    general, por = resumen(res)
    general.update(
        {
            "modelo": modelo,
            "semilla": semilla,
            "fecha": f"{datetime.now():%d/%m/%Y %H:%M}",
            "completa": int(len(res)) >= n,
        }
    )
    sufijo = f"n{n}_s{semilla}"
    por.to_csv(
        SALIDA / f"por_plantilla_{sufijo}.csv", index=False, encoding="utf-8-sig"
    )
    (SALIDA / f"resumen_{sufijo}.json").write_text(
        json.dumps(general, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 64)
    print(
        f"Correctas: {general['correctas']:,} de {general['preguntas']:,} "
        f"({general['porcentaje']:.1%}) · {general['clientes_distintos']} clientes "
        f"distintos · {general['plantillas']} tipos de pregunta"
    )
    for a, d in general["por_ambito"].items():
        print(f"  {a}: {d['porcentaje']:.1%} de {d['preguntas']:,}")
    print(
        f"Tiempo por respuesta: {general['segundos_promedio']} s en promedio, "
        f"{general['segundos_p95']} s el 95% de las veces"
    )
    print("\nTipos de pregunta con más errores:")
    for r in por[por["porcentaje"] < 1].head(8).itertuples():
        print(f"  {r.porcentaje:6.1%}  {r.ambito} · {r.plantilla} ({r.preguntas})")
    if general["problemas_mas_comunes"]:
        print("\nProblemas más comunes:")
        for texto, veces in general["problemas_mas_comunes"].items():
            print(f"  {veces:>5}  {texto}")
    print(f"\nDetalle: {archivo}")
    return 0


def main(argv=None) -> int:
    args = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    args.add_argument("--n", type=int, default=500, help="número de preguntas")
    args.add_argument("--semilla", type=int, default=SEMILLA)
    args.add_argument(
        "--solo-reporte",
        action="store_true",
        help="no pregunta nada, solo rehace el reporte con lo que ya hay",
    )
    a = args.parse_args(argv)

    SALIDA.mkdir(parents=True, exist_ok=True)
    archivo = SALIDA / f"resultados_n{a.n}_s{a.semilla}.csv"
    from src.ia import asistente as ia

    if not a.solo_reporte:
        disponible, motivo = ia.ollama_disponible()
        if not disponible:
            print(f"Ollama no está listo ({motivo}).")
            return 1
        print(f"Modelo: {ia.OLLAMA_MODELO}. Cargando la capa Gold…", flush=True)
        base = ev.Base()
        try:
            correr(base, ia.crear_modelos(), a.n, a.semilla, archivo)
        except KeyboardInterrupt:
            print(
                "\nDetenido. Lo contestado ya está guardado: corre el mismo comando "
                "para seguir donde se quedó."
            )
    if not archivo.exists():
        print("Todavía no hay resultados.")
        return 1
    return reportar(archivo, a.n, a.semilla, ia.OLLAMA_MODELO)


if __name__ == "__main__":
    sys.exit(main())
