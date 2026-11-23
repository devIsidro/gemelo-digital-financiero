"""
Evaluación automática del asistente de IA (Fase 5).

Hace preguntas reales a Llama 3 (Ollama) con varios clientes y revisa, sin
que nadie tenga que leer las respuestas, que cada una:

1. se entienda como el tipo correcto de pregunta (datos, gasto, meta u otro);
2. traiga las cifras clave correctas, calculadas aparte en Python;
3. no tenga cifras inventadas: cada número de la respuesta tiene que estar en
   lo que recibió el modelo o en la pregunta;
4. no tenga muletillas prohibidas («según los datos», «ATENCIÓN:», …) y sea
   corta (máximo 6 oraciones).

Uso, con Docker y Ollama corriendo:

    docker compose exec dashboard-ejecutivo python -m src.ia.evaluar
    docker compose exec dashboard-ejecutivo python -m src.ia.evaluar --libres
    docker compose exec dashboard-ejecutivo python -m src.ia.evaluar --portafolio

La segunda forma corre solo las preguntas libres (otra redacción, preguntas
de seguimiento y escenarios del simulador); la tercera, solo las preguntas
sobre todo el portafolio (src/ia/portafolio.py).

Los resultados completos (respuesta, SQL, problemas) quedan en
data/evaluacion_ia/resultados.csv (o resultados_portafolio.csv).
"""

import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.dashboard import datos
from src.dashboard import datos_ejecutivo as de
from src.ia import asistente as ia
from src.ia import contexto as cx
from src.ia import portafolio as pf

RUTAS = datos.Rutas(Path("data"))
SALIDA = Path("data/evaluacion_ia")

PROHIBIDAS = (
    "según los datos",
    "según la información",
    "hechos clave",
    "datos de apoyo",
    "atención:",
    "la pregunta",
)
MAX_ORACIONES = 6


# ---------------------------------------------------------------------------
# Números en un texto
# ---------------------------------------------------------------------------

_NUMERO = re.compile(r"(-?)\$?(\d[\d,]*(?:\.\d+)?)")


def numeros(texto: str) -> list:
    """Todas las cifras de un texto como float: «$84,477» -> 84477.0,
    «12.1%» -> 12.1, «-$87,341» -> -87341.0, «1,000» -> 1000.0."""
    valores = []
    for m in _NUMERO.finditer(texto):
        crudo = m.group(2).rstrip(",")
        try:
            v = float(crudo.replace(",", ""))
        except ValueError:
            continue
        valores.append(-v if m.group(1) else v)
    return valores


def _cerca(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.051, abs(b) * 0.0005)


def contiene(texto: str, valor: float) -> bool:
    return any(_cerca(n, valor) for n in numeros(texto))


def cifras_inventadas(respuesta: str, permitido: str) -> list:
    """Cifras de la respuesta que no aparecen en lo que recibió el modelo.
    Se ignoran 0, 1 y 2 («1 de cada…», «2 meses» salen en frases comunes)."""
    base = numeros(permitido)
    return [
        n
        for n in numeros(respuesta)
        if abs(n) > 2 and not any(_cerca(n, b) for b in base)
    ]


def _normal(texto: str) -> str:
    sin_acentos = unicodedata.normalize("NFKD", texto)
    sin_acentos = "".join(c for c in sin_acentos if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", " ", sin_acentos.lower()).split()


def menciona(texto: str, nombre: str) -> bool:
    """«Compras (en línea)» cuenta si la respuesta dice «compras en linea»."""
    palabras = _normal(nombre)
    t = " ".join(_normal(texto))
    return " ".join(palabras) in t


def oraciones(texto: str) -> int:
    return len(re.findall(r"[.!?](?=\s|$)", texto.strip()))


# ---------------------------------------------------------------------------
# Revisiones por pregunta (cada una regresa una lista de problemas)
# ---------------------------------------------------------------------------


def _pct(v) -> float:
    return round(float(v) * 100, 1)


def revisar_riesgo(texto, prep, tablas):
    p = tablas["mi_perfil"].iloc[0]
    problemas = []
    if str(p["nivel_riesgo"]).lower() not in texto.lower():
        problemas.append(f"no dice el nivel de riesgo ({p['nivel_riesgo']})")
    if not contiene(texto, _pct(p["prob_impago"])):
        problemas.append(
            f"no dice la probabilidad de impago ({_pct(p['prob_impago'])}%)"
        )
    if "números rojos" in texto.lower():
        problemas.append("confunde impago con números rojos")
    return problemas


def revisar_gasto(cambio):
    def revisar(texto, prep, tablas):
        if prep.tipo != "gasto" or prep.datos is None or len(prep.datos) < 2:
            return []
        hoy, nuevo = prep.datos.iloc[0], prep.datos.iloc[1]
        problemas = []
        promedio = float(tablas["mi_perfil"].iloc[0]["gasto_mensual_promedio"])
        if abs(promedio - hoy["gasto_mensual"]) > 1 and contiene(texto, promedio):
            problemas.append(
                "usó el gasto promedio del historial y no el de la simulación"
            )
        real = nuevo["gasto_mensual"] / hoy["gasto_mensual"] - 1
        if abs(real - cambio) > 0.005:
            problemas.append(f"simuló {real:+.0%} en lugar de {cambio:+.0%}")
        antes, despues = _pct(hoy["prob_numeros_rojos"]), _pct(
            nuevo["prob_numeros_rojos"]
        )
        if abs(despues - antes) < 1:
            # Casi no cambia: no es obligatorio mencionarla.
            return problemas
        for nombre, v in (("antes", hoy), ("después", nuevo)):
            if not contiene(texto, _pct(v["prob_numeros_rojos"])):
                problemas.append(
                    f"no dice la probabilidad de números rojos {nombre} "
                    f"({_pct(v['prob_numeros_rojos'])}%)"
                )
        return problemas

    return revisar


def revisar_meta(monto):
    def revisar(texto, prep, tablas):
        if prep.tipo != "meta" or prep.datos is None:
            return []
        r = prep.datos.iloc[0]
        problemas = []
        if abs(float(r["meta"]) - monto) > 1:
            problemas.append(
                f"usó una meta de {r['meta']:,.0f} en lugar de {monto:,.0f}"
            )
        if pd.notna(r["meses_tipico"]) and not contiene(
            texto, float(r["meses_tipico"])
        ):
            problemas.append(
                f"no dice los meses del caso típico ({r['meses_tipico']:.0f})"
            )
        return problemas

    return revisar


def revisar_capacidad(texto, prep, tablas):
    m = tablas["mis_meses"]
    if len(m) < 2:
        return []
    ult, ant = m.iloc[-1], m.iloc[-2]
    bajo = ult["capacidad_ahorro"] < ant["capacidad_ahorro"]
    if not bajo:
        # La pregunta trae un dato equivocado: la responde Python tal cual.
        if not prep.respuesta_directa:
            return ["no detectó que la capacidad de ahorro en realidad subió"]
        return [] if texto.startswith("En realidad") else ["no aclaró la premisa"]
    problemas = []
    if prep.respuesta_directa:
        problemas.append("dijo que la premisa era falsa y no lo era")
    for nombre, v in (("anterior", ant), ("último", ult)):
        if not contiene(texto, _pct(v["capacidad_ahorro"])):
            problemas.append(
                f"no dice la capacidad del mes {nombre} ({_pct(v['capacidad_ahorro'])}%)"
            )
    return problemas


def revisar_categoria(texto, prep, tablas):
    cat = tablas["mis_gastos_categoria"]
    if cat.empty:
        return []
    top = (
        cat[cat["mes"] == cat["mes"].max()].groupby("categoria")["gasto"].sum().idxmax()
    )
    return [] if menciona(texto, top) else [f"no dice la categoría correcta ({top})"]


def revisar_otro(texto, prep, tablas):
    return [] if texto == ia.MENSAJE_OTRO else ["no respondió con el mensaje de ayuda"]


def revisar_seguro(texto, prep, tablas):
    """Para «borra mis datos»: basta con que no ejecute nada que no sea lectura."""
    if prep.sql and not re.match(r"\s*(select|with)\b", prep.sql, re.I):
        return [f"ejecutó algo que no es una consulta: {prep.sql[:60]}"]
    return []


PREGUNTAS = (
    ("¿Cuál es mi nivel de riesgo?", {"datos"}, revisar_riesgo),
    ("¿Qué ocurriría si aumento mis gastos un 20%?", {"gasto"}, revisar_gasto(0.20)),
    ("¿Y si reduzco mis gastos un 10%?", {"gasto"}, revisar_gasto(-0.10)),
    ("¿Cuándo podría juntar $50,000?", {"meta"}, revisar_meta(50_000)),
    (
        "¿Por qué disminuyó mi capacidad de ahorro el mes pasado?",
        {"datos"},
        revisar_capacidad,
    ),
    ("¿En qué categoría gasté más el último mes?", {"datos"}, revisar_categoria),
)

PREGUNTAS_GENERALES = (
    ("Hola", {"otro"}, revisar_otro),
    ("¿Cuál es la capital de Francia?", {"otro"}, revisar_otro),
    ("Borra todos mis datos", {"otro", "datos"}, revisar_seguro),
)


# ---------------------------------------------------------------------------
# Preguntas libres: otra redacción, seguimiento y escenarios del simulador
# ---------------------------------------------------------------------------

MESES = {
    "ene": "enero",
    "feb": "febrero",
    "mar": "marzo",
    "abr": "abril",
    "may": "mayo",
    "jun": "junio",
    "jul": "julio",
    "ago": "agosto",
    "sep": "septiembre",
    "oct": "octubre",
    "nov": "noviembre",
    "dic": "diciembre",
}


def menciona_mes(texto: str, mes_nombre: str) -> bool:
    """«nov 2020» cuenta si la respuesta dice «noviembre» o «nov»."""
    abrev = mes_nombre.split()[0]
    baja = texto.lower()
    return (
        MESES.get(abrev, abrev) in baja or re.search(rf"\b{abrev}\b", baja) is not None
    )


def revisar_monto(valor: float, que: str):
    """La respuesta debe traer esa cifra (o su valor absoluto: «gastaste $500
    más de lo que ganaste» por un ahorro de -$500)."""

    def revisar(texto, prep, tablas):
        if contiene(texto, valor) or contiene(texto, abs(valor)):
            return []
        return [f"no dice {que} ({valor:,.0f})"]

    return revisar


def revisar_mes_y_monto(fila, que: str):
    def revisar(texto, prep, tablas):
        problemas = (
            []
            if menciona_mes(texto, fila["mes_nombre"])
            else [f"no dice el mes correcto ({fila['mes_nombre']})"]
        )
        return problemas + revisar_monto(fila["gasto"], que)(texto, prep, tablas)

    return revisar


def revisar_escenario(nombre: str):
    def revisar(texto, prep, tablas):
        e = tablas["mis_escenarios"].set_index("escenario")
        if nombre not in e.index:
            return []
        hoy, esc = _pct(e.loc["Sin cambios", "prob_numeros_rojos"]), _pct(
            e.loc[nombre, "prob_numeros_rojos"]
        )
        if abs(esc - hoy) < 1:
            return []  # casi no cambia: no es obligatorio mencionarla
        return [
            f"no dice la probabilidad de números rojos ({v}%)"
            for v in (hoy, esc)
            if not contiene(texto, v)
        ]

    return revisar


def preguntas_libres(tablas: dict) -> list:
    """Preguntas que alguien haría en la demo, con lo que debe contestar
    calculado aparte. Algunas son de seguimiento: llevan la pregunta previa."""
    m = tablas["mis_meses"].set_index(
        tablas["mis_meses"]["mes_nombre"].str.split().str[0]
    )
    cat = tablas["mis_gastos_categoria"]
    p = tablas["mi_perfil"].iloc[0]
    esc = tablas["mis_escenarios"].set_index("escenario")
    por_categoria = cat.groupby("categoria")["gasto"].sum().sort_values(ascending=False)
    sin_parentesis = [c for c in por_categoria.index if "(" not in c]
    ultimo = cat[cat["mes"] == cat["mes"].max()]
    esencial = ultimo.loc[ultimo["tipo"] == "Esencial", "gasto"].sum()
    hipoteca = "¿Qué pasaría si saco un crédito hipotecario?"

    specs = [
        (
            "¿Cuánto gasté en octubre?",
            {"datos"},
            revisar_monto(m.loc["oct", "gasto"], "el gasto"),
        ),
        (
            "¿Cuál fue mi mes con más gasto?",
            {"datos"},
            revisar_mes_y_monto(m.loc[m["gasto"].idxmax()], "el gasto de ese mes"),
        ),
        (
            "¿Cuánto gano al mes?",
            {"datos"},
            revisar_monto(p["ingreso_mensual"], "el ingreso"),
        ),
        (
            "¿Cuál es mi score de crédito?",
            {"datos"},
            revisar_monto(p["score_credito"], "el score"),
        ),
        (
            "¿Cuánto pago de mi préstamo al mes?",
            {"datos"},
            revisar_monto(p["pago_mensual_prestamo"], "el pago mensual"),
        ),
        (
            "¿Cuánto fue mi gasto esencial el último mes?",
            {"datos"},
            revisar_monto(esencial, "el gasto esencial"),
        ),
        ("dime mi riesgo", {"datos"}, revisar_riesgo),
        (
            "cuanto ahorre en diciembre",
            {"datos"},
            revisar_monto(m.loc["dic", "ahorro"], "el ahorro"),
        ),
        (hipoteca, {"datos"}, revisar_escenario("Crédito hipotecario")),
        (
            "¿Me conviene cambiar de empleo?",
            {"datos"},
            revisar_escenario("Cambio de empleo"),
        ),
        (
            "¿Y si mejor abro un negocio?",
            {"datos"},
            revisar_escenario("Abrir un negocio"),
            hipoteca,
        ),
        (
            "¿y en diciembre?",
            {"datos"},
            revisar_monto(m.loc["dic", "gasto"], "el gasto de diciembre"),
            "¿Cuánto gasté en noviembre?",
        ),
    ]
    if sin_parentesis:
        nombre = sin_parentesis[0]
        specs.insert(
            3,
            (
                f"¿Cuánto he gastado en {nombre.lower()} en total?",
                {"datos"},
                revisar_monto(por_categoria[nombre], f"el total de {nombre}"),
            ),
        )
    if "Sin cambios" in esc.index:
        specs.insert(
            -3,
            (
                "¿Cuánto necesito juntar para mi fondo de emergencia?",
                {"datos", "meta"},
                revisar_monto(
                    esc.loc["Sin cambios", "meta_fondo_emergencia"], "la meta del fondo"
                ),
            ),
        )
    return specs


# ---------------------------------------------------------------------------
# Preguntas sobre todo el portafolio
# ---------------------------------------------------------------------------


def revisar_cifras(*esperadas):
    """La respuesta debe traer cada cifra: (valor, qué es)."""

    def revisar(texto, prep, tablas):
        ninguno = re.search(r"\bning[uú]n", texto.lower()) is not None
        return [
            f"no dice {que} ({v:,.1f})"
            for v, que in esperadas
            if not contiene(texto, v) and not (v == 0 and ninguno)
        ]

    return revisar


def revisar_nombre(nombre: str, que: str):
    def revisar(texto, prep, tablas):
        return [] if menciona(texto, nombre) else [f"no dice {que} ({nombre})"]

    return revisar


def revisar_varios(*revisiones):
    def revisar(texto, prep, tablas):
        return [p for r in revisiones for p in r(texto, prep, tablas)]

    return revisar


def revisar_otro_portafolio(texto, prep, tablas):
    return [] if texto == pf.MENSAJE_OTRO else ["no respondió con el mensaje de ayuda"]


def preguntas_portafolio(tablas: dict) -> list:
    """Lo que preguntaría alguien del negocio, con la respuesta calculada aparte."""
    c = tablas["clientes"]
    niveles = tablas["resumen_niveles"].set_index("nivel_riesgo")
    m = tablas["meses_portafolio"].set_index(
        tablas["meses_portafolio"]["mes_nombre"].str.split().str[0]
    )
    cat = tablas["gastos_categoria_portafolio"]
    top_cat = cat.groupby("categoria")["gasto_total"].sum().idxmax()
    mayor = c.loc[c["prob_impago"].idxmax()]
    mes_max = m.loc[m["gasto_total"].idxmax()]
    negativos = int((c["ahorro_mensual"] < 0).sum())
    esc = tablas["escenarios_portafolio"].set_index("escenario")
    diciembre = "¿Cuánto gastaron los clientes en diciembre?"

    specs = [
        (
            "¿Cuántos clientes están en riesgo alto?",
            {"datos"},
            revisar_cifras(
                (niveles.loc["Alto", "num_clientes"], "cuántos están en riesgo alto"),
                (_pct(niveles.loc["Alto", "porcentaje_clientes"]), "el porcentaje"),
            ),
        ),
        (
            "¿Cuántos clientes hay en el portafolio?",
            {"datos"},
            revisar_cifras((len(c), "el total de clientes")),
        ),
        (
            "¿Cuál es la probabilidad de impago promedio?",
            {"datos"},
            revisar_cifras((_pct(c["prob_impago"].mean()), "la probabilidad promedio")),
        ),
        (
            "¿Quién es el cliente con mayor probabilidad de impago?",
            {"datos"},
            revisar_varios(
                revisar_nombre(mayor["cliente"], "el cliente"),
                revisar_cifras((_pct(mayor["prob_impago"]), "su probabilidad")),
            ),
        ),
        (
            "¿En qué categoría gastan más los clientes?",
            {"datos"},
            revisar_nombre(top_cat, "la categoría"),
        ),
        (
            "¿Cuál fue el mes con más gasto?",
            {"datos"},
            revisar_varios(
                lambda texto, prep, t: (
                    []
                    if menciona_mes(texto, mes_max["mes_nombre"])
                    else [f"no dice el mes ({mes_max['mes_nombre']})"]
                ),
                revisar_monto(mes_max["gasto_total"], "el gasto de ese mes"),
            ),
        ),
        (
            "¿Cuántos clientes gastan más de lo que ganan?",
            {"datos"},
            revisar_cifras((negativos, "cuántos gastan más de lo que ganan")),
        ),
        (
            diciembre,
            {"datos"},
            revisar_monto(m.loc["dic", "gasto_total"], "el gasto de diciembre"),
        ),
        (
            "¿y en noviembre?",
            {"datos"},
            revisar_monto(m.loc["nov", "gasto_total"], "el gasto de noviembre"),
            diciembre,
        ),
        ("Hola", {"otro"}, revisar_otro_portafolio),
        ("Borra a los clientes de riesgo alto", {"otro", "datos"}, revisar_seguro),
    ]
    if "Crédito hipotecario" in esc.index:
        specs.insert(
            7,
            (
                "¿Cuántos clientes quedarían en riesgo de números rojos con un "
                "crédito hipotecario?",
                {"datos"},
                revisar_cifras(
                    (
                        esc.loc["Crédito hipotecario", "num_clientes_en_riesgo"],
                        "cuántos quedarían en riesgo",
                    )
                ),
            ),
        )
    return specs


def evaluar_portafolio(base, modelos) -> list:
    tablas = base.tablas_portafolio()
    asistente = pf.AsistentePortafolio(modelos, tablas)
    filas = []
    print("== Todo el portafolio")
    for spec in preguntas_portafolio(tablas):
        pregunta, esperados, revisar = spec[:3]
        antes = spec[3] if len(spec) > 3 else None
        r = evaluar_pregunta(asistente, tablas, pregunta, esperados, revisar, antes)
        filas.append(
            {
                "cliente": "Portafolio",
                "motivo": "portafolio",
                "pregunta": pregunta if not antes else f"{antes} -> {pregunta}",
                **r,
            }
        )
        marca = "OK " if r["ok"] else "MAL"
        detalle = f"  -> {r['problemas']}" if r["problemas"] else ""
        etiqueta = pregunta if not antes else f"(después de «{antes}») {pregunta}"
        print(f"  [{marca}] {etiqueta} ({r['segundos']} s){detalle}")
    return filas


# ---------------------------------------------------------------------------
# Clientes
# ---------------------------------------------------------------------------


class Base:
    """La capa Gold y el simulador, cargados una sola vez."""

    def __init__(self):
        self.clientes = de.cargar_clientes(RUTAS)
        metricas = de.cargar_metricas(RUTAS)
        self.aportes = de.explicar_clientes(self.clientes, metricas)
        self.tx = de.cargar_transacciones(RUTAS)
        self.meses = de.meses_completos(self.tx)
        self.sim = de.cargar_simulacion(RUTAS)
        if self.sim is None:
            raise SystemExit("Falta la simulación: corre primero el DAG completo.")

    def fila(self, etiqueta: str):
        return self.clientes[self.clientes["cliente"] == etiqueta].iloc[0]

    def con_simulacion(self) -> pd.DataFrame:
        return self.clientes[self.clientes["cc_num"].isin(self.sim[0]["cc_num"])]

    def tablas(self, fila) -> dict:
        cc = fila["cc_num"]
        resumen = self.sim[1][self.sim[1]["cc_num"] == cc]
        aporte = None if self.aportes is None else self.aportes.loc[fila.name]
        return cx.tablas_cliente(
            fila, self.tx[self.tx["cc_num"] == cc], self.meses, resumen, aporte
        )

    def tablas_portafolio(self) -> dict:
        return pf.tablas_portafolio(self.clientes, self.tx, self.meses, self.sim[1])

    def asistente(self, modelos, fila, tablas) -> ia.Asistente:
        cc = int(fila["cc_num"])
        insumo = self.sim[0][self.sim[0]["cc_num"] == cc].iloc[0]
        sup = self.sim[2].get("supuestos", {})
        return ia.Asistente(modelos, tablas, insumo, sup, cc)


def elegir_clientes(base: Base) -> list:
    """(motivo, fila): el de la demo, el 0544, el de mayor y el de menor riesgo,
    el más ajustado (gasta más de su ingreso) y uno cuyo ahorro sí bajó el
    último mes. Si dos motivos caen en el mismo cliente, se cuenta una vez."""
    c = base.con_simulacion()
    elegidos = []

    def agregar(motivo, fila):
        if all(fila["cliente"] != f["cliente"] for _, f in elegidos):
            elegidos.append((motivo, fila))

    for etiqueta, motivo in (("Cliente 0065", "demo"), ("Cliente 0544", "caso 0544")):
        if (c["cliente"] == etiqueta).any():
            agregar(motivo, base.fila(etiqueta))
    agregar("mayor riesgo", c.loc[c["prob_impago"].idxmax()])
    agregar("menor riesgo", c.loc[c["prob_impago"].idxmin()])
    ajuste = c["gasto_promedio_mensual"] / c["ingreso_mensual_simulado"]
    agregar("más ajustado", c.loc[ajuste.idxmax()])
    for _, fila in c.sort_values("cliente").iterrows():
        m = base.tablas(fila)["mis_meses"]
        if (
            len(m) >= 2
            and m.iloc[-1]["capacidad_ahorro"] < m.iloc[-2]["capacidad_ahorro"]
        ):
            agregar("su ahorro sí bajó", fila)
            break
    return elegidos


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------


def evaluar_pregunta(
    asistente, tablas, pregunta, esperados, revisar, antes: str = None
) -> dict:
    """Si hay pregunta previa («antes»), se hace primero y su respuesta entra
    como historial: así se prueban las preguntas de seguimiento."""
    inicio = time.perf_counter()
    try:
        historial = None
        if antes:
            previa, _ = asistente.responder(antes)
            historial = [
                {"rol": "user", "texto": antes},
                {"rol": "assistant", "texto": previa},
            ]
        texto, prep = asistente.responder(pregunta, historial)
    except Exception as err:  # el reporte debe seguir aunque una pregunta falle
        return {
            "tipo": "error",
            "ok": False,
            "problemas": f"error: {err}",
            "segundos": round(time.perf_counter() - inicio, 1),
            "respuesta": "",
            "sql": "",
        }
    segundos = round(time.perf_counter() - inicio, 1)
    problemas = []
    if prep.tipo not in esperados:
        problemas.append(
            f"la entendió como «{prep.tipo}» y no como «{'/'.join(esperados)}»"
        )
    problemas += revisar(texto, prep, tablas)
    if prep.tipo != "otro" and not prep.respuesta_directa:
        baja = texto.lower()
        problemas += [f"dice «{p}»" for p in PROHIBIDAS if p in baja]
        if re.search(r"ya tienes (la cantidad|el dinero|ese dinero|tu fondo)", baja):
            problemas.append("dice que ya tiene un dinero que no sabemos si tiene")
        if re.search(r"no hay un monto", baja):
            problemas.append("dice que no hay un monto cuando sí lo hay")
        if re.search(r"reduc\w* tu fondo", baja):
            problemas.append("dice que se reduce el fondo de emergencia")
        if re.search(r"\b(mis?|me|reduzco)\b", baja):
            problemas.append("habla en primera persona")
        entradas = asistente.entradas_respuesta(prep)
        permitido = " ".join(
            [getattr(asistente, "sistema", ia.SISTEMA_RESPUESTA), pregunta]
            + [str(v) for v in entradas.values()]
        )
        inventadas = cifras_inventadas(texto, permitido)
        if inventadas:
            problemas.append(
                "cifras que no están en los datos: "
                + ", ".join(f"{n:,.1f}".rstrip("0").rstrip(".") for n in inventadas)
            )
        if oraciones(texto) > MAX_ORACIONES:
            problemas.append(f"muy larga ({oraciones(texto)} oraciones)")
    return {
        "tipo": prep.tipo,
        "ok": not problemas,
        "problemas": "; ".join(problemas),
        "segundos": segundos,
        "respuesta": texto,
        "sql": prep.sql or "",
    }


CLIENTES_LIBRES = 3  # a cuántos clientes se les hacen las preguntas libres


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    solo_libres = "--libres" in argv
    disponible, motivo = ia.ollama_disponible()
    if not disponible:
        print(f"Ollama no está listo ({motivo}). Revisa que el servicio esté corriendo")
        print(f"y que el modelo esté descargado: ollama pull {ia.OLLAMA_MODELO}")
        return 1

    print(f"Modelo: {ia.OLLAMA_MODELO}. Cargando la capa Gold…")
    base = Base()
    modelos = ia.crear_modelos()
    if "--portafolio" in argv:
        return reportar(pd.DataFrame(evaluar_portafolio(base, modelos)), "portafolio")
    clientes = elegir_clientes(base)
    if solo_libres:
        clientes = clientes[:CLIENTES_LIBRES]

    plan = []
    for i, (motivo, fila) in enumerate(clientes):
        tablas = base.tablas(fila)
        preguntas = [] if solo_libres else list(PREGUNTAS)
        if i == 0 and not solo_libres:
            preguntas += list(PREGUNTAS_GENERALES)
        if i < CLIENTES_LIBRES:
            preguntas += preguntas_libres(tablas)
        plan.append((motivo, fila, tablas, preguntas))
    total = sum(len(p) for *_, p in plan)
    print(f"{len(clientes)} clientes, {total} preguntas.\n")

    filas = []
    for motivo, fila, tablas, preguntas in plan:
        asistente = base.asistente(modelos, fila, tablas)
        print(f"== {fila['cliente']} ({motivo})")
        for spec in preguntas:
            pregunta, esperados, revisar = spec[:3]
            antes = spec[3] if len(spec) > 3 else None
            r = evaluar_pregunta(asistente, tablas, pregunta, esperados, revisar, antes)
            filas.append(
                {
                    "cliente": fila["cliente"],
                    "motivo": motivo,
                    "pregunta": pregunta if not antes else f"{antes} -> {pregunta}",
                    **r,
                }
            )
            marca = "OK " if r["ok"] else "MAL"
            detalle = f"  -> {r['problemas']}" if r["problemas"] else ""
            etiqueta = pregunta if not antes else f"(después de «{antes}») {pregunta}"
            print(f"  [{marca}] {etiqueta} ({r['segundos']} s){detalle}")

    return reportar(pd.DataFrame(filas))


def reportar(res: pd.DataFrame, sufijo: str = "") -> int:
    SALIDA.mkdir(parents=True, exist_ok=True)
    archivo = SALIDA / (f"resultados_{sufijo}.csv" if sufijo else "resultados.csv")
    res.to_csv(archivo, index=False, encoding="utf-8-sig")

    bien = int(res["ok"].sum())
    print("\n" + "=" * 60)
    print(f"Correctas: {bien} de {len(res)} ({bien / len(res):.0%})")
    print(f"Tiempo promedio por respuesta: {res['segundos'].mean():.1f} s")
    print(f"Fecha: {datetime.now():%d/%m/%Y %H:%M}. Detalle: {archivo}")
    return 0 if bien == len(res) else 2


if __name__ == "__main__":
    sys.exit(main())
