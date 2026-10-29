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

Los resultados completos (respuesta, SQL, problemas) quedan en
data/evaluacion_ia/resultados.csv.
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


def evaluar_pregunta(asistente, tablas, pregunta, esperados, revisar) -> dict:
    inicio = time.perf_counter()
    try:
        texto, prep = asistente.responder(pregunta)
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
        if re.search(r"\b(mis?|me|reduzco)\b", baja):
            problemas.append("habla en primera persona")
        entradas = asistente.entradas_respuesta(prep)
        permitido = " ".join(
            [ia.SISTEMA_RESPUESTA, pregunta] + [str(v) for v in entradas.values()]
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


def main() -> int:
    disponible, motivo = ia.ollama_disponible()
    if not disponible:
        print(f"Ollama no está listo ({motivo}). Revisa que el servicio esté corriendo")
        print(f"y que el modelo esté descargado: ollama pull {ia.OLLAMA_MODELO}")
        return 1

    print(f"Modelo: {ia.OLLAMA_MODELO}. Cargando la capa Gold…")
    base = Base()
    modelos = ia.crear_modelos()
    clientes = elegir_clientes(base)
    total = len(clientes) * len(PREGUNTAS) + len(PREGUNTAS_GENERALES)
    print(f"{len(clientes)} clientes, {total} preguntas.\n")

    filas = []
    for i, (motivo, fila) in enumerate(clientes):
        tablas = base.tablas(fila)
        asistente = base.asistente(modelos, fila, tablas)
        preguntas = PREGUNTAS + (PREGUNTAS_GENERALES if i == 0 else ())
        print(f"== {fila['cliente']} ({motivo})")
        for pregunta, esperados, revisar in preguntas:
            r = evaluar_pregunta(asistente, tablas, pregunta, esperados, revisar)
            filas.append(
                {
                    "cliente": fila["cliente"],
                    "motivo": motivo,
                    "pregunta": pregunta,
                    **r,
                }
            )
            marca = "OK " if r["ok"] else "MAL"
            detalle = f"  -> {r['problemas']}" if r["problemas"] else ""
            print(f"  [{marca}] {pregunta} ({r['segundos']} s){detalle}")

    res = pd.DataFrame(filas)
    SALIDA.mkdir(parents=True, exist_ok=True)
    res.to_csv(SALIDA / "resultados.csv", index=False, encoding="utf-8-sig")

    bien = int(res["ok"].sum())
    print("\n" + "=" * 60)
    print(f"Correctas: {bien} de {len(res)} ({bien / len(res):.0%})")
    print(f"Tiempo promedio por respuesta: {res['segundos'].mean():.1f} s")
    print(
        f"Fecha: {datetime.now():%d/%m/%Y %H:%M}. Detalle: {SALIDA / 'resultados.csv'}"
    )
    return 0 if bien == len(res) else 2


if __name__ == "__main__":
    sys.exit(main())
