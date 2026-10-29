"""
Hechos clave ya calculados para el asistente de IA (Fase 5).

Un modelo pequeño (Llama 3 de 8B) redacta bien, pero se equivoca al comparar
números por su cuenta: confunde meses, resta mal o mezcla conceptos. Por eso
las comparaciones importantes se calculan aquí, en Python, a partir de los
mismos datos que recupera el asistente, y el modelo solo las redacta.
"""

import re

import pandas as pd


def _usd(v) -> str:
    return f"-${-v:,.0f}" if v < 0 else f"${v:,.0f}"


def _pct(v) -> str:
    return f"{v * 100:.1f}%"


def _puntos(antes: float, despues: float) -> str:
    """Diferencia entre dos probabilidades en puntos porcentuales (no en %)."""
    d = (despues - antes) * 100
    if abs(d) < 0.05:
        return "prácticamente igual"
    lado = "más" if d > 0 else "menos"
    return f"{abs(d):.1f} puntos porcentuales {lado}"


def _direccion(antes: float, despues: float) -> str:
    if despues > antes:
        return "subió"
    if despues < antes:
        return "bajó"
    return "se mantuvo igual"


def _cambio_categorias(cat: pd.DataFrame, mes_antes, mes_despues) -> pd.Series:
    """Diferencia de gasto por categoría entre dos meses."""
    t = cat[cat["mes"].isin([mes_antes, mes_despues])].pivot_table(
        index="categoria", columns="mes", values="gasto", aggfunc="sum", fill_value=0
    )
    for m in (mes_antes, mes_despues):
        if m not in t.columns:
            t[m] = 0.0
    return (t[mes_despues] - t[mes_antes]).sort_values(), t


def _frase_categoria(
    cambios: pd.Series, t: pd.DataFrame, antes, despues, nombres
) -> list:
    frases = []
    if cambios.empty:
        return frases
    sube = cambios.index[-1]
    if cambios.iloc[-1] > 0:
        frases.append(
            f"De {nombres[0]} a {nombres[1]}, la categoría donde más aumentó tu gasto fue "
            f"{sube}: de {_usd(t.loc[sube, antes])} a {_usd(t.loc[sube, despues])}."
        )
    baja = cambios.index[0]
    if cambios.iloc[0] < 0:
        frases.append(
            f"De {nombres[0]} a {nombres[1]}, la categoría donde más bajó tu gasto fue "
            f"{baja}: de {_usd(t.loc[baja, antes])} a {_usd(t.loc[baja, despues])}."
        )
    return frases


def hechos_meses(tablas: dict) -> list:
    """El último mes contra el anterior, y el mes de menor ahorro."""
    m = tablas["mis_meses"]
    if len(m) < 2:
        return []
    cat = tablas["mis_gastos_categoria"]
    ult, ant = m.iloc[-1], m.iloc[-2]
    hechos = [
        f"El último mes con datos es {ult['mes_nombre']} y el anterior es "
        f"{ant['mes_nombre']}. Tu ingreso fue el mismo los dos meses "
        f"({_usd(ult['ingreso'])}).",
        f"De {ant['mes_nombre']} a {ult['mes_nombre']} tu gasto "
        f"{_direccion(ant['gasto'], ult['gasto'])}: de {_usd(ant['gasto'])} a "
        f"{_usd(ult['gasto'])}. Por eso tu capacidad de ahorro "
        f"{_direccion(ant['capacidad_ahorro'], ult['capacidad_ahorro'])}: de "
        f"{_pct(ant['capacidad_ahorro'])} a {_pct(ult['capacidad_ahorro'])}.",
    ]
    cambios, t = _cambio_categorias(cat, ant["mes"], ult["mes"])
    hechos += _frase_categoria(
        cambios, t, ant["mes"], ult["mes"], (ant["mes_nombre"], ult["mes_nombre"])
    )

    i_min = int(m["capacidad_ahorro"].to_numpy().argmin())
    bajo = m.iloc[i_min]
    hechos.append(
        f"Tu mes con la capacidad de ahorro más baja fue {bajo['mes_nombre']} "
        f"({_pct(bajo['capacidad_ahorro'])}), cuando gastaste {_usd(bajo['gasto'])}."
    )
    if i_min > 0 and i_min != len(m) - 1:
        previo = m.iloc[i_min - 1]
        hechos.append(
            f"En {bajo['mes_nombre']} tu capacidad de ahorro bajó de "
            f"{_pct(previo['capacidad_ahorro'])} (en {previo['mes_nombre']}) a "
            f"{_pct(bajo['capacidad_ahorro'])}."
        )
        cambios, t = _cambio_categorias(cat, previo["mes"], bajo["mes"])
        hechos += _frase_categoria(
            cambios,
            t,
            previo["mes"],
            bajo["mes"],
            (previo["mes_nombre"], bajo["mes_nombre"]),
        )
    return hechos


def hechos_riesgo(tablas: dict) -> list:
    p = tablas["mi_perfil"].iloc[0]
    rango = {"Bajo": "menos de 10%", "Medio": "entre 10% y 30%", "Alto": "30% o más"}
    hechos = [
        f"Tu probabilidad de no pagar tu préstamo es {_pct(p['prob_impago'])}, "
        f"por eso tu nivel de riesgo es {p['nivel_riesgo']} "
        f"({rango.get(p['nivel_riesgo'], '')})."
    ]
    sube, baja, _ = factores_riesgo(tablas)
    return hechos + [frase for frase in (sube, baja) if frase]


_SUSTANTIVOS = r"Tasa de interés|Ingreso|Préstamo|Score de crédito|Plazo"


def _en_tu_voz(factor: str) -> str:
    """'Su pago mensual es 2.2 veces su ingreso' -> 'tu pago mensual es 2.2
    veces tu ingreso'; 'Tasa de interés de 18.0%' -> 'tu tasa de interés es de
    18.0%'; 'Sin aval' -> 'no tienes aval'."""
    f = re.sub(r"(\d[\d,]*) USD", r"$\1", factor.strip())
    f = re.sub(r"\b[Ss]u\b", "tu", f)
    m = re.match(rf"({_SUSTANTIVOS}) de (.+)$", f)
    if m:
        return f"tu {m.group(1).lower()} es de {m.group(2)}"
    for antes, ahora in (
        ("Lleva ", "llevas "),
        ("Tiene ", "tienes "),
        ("Está ", "estás "),
        ("Sin ", "no tienes "),
    ):
        if f.startswith(antes):
            return ahora + f.removeprefix(antes)
    if f[:1].isdigit():
        return "tienes " + f
    if ": " in f:
        nombre, valor = f.split(": ", 1)
        return f"tu {nombre.lower()} es {valor}"
    return f[:1].lower() + f[1:]


def _unir(partes: list) -> str:
    return (
        partes[0] if len(partes) == 1 else ", ".join(partes[:-1]) + " y " + partes[-1]
    )


def _clave(factor: str) -> str:
    """Lo que tiene que aparecer en la respuesta para saber que mencionó el
    factor: su número (2.2, 18.0, 322) o, si no tiene, su última palabra."""
    numero = re.search(r"\d[\d,.]*\d|\d", factor)
    if numero and len(numero.group(0)) > 1:
        return numero.group(0)
    # Un solo dígito («1 línea de crédito») aparecería en cualquier cifra.
    return factor.split()[-1].lower()


def factores_riesgo(tablas: dict) -> tuple:
    """(frase de lo que sube el riesgo, frase de lo que lo baja, claves de lo
    que lo sube) ya redactadas en Python, en segunda persona."""
    f = tablas["mis_factores_riesgo"]
    if f.empty:
        return "", "", []
    sube = list(f.loc[f["efecto"] == "Sube el riesgo", "factor"])[:3]
    baja = list(f.loc[f["efecto"] == "Baja el riesgo", "factor"])[:2]
    frase_sube = (
        "Lo que más sube tu riesgo es que " + _unir([_en_tu_voz(x) for x in sube]) + "."
        if sube
        else ""
    )
    frase_baja = (
        "Lo que lo baja es que " + _unir([_en_tu_voz(x) for x in baja]) + "."
        if baja
        else ""
    )
    claves = [_clave(x) for x in sube]
    # «18.0» también puede aparecer como «18%».
    claves += [c[:-2] + "%" for c in claves if c.endswith(".0")]
    return frase_sube, frase_baja, claves


def hechos_gasto(datos: pd.DataFrame) -> list:
    hoy, nuevo = datos.iloc[0], datos.iloc[1]

    def meses(v):
        return "más de 36 meses" if pd.isna(v) else f"{v:.0f} meses"

    dif_igual = "igual" in _puntos(
        hoy["prob_numeros_rojos"], nuevo["prob_numeros_rojos"]
    )
    return [
        f"Escenario: {nuevo['situacion'].lower()} (gasto mensual de "
        f"{_usd(hoy['gasto_mensual'])} a {_usd(nuevo['gasto_mensual'])}).",
        f"Tu probabilidad de quedarte en números rojos en 36 meses pasaría de "
        f"{_pct(hoy['prob_numeros_rojos'])} a {_pct(nuevo['prob_numeros_rojos'])}, "
        f"o sea {_puntos(hoy['prob_numeros_rojos'], nuevo['prob_numeros_rojos'])}"
        f"{'' if dif_igual else ' de probabilidad'}. "
        "Números rojos significa que tu ahorro acumulado baje de cero; NO es la "
        "probabilidad de dejar de pagar tu préstamo.",
        f"Tu ahorro acumulado típico a 12 meses pasaría de {_usd(hoy['ahorro_12m'])} "
        f"a {_usd(nuevo['ahorro_12m'])}.",
        f"Juntar tu fondo de emergencia te tomaría {meses(nuevo['meses_fondo_emergencia'])} "
        f"en lugar de {meses(hoy['meses_fondo_emergencia'])}.",
    ]


def hechos_meta(datos: pd.DataFrame) -> list:
    r = datos.iloc[0]
    hechos = [
        f"Meta: juntar {_usd(r['meta'])}. Hoy ahorras en promedio "
        f"{_usd(r['ahorro_mensual_tipico'])} al mes."
    ]
    if pd.isna(r["meses_tipico"]):
        hechos.append(
            "Si sigues como hoy, en la mayoría de los futuros simulados no la alcanzas "
            "en 36 meses."
        )
    else:
        hechos.append(
            f"Si sigues como hoy, lo más probable es que la alcances en "
            f"{r['meses_tipico']:.0f} meses."
        )
        if pd.notna(r["meses_caso_malo"]) and r["meses_caso_malo"] > r["meses_tipico"]:
            hechos.append(
                f"Si te va un poco peor de lo normal, te tomaría unos "
                f"{r['meses_caso_malo']:.0f} meses."
            )
    hechos.append(
        f"De cada 100 futuros simulados, en {r['prob_lograrlo_12m'] * 100:.0f} la alcanzas "
        f"dentro de los primeros 12 meses y en {r['prob_lograrlo_36m'] * 100:.0f} dentro "
        "de 36 meses."
    )
    return hechos


_BAJO = re.compile(
    r"\b(baj[óo]|bajaron|disminuy[óo]|disminuyeron|cay[óo]|redujo|empeor[óo])\b"
)
_SUBIO = re.compile(
    r"\b(subi[óo]|subieron|aument[óo]|aumentaron|creci[óo]|increment[óo]|mejor[óo])\b"
)


def _mes_con_cambio(m: pd.DataFrame, columna: str, supuesto: str):
    """Índice del mes más reciente (antes del último) donde la columna sí tuvo
    el cambio que supone la pregunta respecto al mes anterior."""
    for k in range(len(m) - 2, 0, -1):
        if _direccion(m.iloc[k - 1][columna], m.iloc[k][columna]) == supuesto:
            return k
    return None


def revisar_premisa(pregunta: str, tablas: dict) -> str:
    """Si la pregunta supone que algo bajó o subió el último mes y los datos
    dicen lo contrario, regresa la respuesta completa ya escrita: la
    aclaración y el mes en que sí ocurrió ese cambio, con su causa. Si la
    pregunta no supone nada falso, regresa una cadena vacía."""
    texto = pregunta.lower()
    if "capacidad" in texto or "ahorr" in texto:
        columna, nombre = "capacidad_ahorro", "capacidad de ahorro"
    elif "gast" in texto:
        columna, nombre = "gasto", "gasto"
    else:
        return ""
    supuesto = (
        "bajó" if _BAJO.search(texto) else "subió" if _SUBIO.search(texto) else None
    )
    m = tablas["mis_meses"]
    if supuesto is None or len(m) < 2:
        return ""
    ult, ant = m.iloc[-1], m.iloc[-2]
    real = _direccion(ant[columna], ult[columna])
    if real == supuesto:
        return ""
    fmt = _pct if columna == "capacidad_ahorro" else _usd
    inicio = (
        f"En realidad, tu {nombre} {real} el último mes: de {fmt(ant[columna])} en "
        f"{ant['mes_nombre']} a {fmt(ult[columna])} en {ult['mes_nombre']}"
    )
    if columna == "capacidad_ahorro":
        inicio += (
            f", porque tu gasto {_direccion(ant['gasto'], ult['gasto'])} de "
            f"{_usd(ant['gasto'])} a {_usd(ult['gasto'])}"
        )
    inicio += "."
    k = _mes_con_cambio(m, columna, supuesto)
    if k is None:
        explicacion = f"En tu historial no hay un mes en que tu {nombre} {supuesto}."
    else:
        mes, previo = m.iloc[k], m.iloc[k - 1]
        explicacion = (
            f"Donde sí {supuesto} fue en {mes['mes_nombre']}: de "
            f"{fmt(previo[columna])} en {previo['mes_nombre']} a {fmt(mes[columna])}"
        )
        if columna == "capacidad_ahorro":
            explicacion += (
                f", porque tu gasto {_direccion(previo['gasto'], mes['gasto'])} de "
                f"{_usd(previo['gasto'])} a {_usd(mes['gasto'])} con el mismo ingreso"
            )
        explicacion += "."
        cambios, t = _cambio_categorias(
            tablas["mis_gastos_categoria"], previo["mes"], mes["mes"]
        )
        # Si el ahorro bajó (o el gasto subió) importa la categoría que más subió.
        quiere_subida = (columna == "gasto") == (supuesto == "subió")
        if not cambios.empty:
            cat = cambios.index[-1] if quiere_subida else cambios.index[0]
            dif = cambios.loc[cat]
            if (dif > 0) == quiere_subida and dif != 0:
                explicacion += (
                    f" Lo que más {'subió' if dif > 0 else 'bajó'} fue {cat}: de "
                    f"{_usd(t.loc[cat, previo['mes']])} a {_usd(t.loc[cat, mes['mes']])}."
                )
    return f"{inicio} {explicacion}"
