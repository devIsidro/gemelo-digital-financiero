"""
Simulador financiero estocástico (motor Monte Carlo) — Fase 5.

Para cada cliente genera miles de trayectorias posibles de su flujo de
efectivo mes a mes y resume el resultado en bandas de confianza, en lugar
de dar un solo número. Cubre el KPI flujo_efectivo_proyectado del catálogo
(config/kpis.yaml) y el "Simulador Financiero Estocástico" del documento
del proyecto.

Modelo, para cada mes t = 1 … meses (mes 1 = enero 2021, el mes siguiente
al último dato del dataset):

    ingreso_t = ingreso_mensual_simulado × (1 + aumento)^(t/12) × empleado_t
    gasto_t   = gasto_base × (1 + inflación)^(t/12) × estacional_t × ε_t
    flujo_t   = ingreso_t + extras_t − gasto_t − pagos_t
    ahorro_t  = ahorro_(t−1) + flujo_t

  - gasto_base y la variabilidad (ε, lognormal con media 1 y el coeficiente
    de variación del cliente) salen del gasto REAL con tarjeta de julio a
    noviembre de 2020. Diciembre se trata aparte con el factor estacional
    observado en el portafolio.
  - empleado_t: en cualquier mes el cliente puede perder su ingreso con una
    probabilidad pequeña y tarda en promedio unos meses en recuperarlo.
  - El ingreso es el simulado de la Opción A (el mismo de capacidad_ahorro);
    nunca el de la fuente de préstamos (decisión 2, docs/fase4_decisiones_kpis.md).
  - El ahorro arranca en 0: el dataset no trae saldos de cuentas, así que
    "ahorro acumulado" es lo que el cliente junta desde el mes 1. Por lo
    mismo, los pagos iniciales (enganche, inversión del negocio) se suponen
    cubiertos con ahorros previos; para el negocio se mide aparte en cuántos
    meses se recupera la inversión.

Escenarios: base, crédito hipotecario, cambio de empleo y apertura de un
negocio. Todos los supuestos están en Supuestos y en PARAMETROS_ESCENARIOS,
documentados en docs/fase5_simulador_monte_carlo.md.

Solo usa numpy y pandas: corre igual en el contenedor de Airflow (paso del
DAG) y en el del dashboard (simulación interactiva de un cliente).
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ESCENARIOS = ("base", "credito_hipotecario", "cambio_empleo", "negocio")
NOMBRES_ESCENARIOS = {
    "base": "Sin cambios",
    "credito_hipotecario": "Crédito hipotecario",
    "cambio_empleo": "Cambio de empleo",
    "negocio": "Abrir un negocio",
}
HORIZONTES = (1, 3, 6, 12, 24, 36)
PERCENTILES = (5, 25, 50, 75, 95)


@dataclass(frozen=True)
class Supuestos:
    """Supuestos generales (todos los escenarios)."""

    simulaciones: int = 1000
    meses: int = 36
    inflacion_anual: float = 0.03  # sube el gasto
    aumento_salarial_anual: float = 0.03  # sube el ingreso
    factor_diciembre: float = 1.79  # se recalcula con los datos en el job
    cv_minimo: float = 0.05
    cv_maximo: float = 0.80
    prob_perder_ingreso_mensual: float = 0.004  # ≈ 4.7% al año
    meses_sin_ingreso_promedio: float = 3.0
    meses_fondo_emergencia: float = 6.0  # meta: 6 meses de gasto
    semilla: int = 42


@dataclass(frozen=True)
class CreditoHipotecario:
    anos_de_ingreso: float = 3.0  # monto = 3 años de ingreso
    tasa_anual: float = 0.10
    plazo_meses: int = 240


@dataclass(frozen=True)
class CambioEmpleo:
    mes_cambio: int = 6
    meses_sin_ingreso_promedio: float = 1.0  # Poisson, máximo 4
    cambio_salario_min: float = -0.10
    cambio_salario_max: float = 0.25


@dataclass(frozen=True)
class Negocio:
    inversion_meses_ingreso: float = 6.0  # se paga con ahorros previos
    mes_primera_utilidad: int = 4
    meses_para_madurar: int = 12
    margen_medio: float = 0.25  # utilidad mensual madura, en % del ingreso
    margen_desviacion: float = 0.30
    prob_cierre_24m: float = 0.30


@dataclass(frozen=True)
class ParametrosEscenarios:
    credito_hipotecario: CreditoHipotecario = field(default_factory=CreditoHipotecario)
    cambio_empleo: CambioEmpleo = field(default_factory=CambioEmpleo)
    negocio: Negocio = field(default_factory=Negocio)


PARAMETROS_ESCENARIOS = ParametrosEscenarios()


def semilla_cliente(cc_num, escenario: str, semilla: int = 42) -> int:
    """Semilla fija por cliente y escenario: el job del DAG y el simulador
    interactivo del dashboard dan exactamente los mismos números."""
    return (semilla * 1_000_003 + int(cc_num) + ESCENARIOS.index(escenario)) % 2**32


def pago_mensual(monto: float, tasa_anual: float, plazo_meses: int) -> float:
    """Pago fijo de un crédito amortizable (la misma fórmula del DTI)."""
    r = tasa_anual / 12
    if r == 0:
        return monto / plazo_meses
    return monto * r / (1 - (1 + r) ** (-plazo_meses))


# ---------------------------------------------------------------------------
# Insumos por cliente (del gasto real)
# ---------------------------------------------------------------------------


def gasto_mensual_por_cliente(tx: pd.DataFrame) -> pd.DataFrame:
    """Gasto por cliente y mes, solo meses completos (el dataset empieza el
    21 de junio). Columnas = periodos mensuales."""
    fechas = tx["trans_date_trans_time"]
    periodo = fechas.dt.to_period("M")
    dias = fechas.dt.date.groupby(periodo).nunique()
    completos = [p for p, d in dias.items() if d == p.days_in_month]
    tabla = tx.groupby([tx["cc_num"], periodo])["amt"].sum().unstack(fill_value=0.0)
    return tabla.reindex(columns=completos, fill_value=0.0)


def insumos_clientes(
    tx: pd.DataFrame, kpis: pd.DataFrame, supuestos: Supuestos = Supuestos()
) -> tuple:
    """(insumos por cliente, factor de diciembre del portafolio).

    gasto_base = promedio de los meses completos fuera de diciembre;
    cv = desviación / promedio de esos meses, acotado a [cv_minimo, cv_maximo].
    Si un cliente no tiene esos meses, se usa su gasto_promedio_mensual de Gold.
    """
    mensual = gasto_mensual_por_cliente(tx)
    no_dic = [c for c in mensual.columns if c.month != 12]
    dic = [c for c in mensual.columns if c.month == 12]
    base = mensual[no_dic]
    media = base.mean(axis=1)
    cv = (base.std(axis=1) / media).clip(supuestos.cv_minimo, supuestos.cv_maximo)
    if dic and media.sum() > 0:
        factor_dic = float(mensual[dic].mean(axis=1).sum() / media.sum())
    else:
        factor_dic = supuestos.factor_diciembre

    ins = kpis[["cc_num", "ingreso_mensual_simulado", "gasto_promedio_mensual"]].copy()
    ins = ins.merge(
        media.rename("gasto_base"), left_on="cc_num", right_index=True, how="left"
    )
    ins = ins.merge(
        cv.rename("cv_gasto"), left_on="cc_num", right_index=True, how="left"
    )
    sin_datos = ins["gasto_base"].isna() | (ins["gasto_base"] <= 0)
    ins.loc[sin_datos, "gasto_base"] = ins.loc[sin_datos, "gasto_promedio_mensual"]
    ins["cv_gasto"] = ins["cv_gasto"].fillna(
        np.median(cv.dropna()) if len(cv.dropna()) else 0.25
    )
    ins = ins.rename(columns={"ingreso_mensual_simulado": "ingreso"})
    return ins[["cc_num", "ingreso", "gasto_base", "cv_gasto"]], round(factor_dic, 3)


# ---------------------------------------------------------------------------
# Motor
# ---------------------------------------------------------------------------


def _empleo(rng, n, meses, p_perder, meses_promedio):
    """Matriz n × meses con 1 = tiene ingreso, 0 = lo perdió (cadena de Markov)."""
    q = 1.0 / max(meses_promedio, 1.0)
    empleado = np.ones((n, meses))
    actual = np.ones(n, dtype=bool)
    for t in range(meses):
        u = rng.random(n)
        actual = np.where(actual, u >= p_perder, u < q)
        empleado[:, t] = actual
    return empleado


def simular_cliente(
    ingreso: float,
    gasto_base: float,
    cv_gasto: float,
    escenario: str = "base",
    supuestos: Supuestos = Supuestos(),
    parametros: ParametrosEscenarios = PARAMETROS_ESCENARIOS,
    semilla: int = None,
) -> dict:
    """Simula un cliente en un escenario. Devuelve matrices simulaciones × meses
    de ingreso, gasto, flujo y ahorro acumulado, más datos del escenario."""
    if escenario not in ESCENARIOS:
        raise ValueError(f"Escenario desconocido: {escenario}")
    s = supuestos
    n, m = s.simulaciones, s.meses
    rng = np.random.default_rng(s.semilla if semilla is None else semilla)
    t = np.arange(1, m + 1)

    # Gasto: inflación, diciembre y variabilidad del propio cliente.
    estacional = np.where(t % 12 == 0, s.factor_diciembre, 1.0)
    sigma = np.sqrt(np.log(1 + cv_gasto**2))
    eps = rng.lognormal(mean=-(sigma**2) / 2, sigma=sigma, size=(n, m))
    gasto = gasto_base * (1 + s.inflacion_anual) ** (t / 12) * estacional * eps

    # Ingreso: aumento anual y riesgo de perderlo por algunos meses.
    tendencia = ingreso * (1 + s.aumento_salarial_anual) ** (t / 12)
    empleado = _empleo(
        rng, n, m, s.prob_perder_ingreso_mensual, s.meses_sin_ingreso_promedio
    )
    ingreso_t = tendencia * empleado

    extras = np.zeros((n, m))  # ingresos adicionales (utilidad del negocio)
    pagos = np.zeros((n, m))  # pagos fijos (crédito)
    info = {}
    utilidad = None

    if escenario == "credito_hipotecario":
        c = parametros.credito_hipotecario
        monto = c.anos_de_ingreso * 12 * ingreso
        pago = pago_mensual(monto, c.tasa_anual, c.plazo_meses)
        pagos[:, :] = pago
        info = {
            "monto_credito": monto,
            "pago_mensual": pago,
            "pago_entre_ingreso": pago / ingreso if ingreso > 0 else None,
        }

    elif escenario == "cambio_empleo":
        c = parametros.cambio_empleo
        inicio = min(max(c.mes_cambio, 1), m) - 1
        hueco = np.minimum(rng.poisson(c.meses_sin_ingreso_promedio, n), 4)
        cambio = rng.uniform(c.cambio_salario_min, c.cambio_salario_max, n)
        meses_idx = np.arange(m)
        sin_ingreso = (meses_idx >= inicio) & (meses_idx < inicio + hueco[:, None])
        despues = meses_idx >= inicio + hueco[:, None]
        factor = np.where(sin_ingreso, 0.0, np.where(despues, 1 + cambio[:, None], 1.0))
        ingreso_t = ingreso_t * factor
        info = {
            "meses_sin_ingreso_promedio": float(hueco.mean()),
            "cambio_salario_promedio": float(cambio.mean()),
        }

    elif escenario == "negocio":
        c = parametros.negocio
        margen = rng.normal(c.margen_medio, c.margen_desviacion, n)
        rampa = np.clip((t - c.mes_primera_utilidad + 1) / c.meses_para_madurar, 0, 1)
        cierra = rng.random(n) < c.prob_cierre_24m
        mes_cierre = np.where(
            cierra, rng.integers(c.mes_primera_utilidad, 25, n), m + 1
        )
        activo = t[None, :] < mes_cierre[:, None]
        extras = margen[:, None] * ingreso * rampa[None, :] * activo
        utilidad = np.cumsum(extras, axis=1)
        info = {
            "inversion": c.inversion_meses_ingreso * ingreso,
            "prob_cierre": float(cierra.mean()),
        }

    flujo = ingreso_t + extras - gasto - pagos
    ahorro = np.cumsum(flujo, axis=1)
    return {
        "ingreso": ingreso_t,
        "gasto": gasto + pagos,
        "flujo": flujo,
        "ahorro": ahorro,
        "utilidad_negocio": utilidad,
        "info": info,
    }


def resumir(sim: dict, gasto_base: float, supuestos: Supuestos = Supuestos()) -> dict:
    """Indicadores de un cliente en un escenario."""
    flujo, ahorro, ingreso = sim["flujo"], sim["ahorro"], sim["ingreso"]
    res = {}
    for h in HORIZONTES:
        if h > flujo.shape[1]:
            continue
        f = np.percentile(flujo[:, h - 1], [5, 50, 95])
        a = np.percentile(ahorro[:, h - 1], [5, 50, 95])
        res.update(
            {
                f"flujo_m{h}_p5": f[0],
                f"flujo_m{h}_p50": f[1],
                f"flujo_m{h}_p95": f[2],
                f"ahorro_m{h}_p5": a[0],
                f"ahorro_m{h}_p50": a[1],
                f"ahorro_m{h}_p95": a[2],
            }
        )
    negativo = ahorro < 0
    res["prob_deficit_12m"] = float(negativo[:, :12].any(axis=1).mean())
    res["prob_deficit"] = float(negativo.any(axis=1).mean())
    anio1 = slice(0, min(12, flujo.shape[1]))
    ing1 = ingreso[:, anio1].sum(axis=1)
    cap = np.where(
        ing1 > 0, flujo[:, anio1].sum(axis=1) / np.where(ing1 > 0, ing1, 1), np.nan
    )
    res["capacidad_ahorro_12m_p50"] = (
        float(np.nanmedian(cap)) if np.isfinite(cap).any() else None
    )
    meta = supuestos.meses_fondo_emergencia * gasto_base
    alcanza = ahorro >= meta
    primer = np.where(alcanza.any(axis=1), alcanza.argmax(axis=1) + 1, np.inf)
    mediana = float(np.median(primer))
    res["meta_fondo_emergencia"] = meta
    res["meses_meta_p50"] = mediana if np.isfinite(mediana) else None
    res["prob_meta_36m"] = float(np.isfinite(primer).mean())

    # Solo negocio: ¿en cuánto tiempo la utilidad acumulada paga la inversión?
    res["prob_recuperar_inversion"] = None
    res["meses_recuperacion_p50"] = None
    if sim.get("utilidad_negocio") is not None:
        inversion = sim["info"]["inversion"]
        recupera = sim["utilidad_negocio"] >= inversion
        mes = np.where(recupera.any(axis=1), recupera.argmax(axis=1) + 1, np.inf)
        res["prob_recuperar_inversion"] = float(np.isfinite(mes).mean())
        med = float(np.median(mes))
        res["meses_recuperacion_p50"] = med if np.isfinite(med) else None
    return res


def bandas(sim: dict, variable: str = "ahorro") -> pd.DataFrame:
    """Percentiles mes a mes de una variable (para las gráficas de abanico)."""
    datos = sim[variable]
    q = np.percentile(datos, PERCENTILES, axis=0)
    df = pd.DataFrame(q.T, columns=[f"p{p}" for p in PERCENTILES])
    df.insert(0, "mes", np.arange(1, datos.shape[1] + 1))
    return df


# ---------------------------------------------------------------------------
# Job del DAG
# ---------------------------------------------------------------------------


def simular_portafolio(insumos: pd.DataFrame, supuestos: Supuestos) -> tuple:
    """(resumen por cliente y escenario, bandas mensuales del ahorro)."""
    filas, tray = [], []
    for c in insumos.itertuples(index=False):
        for esc in ESCENARIOS:
            sim = simular_cliente(
                c.ingreso,
                c.gasto_base,
                c.cv_gasto,
                esc,
                supuestos,
                semilla=semilla_cliente(c.cc_num, esc, supuestos.semilla),
            )
            filas.append(
                {
                    "cc_num": c.cc_num,
                    "escenario": esc,
                    **resumir(sim, c.gasto_base, supuestos),
                }
            )
            b = bandas(sim)
            b.insert(0, "escenario", esc)
            b.insert(0, "cc_num", c.cc_num)
            tray.append(b)
    return pd.DataFrame(filas), pd.concat(tray, ignore_index=True)


def validar(resumen: pd.DataFrame, trayectorias: pd.DataFrame, clientes: int) -> list:
    """Reglas que el resultado debe cumplir; regresa la lista de fallas."""
    fallas = []
    if resumen["cc_num"].nunique() != clientes:
        fallas.append(f"{resumen['cc_num'].nunique()} clientes simulados de {clientes}")
    if set(resumen["escenario"]) != set(ESCENARIOS):
        fallas.append("faltan escenarios")
    clave = [c for c in resumen.columns if c.startswith(("flujo_", "ahorro_"))]
    if resumen[clave].isna().any().any():
        fallas.append("hay percentiles vacíos")
    for h in HORIZONTES:
        if not (resumen[f"ahorro_m{h}_p5"] <= resumen[f"ahorro_m{h}_p95"]).all():
            fallas.append(f"percentiles desordenados en el mes {h}")
    orden = trayectorias[[f"p{p}" for p in PERCENTILES]].diff(axis=1).iloc[:, 1:]
    if (orden < -1e-6).any().any():
        fallas.append("bandas desordenadas")
    return fallas


def run_simulador_job(
    kpis_path: str = "data/gold/kpis_cliente",
    silver_tx_path: str = "data/silver/transactions",
    salida_dir: str = "data/gold",
    supuestos: Supuestos = None,
) -> dict:
    """Calcula los insumos, simula los 4 escenarios para todos los clientes y
    escribe en Gold: simulacion_insumos, simulacion_resumen,
    simulacion_trayectorias y simulacion_parametros.json."""
    kpis = pd.read_parquet(kpis_path)
    tx = pd.read_parquet(
        silver_tx_path, columns=["cc_num", "amt", "trans_date_trans_time"]
    )
    base = supuestos or Supuestos()
    insumos, factor_dic = insumos_clientes(tx, kpis, base)
    sup = Supuestos(**{**asdict(base), "factor_diciembre": factor_dic})

    resumen, trayectorias = simular_portafolio(insumos, sup)
    fallas = validar(resumen, trayectorias, len(kpis))
    if fallas:
        raise ValueError("Simulación inválida: " + "; ".join(fallas))

    salida = Path(salida_dir)
    salida.mkdir(parents=True, exist_ok=True)
    insumos.to_parquet(salida / "simulacion_insumos.parquet", index=False)
    resumen.to_parquet(salida / "simulacion_resumen.parquet", index=False)
    trayectorias.to_parquet(salida / "simulacion_trayectorias.parquet", index=False)
    parametros = {
        "supuestos": asdict(sup),
        "escenarios": asdict(PARAMETROS_ESCENARIOS),
        "clientes": len(insumos),
        "escenarios_nombres": NOMBRES_ESCENARIOS,
    }
    (salida / "simulacion_parametros.json").write_text(
        json.dumps(parametros, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    por_esc = resumen.groupby("escenario")["prob_deficit"].median().to_dict()
    return {
        "clientes": len(insumos),
        "factor_diciembre": factor_dic,
        "prob_deficit_mediana": por_esc,
    }


if __name__ == "__main__":
    print(run_simulador_job())
