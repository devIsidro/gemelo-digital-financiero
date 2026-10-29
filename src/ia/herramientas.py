"""
Herramientas del asistente que usan el simulador Monte Carlo (Fase 5).

Responden las preguntas de "¿qué pasaría si...?" y "¿cuándo podría...?" del
documento del proyecto con el mismo motor y la misma semilla que el
dashboard, así que los números coinciden con lo que se ve en Perfil 360.
"""

import numpy as np
import pandas as pd

from src.ml import simulador as sm

LIMITE_CAMBIO_GASTO = (-0.9, 3.0)  # de −90% a +300%


def _supuestos(sup: dict) -> sm.Supuestos:
    campos = sm.Supuestos.__dataclass_fields__
    return sm.Supuestos(**{k: v for k, v in (sup or {}).items() if k in campos})


def _simular(insumo: pd.Series, gasto_base: float, sup: sm.Supuestos, cc_num) -> dict:
    return sm.simular_cliente(
        float(insumo["ingreso"]),
        gasto_base,
        float(insumo["cv_gasto"]),
        "base",
        sup,
        semilla=sm.semilla_cliente(cc_num, "base", sup.semilla),
    )


def _fila(nombre: str, sim: dict, gasto_base: float, sup: sm.Supuestos) -> dict:
    r = sm.resumir(sim, gasto_base, sup)
    return {
        "situacion": nombre,
        "gasto_mensual": round(gasto_base, 2),
        "prob_numeros_rojos": round(r["prob_deficit"], 4),
        "ahorro_12m": round(r["ahorro_m12_p50"], 2),
        "ahorro_36m": round(r["ahorro_m36_p50"], 2),
        "capacidad_ahorro_12m": r["capacidad_ahorro_12m_p50"],
        "meses_fondo_emergencia": r["meses_meta_p50"],
    }


def simular_cambio_gasto(
    insumo: pd.Series, cambio: float, sup: dict, cc_num
) -> pd.DataFrame:
    """Compara el futuro del cliente como está hoy contra gastando
    `cambio` más (0.2 = 20% más; −0.1 = 10% menos)."""
    cambio = float(np.clip(cambio, *LIMITE_CAMBIO_GASTO))
    s = _supuestos(sup)
    base = float(insumo["gasto_base"])
    nuevo = base * (1 + cambio)
    signo = "más" if cambio >= 0 else "menos"
    return pd.DataFrame(
        [
            _fila("Como hoy", _simular(insumo, base, s, cc_num), base, s),
            _fila(
                f"Gastando {abs(cambio):.0%} {signo}",
                _simular(insumo, nuevo, s, cc_num),
                nuevo,
                s,
            ),
        ]
    )


def meta_ahorro(insumo: pd.Series, monto: float, sup: dict, cc_num) -> pd.DataFrame:
    """¿En cuántos meses junta `monto` si sigue como hoy? Se mide en los
    1,000 futuros del simulador (36 meses)."""
    s = _supuestos(sup)
    sim = _simular(insumo, float(insumo["gasto_base"]), s, cc_num)
    alcanza = sim["ahorro"] >= monto
    mes = np.where(alcanza.any(axis=1), alcanza.argmax(axis=1) + 1, np.inf)

    def _mes(q):
        with np.errstate(invalid="ignore"):
            v = float(np.percentile(mes, q))
        return v if np.isfinite(v) else None

    return pd.DataFrame(
        [
            {
                "meta": round(float(monto), 2),
                "meses_tipico": _mes(50),
                "meses_caso_bueno": _mes(25),
                "meses_caso_malo": _mes(75),
                "prob_lograrlo_12m": round(float((mes <= 12).mean()), 4),
                "prob_lograrlo_36m": round(float(np.isfinite(mes).mean()), 4),
                "ahorro_mensual_tipico": round(
                    float(np.median(sim["flujo"][:, :12].mean(axis=1))), 2
                ),
            }
        ]
    )
