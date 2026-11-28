"""Pruebas de la evaluación masiva del asistente (sin Ollama)."""

import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.dashboard import datos_ejecutivo as de
from src.ia import evaluacion_masiva as em
from src.ia import evaluar as ev
from src.ml import simulador as sm
from src.ml.explicacion import nivel_riesgo

MESES = [pd.Period(f"2020-{m:02d}", "M") for m in range(7, 13)]
SUP = sm.Supuestos(simulaciones=40)


class BaseFalsa(ev.Base):
    """La misma interfaz que evaluar.Base, con 6 clientes inventados."""

    def __init__(self):
        rng = np.random.default_rng(3)
        n = 6
        prob = np.array([0.03, 0.08, 0.15, 0.22, 0.35, 0.6])
        ingreso = rng.uniform(4_000, 12_000, n).round(2)
        gasto = (ingreso * rng.uniform(0.4, 1.2, n)).round(2)
        self.clientes = pd.DataFrame(
            {
                "cc_num": np.arange(1, n + 1) * 11,
                "cliente": [f"Cliente {i:04d}" for i in range(1, n + 1)],
                "ingreso_mensual_simulado": ingreso,
                "gasto_promedio_mensual": gasto,
                "capacidad_ahorro": ((ingreso - gasto) / ingreso).round(4),
                "prob_impago": prob,
                "nivel": nivel_riesgo(pd.Series(prob)),
                "pago_mensual_prestamo": rng.uniform(300, 2_000, n).round(2),
                "ingreso_mensual_fuente": rng.uniform(3_000, 9_000, n).round(2),
                "ratio_endeudamiento": rng.uniform(0.1, 1.5, n).round(4),
                "monto_prestamo_fuente": rng.uniform(10_000, 200_000, n).round(2),
                "tasa_interes_anual_fuente": rng.uniform(5, 25, n).round(1),
                "plazo_meses_fuente": [12, 24, 36, 48, 60, 36],
                "score_credito_fuente": rng.integers(350, 820, n),
                "categoria_principal": "grocery_pos",
                "num_transacciones": 300,
                "primera_transaccion": pd.Timestamp("2020-06-21"),
            }
        )
        dias = pd.date_range("2020-06-21", "2020-12-31 20:00", freq="13h")
        self.tx = pd.DataFrame(
            {
                "cc_num": rng.choice(self.clientes["cc_num"], len(dias)),
                "amt": rng.uniform(5, 400, len(dias)).round(2),
                "category": rng.choice(sorted(de.TIPO_CONSUMO), len(dias)),
                "trans_date_trans_time": dias,
            }
        )
        self.meses = MESES
        self.aportes = None
        filas, insumos = [], []
        for f in self.clientes.itertuples():
            insumos.append(
                {
                    "cc_num": f.cc_num,
                    "ingreso": f.ingreso_mensual_simulado,
                    "gasto_base": f.gasto_promedio_mensual,
                    "cv_gasto": 0.2,
                }
            )
            for esc in sm.ESCENARIOS:
                sim = sm.simular_cliente(
                    f.ingreso_mensual_simulado,
                    f.gasto_promedio_mensual,
                    0.2,
                    esc,
                    SUP,
                    semilla=sm.semilla_cliente(f.cc_num, esc),
                )
                filas.append(
                    {
                        "cc_num": f.cc_num,
                        "escenario": esc,
                        **sm.resumir(sim, f.gasto_promedio_mensual, SUP),
                    }
                )
        self.sim = (
            pd.DataFrame(insumos),
            pd.DataFrame(filas),
            {"supuestos": {"simulaciones": 40}},
        )

    def asistente(self, modelos, fila, tablas):
        return SimpleNamespace(cliente=fila["cliente"])


@pytest.fixture(scope="module")
def base():
    return BaseFalsa()


def _modelos():
    return {k: FakeListChatModel(responses=["{}"]) for k in ("json", "sql", "texto")}


def _clave(plan):
    clientes, (_, port) = plan
    return [(p.id, p.pregunta, p.antes) for *_, ps in clientes for p in ps] + [
        (p.id, p.pregunta, p.antes) for p in port
    ]


def test_el_plan_es_reproducible_y_del_tamano_pedido(base):
    a = _clave(em.generar_plan(base, 200, 7))
    b = _clave(em.generar_plan(base, 200, 7))
    c = _clave(em.generar_plan(base, 200, 8))
    assert a == b
    assert a != c
    ids = sorted(i for i, *_ in a)
    assert ids == list(range(1, 201))
    _, (_, port) = em.generar_plan(base, 200, 7)
    assert len(port) == round(200 * em.PROPORCION_PORTAFOLIO)


def test_hay_redacciones_informales_y_de_seguimiento(base):
    preguntas = [q for _, q, _ in _clave(em.generar_plan(base, 400, 1))]
    antes = [a for *_, a in _clave(em.generar_plan(base, 400, 1)) if a]
    assert any(q == q.lower() and "¿" not in q for q in preguntas)
    assert any(q.startswith("¿") for q in preguntas)
    assert antes, "debe haber preguntas de seguimiento"


def _prep(esperados):
    return SimpleNamespace(
        tipo=sorted(esperados)[0], datos=None, respuesta_directa=False, sql=None
    )


SIN_CIFRA_OBLIGATORIA = {
    "subir o bajar el gasto",  # se revisa con lo que calculó el simulador
    "meta de ahorro",
    "escenario de vida",  # si el escenario casi no cambia, no se exige
    "fuera de tema",
}


@pytest.mark.parametrize(
    "ambito,plantilla",
    [("cliente", p) for p in em.PLANTILLAS_CLIENTE]
    + [("portafolio", p) for p in em.PLANTILLAS_PORTAFOLIO],
    ids=lambda x: x if isinstance(x, str) else x[0],
)
def test_cada_plantilla_genera_preguntas_que_se_pueden_calificar(
    base, ambito, plantilla
):
    nombre, peso, generar = plantilla
    assert peso > 0
    fila = base.con_simulacion().iloc[2]
    tablas = base.tablas(fila) if ambito == "cliente" else base.tablas_portafolio()
    for s in range(15):
        spec = generar(tablas, random.Random(s))
        if spec is None:
            continue
        pregunta, esperados, revisar = spec[:3]
        assert pregunta and esperados
        problemas = revisar("", _prep(esperados), tablas)
        if nombre not in SIN_CIFRA_OBLIGATORIA:
            assert problemas, f"{nombre}: una respuesta vacía no debería pasar"


def _stub(ok_cada=10, cortar_en=None):
    llamadas = []

    def evaluar(asistente, tablas, pregunta, esperados, revisar, antes=None):
        llamadas.append(pregunta)
        if cortar_en and len(llamadas) == cortar_en:
            raise KeyboardInterrupt
        return {
            "tipo": sorted(esperados)[0],
            "ok": len(llamadas) % ok_cada != 0,
            "problemas": "" if len(llamadas) % ok_cada else "no dice el gasto (100)",
            "segundos": 1.5,
            "respuesta": "respuesta",
            "sql": "",
        }

    return evaluar, llamadas


def test_guarda_cada_respuesta_y_retoma_donde_se_quedo(base, tmp_path):
    archivo = tmp_path / "res.csv"
    evaluar, llamadas = _stub(cortar_en=31)
    with pytest.raises(KeyboardInterrupt):
        em.correr(base, _modelos(), 80, 5, archivo, evaluar)
    assert len(pd.read_csv(archivo)) == 30

    evaluar, llamadas = _stub()
    em.correr(base, _modelos(), 80, 5, archivo, evaluar)
    res = pd.read_csv(archivo)
    assert len(llamadas) == 50
    assert sorted(res["id"]) == list(range(1, 81))

    evaluar, llamadas = _stub()
    em.correr(base, _modelos(), 80, 5, archivo, evaluar)
    assert llamadas == []  # ya estaba completa


def test_resumen_por_plantilla_y_problemas_comunes():
    res = pd.DataFrame(
        {
            "ambito": ["cliente"] * 4 + ["portafolio"] * 2,
            "cliente": ["A", "A", "B", "C", "Portafolio", "Portafolio"],
            "plantilla": ["riesgo", "riesgo", "meta", "meta", "total", "total"],
            "ok": [True, False, True, True, True, False],
            "problemas": [
                "",
                "no dice el nivel de riesgo (Alto)",
                "",
                "",
                "",
                "cifras que no están en los datos: 12.5",
            ],
            "segundos": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    general, por = em.resumen(res)
    assert general["preguntas"] == 6 and general["correctas"] == 4
    assert general["clientes_distintos"] == 3
    assert general["por_ambito"]["portafolio"]["porcentaje"] == 0.5
    assert general["problemas_mas_comunes"] == {
        "no dice el nivel de riesgo": 1,
        "cifras inventadas": 1,
    }
    assert por.iloc[0]["porcentaje"] == 0.5  # lo peor primero
