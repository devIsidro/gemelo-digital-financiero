"""Pruebas del asistente de IA (sin Ollama: se usan modelos falsos de LangChain)."""

import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.ia import asistente as ia
from src.ia import contexto as cx
from src.ia import hechos as hc
from src.ml import simulador as sm

CC = 7


def _fila():
    return pd.Series(
        {
            "cliente": "Cliente 0007",
            "ingreso_mensual_simulado": 10_000.0,
            "gasto_promedio_mensual": 4_000.0,
            "capacidad_ahorro": 0.6,
            "prob_impago": 0.12,
            "nivel": "Medio",
            "pago_mensual_prestamo": 900.0,
            "ingreso_mensual_fuente": 6_000.0,
            "ratio_endeudamiento": 0.15,
            "monto_prestamo_fuente": 50_000.0,
            "tasa_interes_anual_fuente": 12.5,
            "plazo_meses_fuente": 60,
            "score_credito_fuente": 700,
            "categoria_principal": "food_dining",
            "num_transacciones": 4,
            "primera_transaccion": pd.Timestamp("2020-06-30"),
        }
    )


def _tx():
    return pd.DataFrame(
        {
            "cc_num": [CC] * 4,
            "amt": [99.0, 3_000.0, 1_000.0, 5_000.0],
            "category": ["travel", "grocery_pos", "food_dining", "travel"],
            "trans_date_trans_time": pd.to_datetime(
                ["2020-06-30", "2020-07-10", "2020-07-20", "2020-08-05"]
            ),
        }
    )


MESES = [pd.Period("2020-07", "M"), pd.Period("2020-08", "M")]
INSUMO = pd.Series(
    {"cc_num": CC, "ingreso": 10_000.0, "gasto_base": 4_000.0, "cv_gasto": 0.2}
)
SUP = {"simulaciones": 200}


def _resumen():
    filas = []
    s = sm.Supuestos(simulaciones=200)
    for esc in sm.ESCENARIOS:
        sim = sm.simular_cliente(
            10_000, 4_000, 0.2, esc, s, semilla=sm.semilla_cliente(CC, esc)
        )
        filas.append({"cc_num": CC, "escenario": esc, **sm.resumir(sim, 4_000, s)})
    return pd.DataFrame(filas)


@pytest.fixture(scope="module")
def tablas():
    return cx.tablas_cliente(_fila(), _tx(), MESES, _resumen(), None)


def _modelos(json_r=(), sql_r=(), texto_r=("Respuesta.",)):
    return {
        "json": FakeListChatModel(responses=list(json_r) or ["{}"]),
        "sql": FakeListChatModel(responses=list(sql_r) or ["SELECT 1"]),
        "texto": FakeListChatModel(responses=list(texto_r)),
    }


# ---------------------------------------------------------------------------
# Tablas del cliente
# ---------------------------------------------------------------------------


def test_tablas_del_cliente(tablas):
    assert set(tablas) == set(cx.TABLAS)
    perfil = tablas["mi_perfil"].iloc[0]
    assert perfil["ahorro_mensual"] == 6_000
    assert perfil["categoria_principal"] == "Restaurantes"
    meses = tablas["mis_meses"]
    # Junio está incompleto y no entra; julio gastó 4,000 y agosto 5,000.
    assert list(meses["mes_nombre"]) == ["jul 2020", "ago 2020"]
    assert list(meses["gasto"]) == [4_000, 5_000]
    assert list(meses["capacidad_ahorro"]) == [0.6, 0.5]
    cat = tablas["mis_gastos_categoria"]
    assert set(cat["tipo"]) == {"Esencial", "Discrecional"}
    assert list(tablas["mis_escenarios"]["escenario"])[0] == "Sin cambios"
    assert tablas["mis_factores_riesgo"].empty


def test_el_esquema_describe_todas_las_tablas():
    texto = cx.describir_esquema()
    for tabla in cx.TABLAS:
        assert tabla in texto


# ---------------------------------------------------------------------------
# SQL seguro
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE mi_perfil",
        "SELECT 1; SELECT 2",
        "SELECT * FROM kpis_cliente",
        "SELECT * FROM read_csv('data/raw/fraudTest.csv')",
        "COPY mi_perfil TO 'x.csv'",
        "",
    ],
)
def test_validar_sql_bloquea_lo_peligroso(sql):
    with pytest.raises(cx.SQLInvalido):
        cx.validar_sql(sql)


def test_validar_sql_acepta_consultas_normales():
    cx.validar_sql("SELECT EXTRACT(MONTH FROM mes) AS m FROM mis_meses")
    cx.validar_sql(
        "WITH u AS (SELECT mes FROM mis_meses) SELECT * FROM mis_gastos_categoria "
        "WHERE mes IN (SELECT mes FROM u)"
    )


def test_limpiar_sql_quita_texto_y_formato():
    assert cx.limpiar_sql("Claro:\n```sql\nSELECT 1 FROM mi_perfil;\n```") == (
        "SELECT 1 FROM mi_perfil"
    )


def test_duckdb_no_puede_leer_archivos(tablas, tmp_path):
    archivo = tmp_path / "secreto.csv"
    archivo.write_text("a\n1\n")
    con = cx.conectar(tablas)
    with pytest.raises(Exception):
        con.execute(f"SELECT * FROM read_csv('{archivo}')").fetchall()
    assert len(cx.ejecutar_sql(con, "SELECT * FROM mis_meses")) == 2


# ---------------------------------------------------------------------------
# Flujo del asistente
# ---------------------------------------------------------------------------


def test_datos_corrige_el_sql_una_vez(tablas):
    modelos = _modelos(
        ['{"tipo": "datos"}'],
        [
            "SELECT * FROM tabla_inventada",
            "```sql\nSELECT mes_nombre, gasto FROM mis_meses\n```",
        ],
        ["En agosto gastaste más."],
    )
    texto, prep = ia.Asistente(modelos, tablas, INSUMO, SUP, CC).responder(
        "¿Cuánto gasté?"
    )
    assert texto == "En agosto gastaste más."
    assert prep.tipo == "datos" and len(prep.intentos) == 2
    assert list(prep.datos["gasto"]) == [4_000, 5_000]


def test_si_el_sql_nunca_funciona_usa_el_resumen_mensual(tablas):
    modelos = _modelos(['{"tipo": "datos"}'], ["DROP TABLE x", "DELETE FROM mis_meses"])
    prep = ia.Asistente(modelos, tablas, INSUMO, SUP, CC).preparar("¿?")
    assert prep.sql is None and prep.nota
    assert prep.datos.equals(tablas["mis_meses"])


def test_gasto_usa_el_simulador_y_corrige_porcentajes(tablas):
    modelos = _modelos(['{"tipo": "gasto", "cambio_gasto": 20}'])
    prep = ia.Asistente(modelos, tablas, INSUMO, SUP, CC).preparar(
        "¿Y si gasto 20% más?"
    )
    assert prep.tipo == "gasto"
    assert list(prep.datos["situacion"]) == ["Como hoy", "Gastando 20% más"]
    assert prep.datos["gasto_mensual"].iloc[1] == pytest.approx(4_800)
    assert prep.datos["ahorro_12m"].iloc[1] < prep.datos["ahorro_12m"].iloc[0]


def test_meta_sin_monto_usa_el_fondo_de_emergencia(tablas):
    modelos = _modelos(['{"tipo": "meta", "meta": null}'])
    prep = ia.Asistente(modelos, tablas, INSUMO, SUP, CC).preparar(
        "¿Cuándo llego a mi meta?"
    )
    meta = tablas["mis_escenarios"]["meta_fondo_emergencia"].iloc[0]
    assert prep.datos["meta"].iloc[0] == pytest.approx(meta)
    assert 0 <= prep.datos["prob_lograrlo_36m"].iloc[0] <= 1


def test_json_invalido_se_trata_como_consulta_de_datos(tablas):
    modelos = _modelos(["no sé"], ["SELECT nivel_riesgo FROM mi_perfil"])
    prep = ia.Asistente(modelos, tablas, INSUMO, SUP, CC).preparar("¿Mi riesgo?")
    assert prep.tipo == "datos" and prep.datos.iloc[0, 0] == "Medio"


def test_sin_simulacion_las_preguntas_de_gasto_van_a_datos(tablas):
    modelos = _modelos(['{"tipo": "gasto"}'], ["SELECT gasto FROM mis_meses"])
    prep = ia.Asistente(modelos, tablas, None, SUP, CC).preparar("¿Y si gasto más?")
    assert prep.tipo == "datos"


def test_saludo_no_consulta_datos(tablas):
    modelos = _modelos(['{"tipo": "otro"}'], texto_r=["NO DEBE USARSE"])
    texto, prep = ia.Asistente(modelos, tablas, INSUMO, SUP, CC).responder("Hola")
    assert prep.tipo == "otro" and texto == ia.MENSAJE_OTRO


def test_formatear_escribe_porcentajes_y_dolares():
    df = pd.DataFrame(
        {"capacidad_ahorro": [0.601], "gasto": [-1234.5], "num_compras": [3]}
    )
    assert ia.formatear(df).splitlines()[1] == "60.1%|-$1,234|3"


def test_ollama_apagado_se_detecta():
    assert ia.ollama_disponible("http://127.0.0.1:9", timeout=0.5) == (
        False,
        "sin_servicio",
    )


def test_los_prompts_existen():
    assert "{esquema}" in ia.SISTEMA_SQL
    assert "{pregunta}" in ia.PLANTILLA_RESPUESTA


def test_al_modelo_le_llegan_etiquetas_claras(tablas):
    texto = ia.formatear_para_modelo(tablas["mis_meses"])
    assert "Mes: jul 2020" in texto
    assert "Capacidad de ahorro (parte del ingreso que ahorró): 60.0%" in texto
    assert "Fecha del mes" not in texto  # la fecha técnica no hace falta


def test_las_respuestas_de_datos_incluyen_el_historial_mensual(tablas):
    modelos = _modelos(['{"tipo": "datos"}'], ["SELECT nivel_riesgo FROM mi_perfil"])
    asistente = ia.Asistente(modelos, tablas, INSUMO, SUP, CC)
    entradas = asistente.entradas_respuesta(asistente.preparar("¿Mi riesgo?"))
    assert "ago 2020" in entradas["contexto_extra"]
    assert "historial real" in entradas["origen"]


def test_hechos_comparan_el_ultimo_mes_contra_el_anterior(tablas):
    hechos = hc.hechos_meses(tablas)
    # jul: gasto 4,000 (60%); ago: gasto 5,000 (50%) -> la capacidad bajó.
    assert any("tu capacidad de ahorro bajó: de 60.0% a 50.0%" in h for h in hechos)
    assert any("Viajes: de $0 a $5,000" in h for h in hechos)


def test_hechos_del_gasto_aclaran_que_no_es_impago(tablas):
    modelos = _modelos(['{"tipo": "gasto", "cambio_gasto": 0.2}'])
    prep = ia.Asistente(modelos, tablas, INSUMO, SUP, CC).preparar(
        "¿Y si gasto 20% más?"
    )
    texto = " ".join(prep.hechos)
    assert "NO es la probabilidad de dejar de pagar" in texto
    assert "$4,000 a $4,800" in texto
    assert ", o sea " in texto


def test_hechos_de_riesgo_solo_si_preguntan_por_riesgo(tablas):
    modelos = _modelos(['{"tipo": "datos"}'], ["SELECT nivel_riesgo FROM mi_perfil"])
    prep = ia.Asistente(modelos, tablas, INSUMO, SUP, CC).preparar(
        "¿Cuál es mi nivel de riesgo?"
    )
    assert prep.hechos[0].startswith("Tu probabilidad de no pagar tu préstamo es 12.0%")


def test_avisa_cuando_la_pregunta_supone_algo_falso(tablas):
    # En los datos de prueba la capacidad de ahorro bajó (60% -> 50%).
    assert (
        hc.revisar_premisa("¿Por qué disminuyó mi capacidad de ahorro?", tablas) == ""
    )
    aviso = hc.revisar_premisa(
        "¿Por qué aumentó mi capacidad de ahorro el mes pasado?", tablas
    )
    assert aviso and (
        "En realidad, tu capacidad de ahorro bajó el último mes: de 60.0% en jul 2020 "
        "a 50.0% en ago 2020, porque tu gasto subió de $4,000 a $5,000." in aviso
    )
    assert "no hay un mes en que tu capacidad de ahorro subió" in aviso
    assert hc.revisar_premisa("¿En qué gasto más?", tablas) == ""


def test_premisa_falsa_explica_el_mes_en_que_si_bajo():
    meses = pd.DataFrame(
        {
            "mes": ["2020-10", "2020-11", "2020-12"],
            "mes_nombre": ["oct 2020", "nov 2020", "dic 2020"],
            "ingreso": [10000.0] * 3,
            "gasto": [2000.0, 9900.0, 4500.0],
            "capacidad_ahorro": [0.8, 0.01, 0.55],
        }
    )
    cat = pd.DataFrame(
        {
            "mes": ["2020-10", "2020-10", "2020-11", "2020-11"],
            "categoria": ["Compras", "Comida", "Compras", "Comida"],
            "gasto": [500.0, 1500.0, 8000.0, 1900.0],
        }
    )
    aviso = hc.revisar_premisa(
        "¿Por qué disminuyó mi capacidad de ahorro el mes pasado?",
        {"mis_meses": meses, "mis_gastos_categoria": cat},
    )
    assert "En realidad, tu capacidad de ahorro subió el último mes" in aviso
    assert "Donde sí bajó fue en nov 2020: de 80.0% en oct 2020 a 1.0%" in aviso
    assert "porque tu gasto subió de $2,000 a $9,900 con el mismo ingreso" in aviso
    assert "Lo que más subió fue Compras: de $500 a $8,000." in aviso


def test_limpia_muletillas_de_fuente():
    assert (
        ia.limpiar_respuesta("Según los datos, si sigues como hoy la alcanzas.")
        == "Si sigues como hoy la alcanzas."
    )
    assert (
        ia.limpiar_respuesta("Tu riesgo es Medio. De acuerdo con la información, baja.")
        == "Tu riesgo es Medio. Baja."
    )
    assert (
        ia.limpiar_respuesta("Tu riesgo, según los datos, es Medio según los datos.")
        == "Tu riesgo es Medio."
    )
    sin_cambio = "Según la simulación, tu ahorro bajaría."
    assert ia.limpiar_respuesta(sin_cambio) == sin_cambio


def test_quita_etiquetas_y_consejos_que_nadie_pidio():
    texto = (
        "ATENCIÓN: tu ahorro pasaría de 7.6% a 13.2%. Recomendación: revisa tus "
        "gastos y considera ajustarlos. Tu meta llega en 7 meses. Además, te "
        "recomiendo mantener un equilibrio saludable."
    )
    assert (
        ia.limpiar_respuesta(texto)
        == "Tu ahorro pasaría de 7.6% a 13.2%. Tu meta llega en 7 meses."
    )
    # Si el cliente pide consejo, la recomendación se queda.
    assert "Recomendación: revisa" in ia.limpiar_respuesta(texto, consejos=True)
    assert ia.pide_consejo("¿Qué me recomiendas para ahorrar más?")
    assert ia.pide_consejo("¿Cómo puedo juntar $50,000 más rápido?")
    assert not ia.pide_consejo("¿Cuándo podría juntar $50,000?")


def test_limpieza_funciona_aunque_la_respuesta_llegue_en_partes():
    texto = (
        "Según los datos, si sigues como hoy la alcanzas en 7 meses. Recomiendo "
        "que revises tus gastos y ajustes tu presupuesto del 12.5% para asegurarte. "
        "Tu riesgo pasaría de 7.6% a 13.2%.\n\nATENCIÓN: tu gasto bajó. "
    ) * 2
    partes = list(texto)
    unido = "".join(ia.limpiar_stream(partes))
    assert unido == ia.limpiar_respuesta(texto)
    for frase in ("Según los datos", "Recomiendo", "ATENCIÓN"):
        assert frase not in unido


def test_premisa_falsa_se_responde_sin_pasar_por_el_modelo(tablas):
    modelos = _modelos(
        ['{"tipo": "datos"}'],
        ["SELECT mes_nombre, capacidad_ahorro FROM mis_meses"],
        ["Tu gasto subió el mes pasado."],
    )
    texto, prep = ia.Asistente(modelos, tablas, INSUMO, SUP, CC).responder(
        "¿Por qué aumentó mi capacidad de ahorro el mes pasado?"
    )
    assert texto == prep.respuesta_directa
    assert texto.startswith("En realidad, tu capacidad de ahorro bajó el último mes")
    assert "porque tu gasto subió de" in texto
    assert "Tu gasto subió el mes pasado." not in texto


def test_factores_de_riesgo_en_segunda_persona():
    casos = {
        "Su pago mensual es 2.2 veces su ingreso": "tu pago mensual es 2.2 veces tu ingreso",
        "Tasa de interés de 18.0%": "tu tasa de interés es de 18.0%",
        "Lleva 119 meses en su empleo": "llevas 119 meses en tu empleo",
        "Tiene 34 años": "tienes 34 años",
        "Sin aval": "no tienes aval",
        "Ingreso de 12,493 USD al mes": "tu ingreso es de $12,493 al mes",
        "Score de crédito de 322": "tu score de crédito es de 322",
        "3 líneas de crédito": "tienes 3 líneas de crédito",
        "Está desempleado": "estás desempleado",
    }
    for factor, esperado in casos.items():
        assert hc._en_tu_voz(factor) == esperado


def test_respuesta_de_riesgo_siempre_dice_las_causas(tablas):
    tablas = dict(tablas)
    tablas["mis_factores_riesgo"] = pd.DataFrame(
        {
            "factor": ["Tasa de interés de 18.0%", "Plazo de 12 meses"],
            "efecto": ["Sube el riesgo", "Baja el riesgo"],
        }
    )

    def responder(texto_modelo):
        modelos = _modelos(
            ['{"tipo": "datos"}'],
            ["SELECT nivel_riesgo, prob_impago FROM mi_perfil"],
            [texto_modelo],
        )
        return ia.Asistente(modelos, tablas, INSUMO, SUP, CC).responder(
            "¿Cuál es mi nivel de riesgo?"
        )

    texto, prep = responder("Tu nivel de riesgo es Medio.")
    claves, frase = prep.obligatorio
    assert frase.startswith("Lo que más sube tu riesgo es que ")
    assert texto == "Tu nivel de riesgo es Medio. " + frase
    # Si el modelo ya las mencionó, no se repiten.
    assert frase == "Lo que más sube tu riesgo es que tu tasa de interés es de 18.0%."
    con_causa = "Tu nivel de riesgo es Medio por tu tasa de 18%."
    assert responder(con_causa)[0] == con_causa
