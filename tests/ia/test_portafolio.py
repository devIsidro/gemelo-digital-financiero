"""Pruebas de las preguntas del portafolio (sin Ollama: modelos falsos)."""

import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.ia import contexto as cx
from src.ia import portafolio as pf

MESES = [pd.Period("2020-07", "M"), pd.Period("2020-08", "M")]


def _clientes():
    return pd.DataFrame(
        {
            "cc_num": [1, 2, 3, 4],
            "cliente": ["Cliente 0001", "Cliente 0002", "Cliente 0003", "Cliente 0004"],
            "ingreso_mensual_simulado": [10_000.0, 8_000.0, 5_000.0, 3_000.0],
            "gasto_promedio_mensual": [4_000.0, 6_000.0, 5_500.0, 2_000.0],
            "capacidad_ahorro": [0.6, 0.25, -0.1, 0.3333],
            "prob_impago": [0.05, 0.12, 0.45, 0.08],
            "nivel": pd.Categorical(
                ["Bajo", "Medio", "Alto", "Bajo"], categories=["Bajo", "Medio", "Alto"]
            ),
            "ratio_endeudamiento": [0.1, 0.3, 1.2, 0.2],
            "score_credito_fuente": [800, 650, 400, 720],
            "monto_prestamo_fuente": [50_000.0, 30_000.0, 120_000.0, 10_000.0],
            "tasa_interes_anual_fuente": [8.0, 12.5, 23.0, 9.0],
            "categoria_principal": ["travel", "grocery_pos", "food_dining", "travel"],
            "num_transacciones": [2, 1, 1, 1],
        }
    )


def _tx():
    return pd.DataFrame(
        {
            "cc_num": [1, 1, 2, 3, 4, 9],
            "amt": [1_000.0, 500.0, 300.0, 200.0, 700.0, 99_999.0],
            "category": [
                "travel",
                "food_dining",
                "grocery_pos",
                "food_dining",
                "travel",
                "travel",
            ],
            "trans_date_trans_time": pd.to_datetime(
                [
                    "2020-07-10",
                    "2020-08-02",
                    "2020-08-15",
                    "2020-08-20",
                    "2020-07-30",
                    "2020-08-01",  # cliente 9: no está en el portafolio
                ]
            ),
        }
    )


def _resumen():
    filas = []
    for cc, deficits in ((1, 0.0), (2, 0.05), (3, 0.6), (4, 0.2)):
        for esc in ("base", "credito_hipotecario"):
            extra = 0.3 if esc == "credito_hipotecario" else 0.0
            filas.append(
                {
                    "cc_num": cc,
                    "escenario": esc,
                    "prob_deficit": min(deficits + extra, 1.0),
                    "ahorro_m12_p50": 1_000.0 * cc,
                    "ahorro_m36_p50": 3_000.0 * cc,
                }
            )
    return pd.DataFrame(filas)


@pytest.fixture(scope="module")
def tablas():
    return pf.tablas_portafolio(_clientes(), _tx(), MESES, _resumen())


def _modelos(json_r=(), sql_r=(), texto_r=("Respuesta.",)):
    return {
        "json": FakeListChatModel(responses=list(json_r) or ['{"tipo": "datos"}']),
        "sql": FakeListChatModel(responses=list(sql_r) or ["SELECT 1"]),
        "texto": FakeListChatModel(responses=list(texto_r)),
    }


# ---------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------


def test_tablas_del_portafolio(tablas):
    assert set(tablas) == set(pf.TABLAS)
    c = tablas["clientes"]
    assert len(c) == 4
    assert list(c["ahorro_mensual"]) == [6_000, 2_000, -500, 1_000]
    assert c.loc[0, "categoria_principal"] == "Viajes"
    assert list(c["prob_numeros_rojos"]) == [0.0, 0.05, 0.6, 0.2]

    n = tablas["resumen_niveles"].set_index("nivel_riesgo")
    assert list(n.index) == ["Bajo", "Medio", "Alto"]
    assert list(n["num_clientes"]) == [2, 1, 1]
    assert n.loc["Bajo", "porcentaje_clientes"] == 0.5
    assert n.loc["Bajo", "monto_prestamo_total"] == 60_000

    m = tablas["meses_portafolio"]
    assert list(m["mes_nombre"]) == ["jul 2020", "ago 2020"]
    # El cliente 9 no es del portafolio: su compra no cuenta.
    assert list(m["gasto_total"]) == [1_700, 1_000]
    assert list(m["num_clientes_con_compras"]) == [2, 3]

    cat = tablas["gastos_categoria_portafolio"]
    viajes_julio = cat[
        (cat["mes_nombre"] == "jul 2020") & (cat["categoria"] == "Viajes")
    ]
    assert viajes_julio["gasto_total"].iloc[0] == 1_700
    assert viajes_julio["num_clientes"].iloc[0] == 2


def test_escenarios_del_portafolio(tablas):
    e = tablas["escenarios_portafolio"].set_index("escenario")
    # En riesgo: al menos 10% de probabilidad de números rojos.
    assert e.loc["Sin cambios", "num_clientes_en_riesgo"] == 2
    assert e.loc["Crédito hipotecario", "num_clientes_en_riesgo"] == 4
    assert e.loc["Crédito hipotecario", "porcentaje_en_riesgo"] == 1.0
    assert e.loc["Crédito hipotecario", "ahorro_12m_mediano"] == 2_500


def test_sin_simulacion_las_tablas_siguen_funcionando():
    t = pf.tablas_portafolio(_clientes(), _tx(), MESES, None)
    assert t["escenarios_portafolio"].empty
    assert t["clientes"]["prob_numeros_rojos"].isna().all()


def test_el_esquema_describe_todas_las_tablas():
    texto = pf.describir_esquema()
    for tabla in pf.TABLAS:
        assert tabla in texto


def test_panorama_calculado_en_python(tablas):
    texto = " ".join(pf.panorama(tablas))
    assert "4 clientes" in texto
    assert "2 en riesgo bajo (50.0%)" in texto
    assert "1 en riesgo alto (25.0%)" in texto
    assert "1 clientes (25.0%) gastan más de lo que ganan" in texto
    assert "jul 2020" in texto and "$1,700" in texto


# ---------------------------------------------------------------------------
# SQL seguro
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE clientes",
        "SELECT * FROM mi_perfil",  # tablas de un solo cliente: aquí no existen
        "SELECT * FROM read_parquet('data/gold/kpis_cliente')",
    ],
)
def test_sql_prohibido(sql):
    with pytest.raises(cx.SQLInvalido):
        cx.validar_sql(sql, pf.TABLAS)


def test_el_validador_del_cliente_no_cambia():
    with pytest.raises(cx.SQLInvalido):
        cx.validar_sql("SELECT * FROM clientes")
    assert cx.validar_sql("SELECT * FROM mi_perfil") == "SELECT * FROM mi_perfil"


# ---------------------------------------------------------------------------
# Asistente del portafolio
# ---------------------------------------------------------------------------


def test_cuenta_clientes_con_sql(tablas):
    sql = (
        "SELECT nivel_riesgo, num_clientes, porcentaje_clientes "
        "FROM resumen_niveles WHERE nivel_riesgo = 'Alto'"
    )
    a = pf.AsistentePortafolio(
        _modelos(sql_r=[sql], texto_r=["Hay 1 cliente."]), tablas
    )
    texto, prep = a.responder("¿Cuántos clientes están en riesgo alto?")
    assert texto == "Hay 1 cliente."
    assert prep.tipo == "datos" and prep.sql == sql
    assert prep.datos.iloc[0]["num_clientes"] == 1
    # Un solo renglón: se le marca al modelo como la respuesta exacta.
    assert "Respuesta exacta" in prep.hechos[0]
    assert "25.0%" in prep.hechos[0]


def test_lista_de_clientes(tablas):
    sql = "SELECT cliente, prob_impago FROM clientes ORDER BY prob_impago DESC LIMIT 2"
    a = pf.AsistentePortafolio(_modelos(sql_r=[sql]), tablas)
    prep = a.preparar("¿Quiénes tienen más riesgo?")
    assert list(prep.datos["cliente"]) == ["Cliente 0003", "Cliente 0002"]
    entradas = a.entradas_respuesta(prep)
    assert "Probabilidad de impago: 45.0%" in entradas["datos"]
    assert "El portafolio tiene 4 clientes" in entradas["panorama"]


def test_corrige_el_sql_una_vez(tablas):
    malo = "SELECT * FROM mi_perfil"
    bueno = "SELECT COUNT(*) AS num_clientes FROM clientes WHERE ahorro_mensual < 0"
    a = pf.AsistentePortafolio(_modelos(sql_r=[malo, bueno]), tablas)
    prep = a.preparar("¿Cuántos gastan más de lo que ganan?")
    assert prep.intentos == [malo, bueno]
    assert prep.datos.iloc[0]["num_clientes"] == 1


def test_sin_sql_valido_usa_el_resumen(tablas):
    a = pf.AsistentePortafolio(_modelos(sql_r=["DROP TABLE clientes"] * 2), tablas)
    prep = a.preparar("Algo raro")
    assert prep.datos is tablas["resumen_niveles"]
    assert prep.nota


def test_otro_contesta_con_el_mensaje_de_ayuda(tablas):
    a = pf.AsistentePortafolio(_modelos(json_r=['{"tipo": "otro"}']), tablas)
    texto, prep = a.responder("Hola")
    assert prep.tipo == "otro" and texto == pf.MENSAJE_OTRO


def test_json_raro_se_toma_como_datos(tablas):
    a = pf.AsistentePortafolio(_modelos(json_r=["no sé"]), tablas)
    assert a.clasificar("¿Cuántos clientes hay?") == "datos"


def test_quita_muletillas(tablas):
    a = pf.AsistentePortafolio(
        _modelos(texto_r=["Según los datos, hay 4 clientes."]), tablas
    )
    texto, _ = a.responder("¿Cuántos clientes hay?")
    assert texto == "Hay 4 clientes."
