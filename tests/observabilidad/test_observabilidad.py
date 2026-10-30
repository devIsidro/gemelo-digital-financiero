"""Pruebas de la Fase 6: consistencia entre capas, SLA y alertas."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src.dashboard import datos
from src.observabilidad import alertas as al
from src.observabilidad import slas
from src.observabilidad.consistencia import Chequeo, revisar_consistencia
from src.observabilidad.revisar import revisar

AHORA = pd.Timestamp("2026-10-30 12:00", tz="America/Mexico_City")


# ---------------------------------------------------------------------------
# Un Lakehouse chiquito: 2 clientes, 4 compras, 3 préstamos
# ---------------------------------------------------------------------------


def _parquet(ruta, df):
    ruta.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ruta / "part-0.parquet", index=False)


@pytest.fixture
def lakehouse(tmp_path):
    r = datos.Rutas(tmp_path)
    _parquet(r.bronze_tx, pd.DataFrame({"trans_num": ["a", "b", "c", "d", "e"]}))
    _parquet(
        r.silver_tx,
        pd.DataFrame(
            {
                "trans_num": ["a", "b", "c", "d"],
                "cc_num": [1, 1, 2, 2],
                "amt": [10.0, 20.0, 5.5, 4.5],
            }
        ),
    )
    _parquet(
        r.gold_perfil,
        pd.DataFrame(
            {"cc_num": [1, 2], "gasto_total": [30.0, 10.0], "num_transacciones": [2, 2]}
        ),
    )
    _parquet(r.gold_kpis, pd.DataFrame({"cc_num": [1, 2], "loan_id": ["L1", "L2"]}))
    _parquet(r.gold_prob, pd.DataFrame({"cc_num": [1, 2], "prob_impago": [0.1, 0.4]}))
    _parquet(r.bronze_loans, pd.DataFrame({"LoanID": ["L1", "L2", "L3"]}))
    _parquet(r.silver_loans, pd.DataFrame({"loan_id": ["L1", "L2", "L3"]}))
    r.simulacion_insumos.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"cc_num": [1, 2]}).to_parquet(r.simulacion_insumos)
    pd.DataFrame(
        {"cc_num": [1, 1, 2, 2], "escenario": ["base", "negocio", "base", "negocio"]}
    ).to_parquet(r.simulacion_resumen)
    return r


def _por_nombre(chequeos):
    return {c.nombre: c for c in chequeos}


# ---------------------------------------------------------------------------
# Consistencia entre capas
# ---------------------------------------------------------------------------


def test_lakehouse_consistente(lakehouse, monkeypatch):
    monkeypatch.setattr("src.observabilidad.consistencia.MAX_DESCARTE", 0.5)
    chequeos = revisar_consistencia(lakehouse)
    assert all(c.ok for c in chequeos), [c for c in chequeos if not c.ok]
    gasto = _por_nombre(chequeos)["Gasto: Silver → Gold"]
    assert "$40.00 en Silver y $40.00 en Gold" in gasto.detalle


def test_detecta_descarte_excesivo_y_gasto_que_no_cuadra(lakehouse):
    # 1 de 5 transacciones descartada = 20% (> 1%)
    _parquet(
        lakehouse.gold_perfil,
        pd.DataFrame(
            {"cc_num": [1, 2], "gasto_total": [30.0, 99.0], "num_transacciones": [2, 2]}
        ),
    )
    c = _por_nombre(revisar_consistencia(lakehouse))
    assert c["Transacciones: Bronze → Silver"].ok is False
    assert "Se descartó el 20.0%" in c["Transacciones: Bronze → Silver"].detalle
    assert c["Gasto: Silver → Gold"].ok is False
    assert "1 clientes con gasto distinto" in c["Gasto: Silver → Gold"].detalle


def test_detecta_clientes_faltantes_y_probabilidad_fuera_de_rango(lakehouse):
    _parquet(lakehouse.gold_prob, pd.DataFrame({"cc_num": [1], "prob_impago": [1.4]}))
    c = _por_nombre(revisar_consistencia(lakehouse))
    assert c["Clientes: Silver → Gold"].ok is False
    assert "probabilidad de impago: faltan 1" in c["Clientes: Silver → Gold"].detalle
    assert c["Probabilidad de impago"].ok is False


def test_detecta_prestamo_repetido_y_escenario_faltante(lakehouse):
    _parquet(
        lakehouse.gold_kpis, pd.DataFrame({"cc_num": [1, 2], "loan_id": ["L1", "L1"]})
    )
    pd.DataFrame(
        {"cc_num": [1, 1, 2], "escenario": ["base", "negocio", "base"]}
    ).to_parquet(lakehouse.simulacion_resumen)
    c = _por_nombre(revisar_consistencia(lakehouse))
    assert c["Préstamo asignado a cada cliente"].ok is False
    assert c["Simulador Monte Carlo"].ok is False


def test_sin_datos_no_cuenta_como_falla(tmp_path):
    chequeos = revisar_consistencia(datos.Rutas(tmp_path))
    assert all(c.ok is None for c in chequeos)


# ---------------------------------------------------------------------------
# SLA
# ---------------------------------------------------------------------------


def _corridas(estados, minutos=5.0):
    fin = [AHORA - pd.Timedelta(hours=2 + 24 * i) for i in range(len(estados))][::-1]
    return pd.DataFrame({"state": estados, "minutos": minutos, "fin": fin})


def _regla(ok):
    return SimpleNamespace(nombre="regla", ok=ok, detalle="")


def test_todos_los_sla_se_cumplen():
    evaluacion = {"correctas": 75, "total": 75, "segundos": 3.5, "fecha": AHORA}
    res = slas.evaluar(
        _corridas(["success"] * 10),
        [_regla(True)] * 3,
        [Chequeo("x", True, "")],
        evaluacion,
        AHORA,
    )
    assert [r.cumple for r in res] == [True] * 6
    assert res[1].real == "2.0 h"  # frescura


def test_sla_que_no_se_cumplen():
    evaluacion = {"correctas": 60, "total": 75, "segundos": 3.5, "fecha": AHORA}
    corridas = _corridas(["failed", "failed", "success"], minutos=20.0)
    corridas.loc[2, "fin"] = AHORA - pd.Timedelta(hours=30)
    res = {
        r.clave: r
        for r in slas.evaluar(
            corridas, [_regla(False)], [Chequeo("x", False, "")], evaluacion, AHORA
        )
    }
    assert res["duracion"].cumple is False and res["duracion"].real == "20.0 min"
    assert res["frescura"].cumple is False
    assert res["tasa_exito"].real == "33%"
    assert res["calidad"].cumple is False
    assert res["consistencia"].cumple is False
    assert res["asistente"].cumple is False


def test_sin_datos_los_sla_quedan_sin_medir():
    res = slas.evaluar(None, [], [Chequeo("x", None, "")], None, AHORA)
    assert all(r.cumple is None for r in res)


def test_la_corrida_actual_manda_en_duracion_y_frescura():
    res = slas.evaluar(
        _corridas(["success"], minutos=4.0),
        [],
        [],
        None,
        AHORA,
        corrida_actual={"minutos": 16.0, "fallidos": []},
    )
    assert res[0].real == "16.0 min" and res[0].cumple is False
    assert res[1].real == "0.0 h" and res[1].cumple is True


def test_lee_la_evaluacion_del_asistente(tmp_path):
    ruta = tmp_path / "resultados.csv"
    pd.DataFrame({"ok": [True, True, False], "segundos": [2.0, 4.0, 6.0]}).to_csv(ruta)
    ev = slas.leer_evaluacion(ruta)
    assert ev["correctas"] == 2 and ev["total"] == 3 and ev["segundos"] == 4.0
    assert slas.leer_evaluacion(tmp_path / "no_existe.csv") is None


# ---------------------------------------------------------------------------
# Alertas
# ---------------------------------------------------------------------------


def test_alertas_solo_de_los_sla_que_fallan():
    res = slas.evaluar(_corridas(["success"]), [_regla(False)], [], None, AHORA)
    alertas = al.de_slas(res)
    assert [a.origen for a in alertas] == ["SLA calidad"]
    assert alertas[0].nivel == "crítica"


def test_registro_e_historial(tmp_path):
    ruta = tmp_path / "alertas.jsonl"
    vieja = al.Alerta(
        "crítica", "Falló el paso x", "d", "paso x", "2026-10-29T10:00:00+00:00"
    )
    nueva = al.Alerta("advertencia", "SLA y", "d", "SLA y", "2026-10-30T10:00:00+00:00")
    al.registrar([vieja], "enviado", ruta)
    with ruta.open("a", encoding="utf-8") as f:
        f.write("línea dañada\n")
    al.registrar([nueva], "no configurado", ruta)
    h = al.historial(ruta)
    assert list(h["titulo"]) == ["SLA y", "Falló el paso x"]
    assert list(h["correo"]) == ["no configurado", "enviado"]
    assert al.historial(tmp_path / "no_existe.jsonl").empty


def test_correo_sin_configurar_no_truena(tmp_path, monkeypatch):
    for var in ("ALERTAS_SMTP_USUARIO", "ALERTAS_SMTP_CONTRASENA"):
        monkeypatch.delenv(var, raising=False)
    assert al.config_correo({}) is None
    assert al.config_correo({"ALERTAS_SMTP_USUARIO": "yo@x.com"}) is None
    estado = al.notificar(
        [al.Alerta("crítica", "Falló", "d", "o")], tmp_path / "a.jsonl"
    )
    assert estado == "no configurado"
    assert al.historial(tmp_path / "a.jsonl").iloc[0]["correo"] == "no configurado"


class SMTPFalso:
    enviados = []

    def __init__(self, host, puerto, timeout):
        self.host, self.puerto = host, puerto

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def starttls(self):
        pass

    def login(self, usuario, contrasena):
        if contrasena == "mala":
            raise OSError("credenciales inválidas")

    def send_message(self, msg):
        SMTPFalso.enviados.append(msg)


def test_manda_un_solo_correo_con_todas_las_alertas(tmp_path):
    config = al.config_correo(
        {"ALERTAS_SMTP_USUARIO": "yo@x.com", "ALERTAS_SMTP_CONTRASENA": "clave"}
    )
    assert config["host"] == "smtp.gmail.com" and config["para"] == "yo@x.com"
    alertas = [
        al.Alerta("crítica", "Falló el paso clean_silver", "d", "paso clean_silver"),
        al.Alerta("advertencia", "SLA duración", "d", "SLA duracion"),
    ]
    SMTPFalso.enviados = []
    estado = al.notificar(alertas, tmp_path / "a.jsonl", config, SMTPFalso)
    assert estado == "enviado" and len(SMTPFalso.enviados) == 1
    msg = SMTPFalso.enviados[0]
    assert msg["Subject"] == (
        "[Gemelo Digital] 2 alertas (1 crítica): Falló el paso clean_silver"
    )
    assert "[CRÍTICA] Falló el paso clean_silver" in msg.get_content()
    assert len(al.historial(tmp_path / "a.jsonl")) == 2


def test_si_el_correo_falla_se_registra_el_error(tmp_path):
    config = al.config_correo(
        {"ALERTAS_SMTP_USUARIO": "yo@x.com", "ALERTAS_SMTP_CONTRASENA": "mala"}
    )
    estado = al.notificar(
        [al.Alerta("crítica", "x", "d", "o")], tmp_path / "a.jsonl", config, SMTPFalso
    )
    assert estado.startswith("error: credenciales inválidas")
    assert al.historial(tmp_path / "a.jsonl").iloc[0]["correo"].startswith("error")


# ---------------------------------------------------------------------------
# Revisión completa
# ---------------------------------------------------------------------------


def test_revision_completa_guarda_el_estado(lakehouse, tmp_path):
    estado = tmp_path / "estado.json"
    revision = revisar(
        lakehouse,
        url_airflow=f"sqlite:///{tmp_path / 'no_existe' / 'airflow.db'}",
        ahora=AHORA,
        ruta_estado=estado,
        ruta_evaluacion=tmp_path / "no_existe.csv",
    )
    assert revision["error_airflow"]
    guardado = json.loads(estado.read_text("utf-8"))
    assert [s["clave"] for s in guardado["slas"]][:2] == ["duracion", "frescura"]
    assert len(guardado["consistencia"]) == 7
