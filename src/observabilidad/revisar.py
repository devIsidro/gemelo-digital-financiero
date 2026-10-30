"""
Revisión completa de la plataforma (Fase 6): SLA, consistencia entre capas y
alertas. La usan el último paso del DAG (revisar_slas) y la línea de comandos:

    # Ver cómo está todo (no manda nada)
    docker compose exec dashboard python -m src.observabilidad.revisar

    # Además registrar y mandar por correo las alertas
    docker compose exec dashboard python -m src.observabilidad.revisar --notificar

    # Probar las alertas: manda un correo de prueba y lo deja en el panel
    docker compose exec dashboard python -m src.observabilidad.revisar --correo-prueba

Termina con código 0 si todos los SLA medibles se cumplen y 2 si alguno no.
El resultado también queda en data/observabilidad/estado.json.
"""

import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.dashboard import datos
from src.observabilidad import alertas as al
from src.observabilidad import slas
from src.observabilidad.consistencia import revisar_consistencia
from src.spark.validate_silver import ResultadoExpectativa

RUTAS = datos.Rutas(Path("data"))
RUTA_ESTADO = Path("data/observabilidad/estado.json")
RUTA_EVALUACION_IA = Path("data/evaluacion_ia/resultados.csv")


def _a_json(valor):
    """numpy.bool_, numpy.int64, Timestamp… a algo que JSON entienda."""
    return valor.item() if hasattr(valor, "item") else str(valor)


def reglas_de_calidad(rutas) -> list:
    """Las reglas de Silver (las mismas que muestra el panel). Si la
    validación misma truena, cuenta como regla que no se cumple."""
    reglas = []
    try:
        if rutas.silver_tx.exists():
            reglas += list(datos.validar_transacciones(rutas))
        if rutas.silver_loans.exists():
            reglas += list(datos.validar_prestamos(datos.prestamos_silver(rutas)))
    except Exception as err:  # noqa: BLE001
        reglas.append(
            ResultadoExpectativa(
                "Validación de calidad", False, f"No se pudo correr: {err}"
            )
        )
    return reglas


def revisar(
    rutas=RUTAS,
    url_airflow: str = None,
    ahora=None,
    corrida_actual: dict = None,
    ruta_estado: Path = RUTA_ESTADO,
    ruta_evaluacion: Path = RUTA_EVALUACION_IA,
) -> dict:
    ahora = ahora or pd.Timestamp.now(tz=datos.ZONA_HORARIA)
    corridas, error_airflow = None, None
    try:
        corridas = datos.corridas(url_airflow or datos.url_airflow())
    except Exception as err:  # noqa: BLE001
        error_airflow = str(err).splitlines()[0][:160]
    chequeos = revisar_consistencia(rutas)
    resultados = slas.evaluar(
        corridas,
        reglas_de_calidad(rutas),
        chequeos,
        slas.leer_evaluacion(ruta_evaluacion),
        ahora,
        corrida_actual,
    )
    estado = {
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "origen": "DAG" if corrida_actual is not None else "manual",
        "slas": [asdict(r) for r in resultados],
        "consistencia": [asdict(c) for c in chequeos],
        "error_airflow": error_airflow,
    }
    ruta_estado = Path(ruta_estado)
    ruta_estado.parent.mkdir(parents=True, exist_ok=True)
    ruta_estado.write_text(
        json.dumps(estado, ensure_ascii=False, indent=2, default=_a_json), "utf-8"
    )
    return {
        "resultados": resultados,
        "chequeos": chequeos,
        "error_airflow": error_airflow,
    }


def imprimir(revision: dict) -> None:
    marca = {True: "OK ", False: "MAL", None: " — "}
    print("SLA")
    for r in revision["resultados"]:
        print(f"  [{marca[r.cumple]}] {r.nombre}: {r.real} (objetivo {r.objetivo})")
        if r.detalle:
            print(f"        {r.detalle}")
    print("\nConsistencia entre capas")
    for c in revision["chequeos"]:
        print(f"  [{marca[c.ok]}] {c.nombre}: {c.detalle}")
    if revision["error_airflow"]:
        print(f"\nSin conexión a Airflow: {revision['error_airflow']}")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--correo-prueba" in argv:
        prueba = al.Alerta(
            "advertencia",
            "Correo de prueba",
            "Si te llegó este correo, las alertas del Gemelo Digital ya funcionan.",
            "prueba manual",
        )
        # También queda en el panel, para comprobar las dos vías de aviso.
        estado = al.notificar([prueba])
        print(f"Correo de prueba: {estado}")
        return 0 if estado == "enviado" else 1
    revision = revisar()
    imprimir(revision)
    if "--notificar" in argv:
        estado = al.notificar(al.de_slas(revision["resultados"]))
        print(f"\nAlertas: {estado}")
    return 2 if any(r.cumple is False for r in revision["resultados"]) else 0


if __name__ == "__main__":
    sys.exit(main())
