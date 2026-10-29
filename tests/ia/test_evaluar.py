"""Pruebas de las revisiones automáticas de la evaluación del asistente."""

from src.ia import evaluar as ev


def test_lee_cifras_como_numeros():
    assert ev.numeros("De $84,477 a $72,605, o sea 12.1%. En 1,000 futuros.") == [
        84477.0,
        72605.0,
        12.1,
        1000.0,
    ]


def test_lee_cifras_negativas():
    assert ev.numeros("Bajó a -87.1% y tu ahorro sería de -$87,341.") == [
        -87.1,
        -87341.0,
    ]
    assert ev.contiene("se redujo a -87.1%", -87.1)
    assert not ev.contiene("se redujo a 87.1%", -87.1)


def test_detecta_cifras_inventadas():
    permitido = "Tu probabilidad es 12.1%. En 7 meses juntas $50,000."
    assert ev.cifras_inventadas("Es 12.1% y la juntas en 7 meses.", permitido) == []
    assert ev.cifras_inventadas("La juntas en 9 meses con $6,300.", permitido) == [
        9.0,
        6300.0,
    ]


def test_reconoce_la_categoria_aunque_cambie_el_formato():
    assert ev.menciona("Gastaste más en compras en linea.", "Compras (en línea)")
    assert not ev.menciona("Gastaste más en viajes.", "Compras (en línea)")


def test_cuenta_oraciones_sin_cortar_decimales():
    assert ev.oraciones("Pasaría de 7.6% a 13.2%. Tu ahorro baja.") == 2
