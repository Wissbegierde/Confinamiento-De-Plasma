import numpy as np


def campo_electrico_constante(x, E0=(0.0, 0.0, 0.0)):
    """
    Campo eléctrico uniforme y constante en el espacio.
    x se mantiene por compatibilidad de interfaz.
    """
    return np.array(E0, dtype=float)


def campo_magnetico_solenoide(x, B0=1.0, radio=0.5):
    """
    Modelo sencillo de solenoide:
    - Dentro de un cilindro de radio 'radio' alrededor del eje z: B = (0, 0, B0)
    - Fuera del cilindro: B = 0
    """
    r = np.linalg.norm(x[:2])
    if r <= radio:
        return np.array([0.0, 0.0, B0])
    return np.zeros(3)

