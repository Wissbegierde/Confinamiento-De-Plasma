"""
multiples.py
============
Genera N partículas distribuidas dentro del contenedor
con velocidades Maxwell-Boltzmann.
"""

import numpy as np
from particulas import Particula
from colisiones import ColisionEstocastica, velocidad_inicial_mb
from contenedor import (
    ContenedorCilindrico, ContenedorEsferico,
    ContenedorCaja, ContenedorPlacasParalelas, ContenedorTokamak
)


def _crear_contenedor(geometria, radio, altura, L, R_tokamak):
    """Fábrica: devuelve el objeto contenedor según la geometría."""
    opciones = {
        "cilindro": lambda: ContenedorCilindrico(radio=radio, altura=altura),
        "esfera":   lambda: ContenedorEsferico(radio=radio),
        "caja":     lambda: ContenedorCaja(Lx=L, Ly=L, Lz=L),
        "placas":   lambda: ContenedorPlacasParalelas(d=altura, L=L),
        "tokamak":  lambda: ContenedorTokamak(R=R_tokamak, a=radio),
    }
    if geometria not in opciones:
        raise ValueError(
            f"Geometría '{geometria}' no reconocida. "
            f"Opciones: {list(opciones)}"
        )
    return opciones[geometria]()


def Cantidad(
    m_particula = 1e-9,
    q_particula = 1e-6,
    T_plasma    = 1e4,
    nu_colision = 500.0,
    dt          = 1e-10,
    geometria   = "cilindro",
    radio       = 0.5,
    altura      = 1.0,
    L           = 1.0,
    R_tokamak   = 1.0,
):
    n   = int(input("Ingrese el número de partículas de prueba: "))
    rng = np.random.default_rng(42)

    contenedor       = _crear_contenedor(geometria, radio, altura, L, R_tokamak)
    particulas       = []
    motores_colision = []

    for i in range(n):
        x0 = contenedor.posicion_aleatoria(rng)          # distribución correcta
        v0 = velocidad_inicial_mb(m_particula, T_plasma, rng=rng)

        p  = Particula(i, q=q_particula, m=m_particula, x0=x0, v0=v0)
        particulas.append(p)

        col = ColisionEstocastica(
            nu=nu_colision, m=m_particula, T=T_plasma, dt=dt, seed=i
        )
        motores_colision.append(col)

    # Retorna el contenedor también — lo necesita el motor
    return particulas, motores_colision, n, contenedor
