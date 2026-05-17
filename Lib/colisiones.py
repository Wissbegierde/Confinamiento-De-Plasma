"""
colisiones.py
=============
Modelo estocástico de colisiones.

Módulo 1: Distribución de Maxwell-Boltzmann
    - velocidad_maxwell_boltzmann  : muestrea |v| de la distribución MB
    - velocidad_inicial_mb         : genera un vector v 3D isotrópico con |v|~MB

Módulo 2: Modelo estocástico de colisión
    - ColisionEstocastica          : clase que decide si hay colisión en cada paso
                                     y redistribuye la velocidad de la partícula
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# PARTE 1 — Distribución de Maxwell-Boltzmann
# ──────────────────────────────────────────────────────────────────────────────

def velocidad_maxwell_boltzmann(m: float, T: float, rng=None) -> float:
    """
    Muestrea la rapidez escalar |v| de la distribución de Maxwell-Boltzmann.

    La distribución MB de rapideces es:
        f(v) = 4π n (m / 2πkT)^(3/2) · v² · exp(-mv² / 2kT)

    Se muestrea usando el hecho de que v² ~ χ²(3) escalado, o equivalentemente
    sumando los cuadrados de tres gaussianas independientes N(0, kT/m).

    Parámetros
    ----------
    m   : masa de la partícula [kg]
    T   : temperatura del plasma [K]
    rng : numpy.random.Generator (opcional); si None, usa numpy global.

    Retorna
    -------
    rapidez : float  [m/s]
    """
    kB = 1.380649e-23          # Constante de Boltzmann [J/K]
    sigma = np.sqrt(kB * T / m)  # desviación estándar de cada componente
    rng = rng or np.random.default_rng()
    componentes = rng.normal(0.0, sigma, size=3)
    return float(np.linalg.norm(componentes))


def velocidad_inicial_mb(m: float, T: float, rng=None) -> np.ndarray:
    """
    Genera un vector de velocidad 3D con módulo distribuido según
    Maxwell-Boltzmann y dirección uniforme en la esfera unitaria.

    Parámetros
    ----------
    m   : masa de la partícula [kg]
    T   : temperatura del plasma [K]
    rng : numpy.random.Generator (opcional)

    Retorna
    -------
    v : np.ndarray de forma (3,)  [m/s]
    """
    kB = 1.380649e-23
    sigma = np.sqrt(kB * T / m)
    rng = rng or np.random.default_rng()
    # Las tres componentes son N(0,σ) ⟹ |v| ~ Maxwell-Boltzmann
    return rng.normal(0.0, sigma, size=3)


# ──────────────────────────────────────────────────────────────────────────────
# PARTE 2 — Modelo estocástico de colisión
# ──────────────────────────────────────────────────────────────────────────────

class ColisionEstocastica:
    """
    Modelo probabilístico de colisión partícula–partícula / partícula–fondo.

    En cada paso de tiempo dt la probabilidad de colisión es:
        P_colision = 1 - exp(-ν · dt)
    donde ν [Hz] es la frecuencia media de colisiones.

    Si ocurre la colisión, la velocidad de la partícula se sustituye por un
    nuevo vector muestreado de la distribución de Maxwell-Boltzmann a la
    temperatura T_fondo del plasma de fondo.

    Parámetros
    ----------
    nu    : frecuencia de colisión media [Hz]
    m     : masa de la partícula [kg]
    T     : temperatura del plasma de fondo [K]
    dt    : paso de tiempo de la simulación [s]
    seed  : semilla opcional para el generador (reproducibilidad)
    """

    def __init__(self, nu: float, m: float, T: float, dt: float, seed=None):
        self.nu = nu
        self.m = m
        self.T = T
        self.dt = dt
        self.p_colision = 1.0 - np.exp(-nu * dt)
        self.rng = np.random.default_rng(seed)

        # Contadores de diagnóstico
        self.n_pasos = 0
        self.n_colisiones = 0

    def aplicar(self, v: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Evalúa si ocurre una colisión en este paso y, en caso afirmativo,
        redistribuye la velocidad.

        Parámetros
        ----------
        v : np.ndarray de forma (3,) — velocidad actual de la partícula

        Retorna
        -------
        v_nueva   : np.ndarray (3,) — velocidad tras el paso
        colisionó : bool            — True si ocurrió una colisión
        """
        self.n_pasos += 1
        if self.rng.random() < self.p_colision:
            self.n_colisiones += 1
            v_nueva = velocidad_inicial_mb(self.m, self.T, rng=self.rng)
            return v_nueva, True
        return v.copy(), False

    @property
    def tasa_colision_real(self) -> float:
        """Fracción de pasos en que ocurrió una colisión (empírica)."""
        if self.n_pasos == 0:
            return 0.0
        return self.n_colisiones / self.n_pasos

    def resumen(self) -> str:
        return (
            f"ColisionEstocastica | ν={self.nu:.2e} Hz | "
            f"P_col/paso={self.p_colision:.4f} | "
            f"Colisiones: {self.n_colisiones}/{self.n_pasos} "
            f"({100*self.tasa_colision_real:.2f}%)"
        )
