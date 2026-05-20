"""
test_colisiones.py
==================
Pruebas para colisiones.py (Maxwell-Boltzmann, ColisionEstocastica).

Ejecutar desde Lib/:
    python test_colisiones.py
"""
import numpy as np

from colisiones import ColisionEstocastica, velocidad_inicial_mb
from test_helpers import (
    afirmar,
    imprimir_resumen_final,
    reset_resultados,
    setup_utf8_stdout_win,
)

kB = 1.380649e-23


def test_velocidad_vector_3d():
    v = velocidad_inicial_mb(m=1e-9, T=1e4)
    afirmar(v.shape == (3,), "velocidad_inicial_mb devuelve vector 3D")


def test_media_modulo_velocidad_mb():
    m, T = 1e-9, 1e4
    N = 100_000
    rng = np.random.default_rng(0)
    muestras = np.array(
        [np.linalg.norm(velocidad_inicial_mb(m, T, rng=rng)) for _ in range(N)]
    )
    v_mean_teorico = np.sqrt(8 * kB * T / (np.pi * m))
    v_mean_numerico = muestras.mean()
    error_rel = abs(v_mean_numerico - v_mean_teorico) / v_mean_teorico
    afirmar(
        error_rel < 0.01,
        f"<|v|> MB: err {error_rel*100:.3f}% < 1%",
    )


def test_isotropia_componentes():
    m, T = 1e-9, 1e4
    N = 100_000
    rng = np.random.default_rng(0)
    v_mean_teorico = np.sqrt(8 * kB * T / (np.pi * m))
    vecs = np.array([velocidad_inicial_mb(m, T, rng=rng) for _ in range(N)])
    medias = vecs.mean(axis=0)
    afirmar(
        np.all(np.abs(medias) < 0.01 * v_mean_teorico),
        "componentes isotrópicas (~0)",
    )


def test_probabilidad_colision():
    m, T = 1e-9, 1e4
    nu, dt = 500.0, 5e-5
    p_teorica = 1.0 - np.exp(-nu * dt)
    col = ColisionEstocastica(nu=nu, m=m, T=T, dt=dt, seed=42)
    v_test = np.zeros(3)
    for _ in range(200_000):
        v_test, _ = col.aplicar(v_test)
    error_p = abs(col.tasa_colision_real - p_teorica)
    afirmar(error_p < 0.002, f"P_colisión empírica vs teórica Δ={error_p:.5f}")


def test_colision_cambia_velocidad():
    m, T = 1e-9, 1e4
    col2 = ColisionEstocastica(nu=1e9, m=m, T=T, dt=1.0, seed=0)
    v_antes = np.zeros(3)
    v_despues, colisiono = col2.aplicar(v_antes)
    afirmar(
        colisiono and np.linalg.norm(v_despues) > 0,
        "tras colisión (nu alto) |v| > 0",
    )


def test_sin_colision_velocidad_constante():
    m, T = 1e-9, 1e4
    dt = 5e-5
    col3 = ColisionEstocastica(nu=0.0, m=m, T=T, dt=dt, seed=0)
    v_orig = np.array([1.0, 2.0, 3.0])
    v_out, colisiono = col3.aplicar(v_orig)
    afirmar(
        not colisiono and np.allclose(v_orig, v_out),
        "nu=0 no altera v",
    )


if __name__ == "__main__":
    setup_utf8_stdout_win()
    reset_resultados()
    print("\ntest_colisiones.py\n")
    test_velocidad_vector_3d()
    test_media_modulo_velocidad_mb()
    test_isotropia_componentes()
    test_probabilidad_colision()
    test_colision_cambia_velocidad()
    test_sin_colision_velocidad_constante()
    col = ColisionEstocastica(nu=500.0, m=1e-9, T=1e4, dt=5e-5, seed=42)
    print("\n" + col.resumen())
    imprimir_resumen_final()
