"""
Diagnostico de rendimiento de motor_lite para reproducir el cuelgue en [18/18].
Ejecuta la corrida exacta que se cuelga: B0=1.0T, seed=1.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from particulas   import Particula
from colisiones   import ColisionEstocastica, velocidad_inicial_mb
from contenedor   import ContenedorCilindrico
from motor_lite   import motor_lite, campo_B_solenoide_vec, campo_E_cero_vec

# Parámetros idénticos al barrido
M_PARTICULA   = 1.673e-27
Q_PARTICULA   = 1.602e-19
T_PLASMA      = 1e4
NU_COLISION   = 500.0
RADIO         = 0.01
ALTURA        = 0.02
DT            = 1e-8
N_PARTICULAS  = 40
PASOS         = 10_000

B0   = 1.0
SEED = 1

print(f"Diagnostico: B0={B0}T seed={SEED}")
print(f"N={N_PARTICULAS} PASOS={PASOS} DT={DT:.0e}")
print()

rng  = np.random.default_rng(SEED)
cont = ContenedorCilindrico(radio=RADIO, altura=ALTURA)

particulas, motores = [], []
for i in range(N_PARTICULAS):
    x0  = cont.posicion_aleatoria(rng)
    v0  = velocidad_inicial_mb(M_PARTICULA, T_PLASMA, rng=rng)
    p   = Particula(i, q=Q_PARTICULA, m=M_PARTICULA, x0=x0, v0=v0)
    col = ColisionEstocastica(nu=NU_COLISION, m=M_PARTICULA, T=T_PLASMA,
                              dt=DT, seed=SEED * 1000 + i)
    particulas.append(p)
    motores.append(col)

# Mostrar velocidades iniciales para detectar valores extremos
V0 = np.array([p.v for p in particulas])
v_norms = np.linalg.norm(V0, axis=1)
print(f"v_th esperada: {np.sqrt(1.380649e-23 * T_PLASMA / M_PARTICULA):.0f} m/s")
print(f"v_norm min/max/mean: {v_norms.min():.0f} / {v_norms.max():.0f} / {v_norms.mean():.0f} m/s")

fn_E = campo_E_cero_vec
fn_B = lambda X: campo_B_solenoide_vec(X, B0=B0, radio=RADIO)

print(f"\nEjecutando motor_lite con verbose cada 1000 pasos...")
t0 = time.perf_counter()

te, Ecin = motor_lite(
    pasos            = PASOS,
    particulas       = particulas,
    motores_colision = motores,
    fn_E             = fn_E,
    fn_B             = fn_B,
    dt               = DT,
    contenedor       = cont,
    registrar_energia= True,
    verbose          = True,
    intervalo_log    = 1000,
)

elapsed = time.perf_counter() - t0
print(f"\nTiempo total: {elapsed:.2f} s")
print(f"Tiempo por paso: {elapsed/PASOS*1000:.3f} ms")
print(f"Escaparon: {len(te)}/{N_PARTICULAS}")
