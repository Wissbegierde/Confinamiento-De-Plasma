"""Test rapido de motor_lite + montecarlo"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from particulas   import Particula
from colisiones   import ColisionEstocastica, velocidad_inicial_mb
from contenedor   import ContenedorCilindrico
import campos     as campos_mod
import montecarlo as mc
from motor_lite   import motor_lite

N = 10; PASOS = 300; DT = 1e-10
RADIO = 0.05; ALTURA = 0.10
M = 9.109e-31; Q = -1.602e-19; T = 1e6
B0 = 1.0; NU = 1e8

rng  = np.random.default_rng(42)
cont = ContenedorCilindrico(radio=RADIO, altura=ALTURA)

particulas, motores = [], []
for i in range(N):
    x0  = cont.posicion_aleatoria(rng)
    v0  = velocidad_inicial_mb(M, T, rng=rng)
    p   = Particula(i, q=Q, m=M, x0=x0, v0=v0)
    col = ColisionEstocastica(nu=NU, m=M, T=T, dt=DT, seed=i)
    particulas.append(p)
    motores.append(col)

fn_E = lambda pos: campos_mod.campo_electrico_constante(pos)
fn_B = lambda pos: campos_mod.campo_magnetico_solenoide(pos, B0=B0, radio=RADIO)

te, E_hist = motor_lite(
    pasos=PASOS, particulas=particulas, motores_colision=motores,
    fn_E=fn_E, fn_B=fn_B, dt=DT, contenedor=cont,
    registrar_energia=True, verbose=True, intervalo_log=100
)

stats        = mc.calcular_tau(te, N, DT, PASOS)
t_arr, N_arr = mc.curva_decaimiento(te, N, DT, PASOS)

print()
print("Escaparon  :", stats["n_escaparon"], "/", N)
print("tau medio  :", round(stats["tau_medio"] * 1e9, 2), "ns")
print("len(E_hist):", len(E_hist), " (esperado", PASOS, ")")
print("N_arr[0]   :", N_arr[0], " (esperado", N, ")")

diffs = [N_arr[i+1] - N_arr[i] for i in range(len(N_arr)-1)]
print("N_arr mono :", all(d <= 0 for d in diffs))
print()

mc.imprimir_resumen(stats, escala_key="ns",
                    tau_ref=mc.tau_bohm(RADIO, B0, T, abs(Q)))
print("TEST motor_lite: OK")
