"""
verificar_corrida.py
====================
Corre una simulación y comprueba coherencia interna de los datos
(tiempos de escape, impactos, energía, escalas físicas).

Uso:
    python verificar_corrida.py
    python verificar_corrida.py --B0 0.2
"""

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from particulas import Particula
from colisiones import ColisionEstocastica, velocidad_inicial_mb
from contenedor import ContenedorCilindrico
from motor_lite import motor_lite, campo_B_solenoide_vec, campo_E_cero_vec
import montecarlo as mc
from mapas_calor import densidad_rz_desde_muestreos, flujo_pared_desde_impactos

M, Q = 1.673e-27, 1.602e-19
K_B = 1.380649e-23


def verificar(B0=0.2, N=40, pasos=10000, dt=1e-8, radio=0.01, altura=0.02,
              T=1e4, nu=500.0, seed=123):
    print("=" * 60)
    print(f"VERIFICACION DE CORRIDA  B0={B0} T  N={N}  pasos={pasos}")
    print("=" * 60)

    rng = np.random.default_rng(seed)
    cont = ContenedorCilindrico(radio=radio, altura=altura)
    particulas, motores = [], []
    for i in range(N):
        x0 = cont.posicion_aleatoria(rng)
        v0 = velocidad_inicial_mb(M, T, rng=rng)
        particulas.append(Particula(i, q=Q, m=M, x0=x0, v0=v0))
        motores.append(ColisionEstocastica(nu=nu, m=M, T=T, dt=dt, seed=seed + i))

    historial, impactos = [], []
    fn_B = lambda X, b=B0: campo_B_solenoide_vec(X, B0=b, radio=radio)

    te, E_hist = motor_lite(
        pasos=pasos,
        particulas=particulas,
        motores_colision=motores,
        fn_E=campo_E_cero_vec,
        fn_B=fn_B,
        dt=dt,
        contenedor=cont,
        registrar_energia=True,
        registrar_muestreo=True,
        intervalo_muestreo=20,
        historial_posiciones=historial,
        registrar_impactos=True,
        impactos_pared=impactos,
        verbose=True,
    )

    stats = mc.calcular_tau(te, N, dt, pasos)
    v_th = np.sqrt(3 * K_B * T / M)
    r_L = M * v_th / (Q * B0)
    T_c = 2 * np.pi * M / (Q * B0)
    t_cross = radio / v_th

    print("\n--- Escalas físicas ---")
    print(f"  v_th       = {v_th:.0f} m/s")
    print(f"  r_L        = {r_L*1e3:.2f} mm  (R={radio*1e3:.0f} mm)")
    print(f"  T_ciclo    = {T_c*1e9:.1f} ns   dt/T_c = {dt/T_c:.4f}")
    print(f"  t_cruce R  = {t_cross*1e6:.2f} us")

    print("\n--- Coherencia interna ---")
    ok = True

    # 1. Un impacto por partícula escapada
    n_esc = len(te)
    n_imp = len(impactos)
    c1 = n_imp == n_esc
    print(f"  [{'OK' if c1 else 'FALLO'}] impactos ({n_imp}) == escapadas ({n_esc})")
    ok &= c1

    # 2. IDs de escape coinciden
    ids_te = set(te.keys())
    ids_imp = {imp["id"] for imp in impactos}
    c2 = ids_te == ids_imp
    print(f"  [{'OK' if c2 else 'FALLO'}] IDs escape == IDs impactos")
    ok &= c2

    # 3. Tiempos de escape coherentes con paso
    errs_t = []
    for imp in impactos:
        t_esp = te.get(imp["id"])
        if t_esp is not None:
            errs_t.append(abs(t_esp - imp["t"]))
    c3 = max(errs_t) < 1e-15 if errs_t else True
    print(f"  [{'OK' if c3 else 'FALLO'}] t_escape en impactos == dict te")
    ok &= c3

    # 4. tau medio vs manual
    if n_esc > 0:
        tau_manual = np.mean(list(te.values()))
        tau_stats = stats["tau_medio"]
        rel = abs(tau_manual - tau_stats) / tau_stats if tau_stats > 0 else 0
        c4 = rel < 1e-10 or (n_esc < N)  # con censuradas, mean difiere
        if n_esc == N:
            c4 = rel < 1e-10
            print(f"  [{'OK' if c4 else 'FALLO'}] tau_medio stats={tau_stats*1e6:.3f} "
                  f"manual={tau_manual*1e6:.3f} us")
        else:
            print(f"  [OK ] tau_medio (con censuradas) = {tau_stats*1e6:.3f} us")
            c4 = True
        ok &= c4

    # 5. Energía no negativa
    E = np.array(E_hist)
    c5 = np.all(E >= 0)
    print(f"  [{'OK' if c5 else 'FALLO'}] E_cin >= 0 en todos los pasos")
    ok &= c5

    # 6. Densidad interior no trivial
    H, re, ze = densidad_rz_desde_muestreos(historial, cont)
    r_cent = 0.5 * (re[:-1] + re[1:])
    frac_interior = (H[r_cent < radio * 0.8, :].sum() / H.sum()
                       if H.sum() > 0 else 0)
    c6 = frac_interior > 0.1
    print(f"  [{'OK' if c6 else 'AVISO'}] fraccion densidad en r<0.8R: "
          f"{frac_interior*100:.1f}%  (mapa no solo paredes)")
    if not c6:
        print("       -> Si es 0%, todas escaparon muy rapido o revisar muestreo.")

    # 7. Flujo suma
    flujo, _, tipos = flujo_pared_desde_impactos(impactos, cont)
    suma_flujo = sum(flujo[t].sum() for t in tipos)
    c7 = abs(suma_flujo - n_imp) < 0.5
    print(f"  [{'OK' if c7 else 'FALLO'}] suma flujo pared = {suma_flujo:.0f}")
    ok &= c7

    # 8. Orden de magnitud tau vs t_cross
    ratio = stats["tau_medio"] / t_cross
    c8 = 0.3 < ratio < 50
    print(f"  [{'OK' if c8 else 'AVISO'}] tau/t_cruce = {ratio:.2f}  "
          f"(esperado ~1-10 para cilindro corto)")
    if not c8:
        print("       -> Revisar parametros si ratio fuera de rango.")

    print("\n--- Resultado ---")
    frac_conf = stats["n_confinadas"] / N
    print(f"  Confinadas: {stats['n_confinadas']}/{N} ({frac_conf*100:.1f}%)")
    print(f"  tau_medio = {stats['tau_medio']*1e6:.2f} us")
    print(f"  Impactos: lateral={sum(1 for i in impactos if i['tipo_pared']=='lateral')}, "
          f"z_min={sum(1 for i in impactos if i['tipo_pared']=='z_min')}, "
          f"z_max={sum(1 for i in impactos if i['tipo_pared']=='z_max')}")

    if ok:
        print("\n  VERIFICACION PASADA (coherencia interna OK)")
    else:
        print("\n  VERIFICACION CON FALLOS — revisar arriba")

    return ok, stats, te, impactos, H


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--B0", type=float, default=0.2)
    p.add_argument("--N", type=int, default=40)
    args = p.parse_args()
    verificar(B0=args.B0, N=args.N)
