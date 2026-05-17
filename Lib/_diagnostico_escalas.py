"""
Diagnostico de escalas fisicas para el barrido de B0.
Calcula: r_Larmor, T_ciclotron, tiempo de cruce, y 
verifica que dt << T_c y r_L sea comparable al radio del contenedor.
"""
import numpy as np

kB = 1.380649e-23

print("=" * 60)
print("  DIAGNOSTICO DE ESCALAS FISICAS")
print("=" * 60)

for nombre, m, q in [
    ("Electron",  9.109e-31, 1.602e-19),
    ("Proton",    1.673e-27, 1.602e-19),
    ("Deuterio",  3.344e-27, 1.602e-19),
]:
    print(f"\n--- {nombre} ---")
    for T in [1e4, 1e6]:
        v_th = np.sqrt(kB * T / m)
        print(f"  T = {T:.0e} K  | v_th = {v_th:.3e} m/s")
        for B0 in [0.01, 0.1, 1.0, 5.0]:
            T_c   = 2*np.pi*m / (q*B0)          # periodo ciclotron [s]
            r_L   = m * v_th / (q * B0)         # radio de Larmor [m]
            print(f"    B0={B0:.2f}T:  T_c={T_c*1e9:.2f} ns  "
                  f"r_L={r_L*1000:.4f} mm")

print("\n" + "=" * 60)
print("  REGIMEN OPTIMO PARA EL BARRIDO")
print("  Requisito: dt << T_c  y  r_L varia de >R a <R")
print("=" * 60)

# Buscar parametros donde el efecto sea visible
for nombre, m, q in [
    ("Proton",  1.673e-27, 1.602e-19),
]:
    T = 1e4   # K — temperatura baja para velocidades moderadas
    v_th = np.sqrt(kB * T / m)
    print(f"\n{nombre} a T={T:.0e} K, v_th={v_th:.2f} m/s")
    
    R_vals = [0.01, 0.05, 0.1]
    dt_vals = [1e-9, 1e-8, 1e-7]
    
    for R in R_vals:
        for dt in dt_vals:
            t_cruce = R / v_th
            B0_lim  = m * v_th / (q * R)  # r_L = R cuando B0 = B0_lim
            T_c_lim = 2*np.pi*m / (q*B0_lim)
            ratio   = dt / T_c_lim
            print(f"  R={R:.3f}m dt={dt:.0e}s | "
                  f"t_cruce={t_cruce*1e3:.1f}ms | "
                  f"B0_lim={B0_lim:.3f}T | "
                  f"T_c(B0_lim)={T_c_lim*1e6:.1f}us | "
                  f"dt/T_c={ratio:.3f}  {'OK' if ratio < 0.1 else 'MAL'}")
