"""
validacion_fisica.py
====================
Semana 14: validación física integral y análisis de deriva de energía.

Corridas y datos guardados en data/validacion/:
  - larmor_validacion.csv / .png
  - ciclotron_validacion.csv / .png
  - tau_vs_bohm.csv / .png
  - deriva_energia.csv / .png
  - resumen_validacion.json
"""

import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(__file__))

from particulas import Particula
from integradores import boris_step
from contenedor import ContenedorCilindrico
from colisiones import ColisionEstocastica, velocidad_inicial_mb
from motor_lite import motor_lite, campo_B_solenoide_vec, campo_E_cero_vec
import montecarlo as mc

GUARDAR_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "validacion"
)

# Protón de referencia
M_P = 1.673e-27
Q_P = 1.602e-19
K_B = 1.380649e-23


# ══════════════════════════════════════════════════════════════
#  1. RADIO DE LARMOR
# ══════════════════════════════════════════════════════════════

def validar_larmor(
    q=Q_P, m=M_P, B0=0.5,
    v_perp_vals=(5e3, 1e4, 2e4),
    dt=5e-10, pasos=4000,
    carpeta=GUARDAR_DIR,
):
    """Compara r_L teórico vs. numérico para varias v_perp."""
    os.makedirs(carpeta, exist_ok=True)
    filas = []

    fig, axes = plt.subplots(1, len(v_perp_vals), figsize=(4 * len(v_perp_vals), 3.5))
    if len(v_perp_vals) == 1:
        axes = [axes]

    for ax, v_perp in zip(axes, v_perp_vals):
        # Protón (q>0), B||+z: giro horario → v tangencial en -y
        v0 = np.array([0.0, -v_perp, 0.0])
        r_teo = m * v_perp / (abs(q) * B0)
        x0 = np.array([r_teo, 0.0, 0.0])
        p = Particula(0, q=q, m=m, x0=x0, v0=v0)
        B = np.array([0.0, 0.0, B0])

        for _ in range(pasos):
            x_n, v_n = boris_step(p.x, p.v, np.zeros(3), B, q, m, dt)
            p.actualizar_estado(x_n, v_n)

        traj = np.array(p.historia_x)
        t = np.arange(traj.shape[0]) * dt
        r_xy = np.sqrt(traj[:, 0] ** 2 + traj[:, 1] ** 2)
        # Descartar transitorio inicial (~10 %)
        n0 = max(10, len(r_xy) // 10)
        r_num = float(np.median(r_xy[n0:]))
        err_pct = 100 * abs(r_num - r_teo) / r_teo

        filas.append([v_perp, B0, r_teo, r_num, err_pct])
        ax.plot(traj[:, 0] * 1e3, traj[:, 1] * 1e3, lw=0.6)
        ax.set_aspect("equal")
        ax.set_title(f"v⊥={v_perp/1e3:.0f} km/s\n"
                     f"r_L: {r_teo*1e3:.2f} vs {r_num*1e3:.2f} mm")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")

    arr = np.array(filas)
    np.savetxt(
        os.path.join(carpeta, "larmor_validacion.csv"),
        arr,
        delimiter=",",
        header="v_perp_m_s,B0_T,r_L_teorico_m,r_L_numerico_m,error_pct",
        comments="",
    )
    fig.suptitle("Validación radio de Larmor (Boris, B uniforme)")
    fig.savefig(os.path.join(carpeta, "larmor_validacion.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Larmor: error máx = {arr[:, 4].max():.3f}%")
    return arr


# ══════════════════════════════════════════════════════════════
#  2. PERIODO CICLOTRÓNICO
# ══════════════════════════════════════════════════════════════

def validar_ciclotron(
    q=Q_P, m=M_P, B0=0.5, v_perp=1e4,
    dt=5e-10, pasos=12000,
    carpeta=GUARDAR_DIR,
):
    """Mide T_c numérico del movimiento en x(t) y compara con 2πm/(qB)."""
    os.makedirs(carpeta, exist_ok=True)
    v0 = np.array([0.0, -v_perp, 0.0])
    r_teo = m * v_perp / (abs(q) * B0)
    p = Particula(0, q=q, m=m, x0=[r_teo, 0, 0], v0=v0)
    B = np.array([0.0, 0.0, B0])

    for _ in range(pasos):
        x_n, v_n = boris_step(p.x, p.v, np.zeros(3), B, q, m, dt)
        p.actualizar_estado(x_n, v_n)

    traj = np.array(p.historia_x)
    t = np.arange(traj.shape[0]) * dt
    x_sig = traj[:, 0]

    T_teo = 2 * np.pi * m / (abs(q) * B0)
    # Picos en x(t): distancia entre máximos ≈ medio período
    n_per = max(20, int(T_teo / dt))
    peaks, _ = find_peaks(x_sig, distance=int(0.8 * n_per))
    if len(peaks) >= 3:
        periodos = np.diff(t[peaks])
        T_num = float(np.median(periodos))
    else:
        # Fallback: FFT del espectro
        freqs = np.fft.rfftfreq(len(x_sig), dt)
        spec = np.abs(np.fft.rfft(x_sig - x_sig.mean()))
        idx = 1 + int(np.argmax(spec[1:]))
        f_c = freqs[idx]
        T_num = 1.0 / f_c if f_c > 0 else np.nan

    err_pct = 100 * abs(T_num - T_teo) / T_teo if np.isfinite(T_num) else np.nan

    np.savetxt(
        os.path.join(carpeta, "ciclotron_validacion.csv"),
        [[B0, v_perp, T_teo, T_num, err_pct]],
        delimiter=",",
        header="B0_T,v_perp_m_s,T_ciclo_teorico_s,T_ciclo_numerico_s,error_pct",
        comments="",
    )

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t * 1e6, x_sig * 1e3, lw=0.7)
    if len(peaks) > 0:
        ax.plot(t[peaks] * 1e6, x_sig[peaks] * 1e3, "rx", ms=4)
    ax.set_xlabel("t [µs]")
    ax.set_ylabel("x [mm]")
    ax.set_title(
        f"Período ciclotrónico: T_teo={T_teo*1e9:.2f} ns, "
        f"T_num={T_num*1e9:.2f} ns ({err_pct:.2f}% err)"
    )
    fig.savefig(os.path.join(carpeta, "ciclotron_validacion.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Ciclotrón: error = {err_pct:.3f}%")
    return T_teo, T_num, err_pct


# ══════════════════════════════════════════════════════════════
#  3. τ SIMULADO VS. BOHM
# ══════════════════════════════════════════════════════════════

def validar_tau_vs_bohm(
    B0_vals=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
    N=30, pasos=8000, dt=1e-8,
    radio=0.01, altura=0.02, T_plasma=1e4,
    nu=200.0, seed=7,
    carpeta=GUARDAR_DIR,
):
    """Compara τ medio Monte Carlo con τ_Bohm teórico."""
    os.makedirs(carpeta, exist_ok=True)
    rng = np.random.default_rng(seed)
    filas = []

    for B0 in B0_vals:
        cont = ContenedorCilindrico(radio=radio, altura=altura)
        particulas, motores = [], []
        for i in range(N):
            x0 = cont.posicion_aleatoria(rng)
            v0 = velocidad_inicial_mb(M_P, T_plasma, rng=rng)
            particulas.append(Particula(i, q=Q_P, m=M_P, x0=x0, v0=v0))
            motores.append(ColisionEstocastica(
                nu=nu, m=M_P, T=T_plasma, dt=dt, seed=seed + i
            ))

        fn_B = lambda X, b=B0: campo_B_solenoide_vec(X, B0=b, radio=radio)
        te, _ = motor_lite(
            pasos=pasos, particulas=particulas, motores_colision=motores,
            fn_E=campo_E_cero_vec, fn_B=fn_B, dt=dt, contenedor=cont,
            registrar_energia=False, verbose=False,
        )
        stats = mc.calcular_tau(te, N, dt, pasos)
        tau_sim = stats["tau_medio"]
        tau_b = mc.tau_bohm(radio, B0, T_plasma, Q_P)
        ratio = tau_sim / tau_b if tau_b > 0 else np.nan
        frac_conf = stats["n_confinadas"] / stats["n_total"]
        filas.append([B0, tau_sim, tau_b, ratio, frac_conf])

    arr = np.array(filas)
    np.savetxt(
        os.path.join(carpeta, "tau_vs_bohm.csv"),
        arr,
        delimiter=",",
        header="B0_T,tau_simulado_s,tau_Bohm_s,ratio_sim_bohm,frac_confinadas",
        comments="",
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(arr[:, 0], arr[:, 1] * 1e6, "o-", label="τ simulado")
    ax.plot(arr[:, 0], arr[:, 2] * 1e6, "s--", label="τ Bohm (ref.)")
    ax.set_xlabel("B₀ [T]")
    ax.set_ylabel("τ [µs]")
    ax.set_title("Tiempo de confinamiento: simulación vs. Bohm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(carpeta, "tau_vs_bohm.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  tau vs Bohm: {len(B0_vals)} puntos guardados")
    return arr


# ══════════════════════════════════════════════════════════════
#  4. DERIVA DE ENERGÍA
# ══════════════════════════════════════════════════════════════

def _corrida_sin_colisiones(B0, dt, pasos, v_perp=1e4):
    """
    Una partícula en B uniforme sin fronteras: mide conservación de energía
    del integrador Boris (sin pérdidas en pared ni colisiones).
    """
    r_teo = M_P * v_perp / (Q_P * B0)
    p = Particula(0, q=Q_P, m=M_P, x0=[r_teo, 0, 0], v0=[0, -v_perp, 0])
    B = np.array([0.0, 0.0, B0])
    E_hist = []
    for _ in range(pasos):
        x_n, v_n = boris_step(p.x, p.v, np.zeros(3), B, Q_P, M_P, dt)
        p.x, p.v = x_n, v_n
        E_hist.append(0.5 * M_P * np.dot(v_n, v_n))
    E = np.array(E_hist)
    n0 = max(10, len(E) // 10)
    E0 = float(np.mean(E[n0 : n0 + max(50, len(E) // 20)]))
    drift_pct = 100 * (E[-1] - E0) / E0
    rms_rel = 100 * np.std(E[n0:] / E0)
    return drift_pct, rms_rel, E / E0


def analisis_deriva_energia(
    dt_vals=(5e-9, 1e-8, 2e-8, 5e-8),
    B0=0.1, pasos=5000, radio=0.05,
    carpeta=GUARDAR_DIR,
):
    """
    Barrido en dt: mide deriva relativa de E_cin en simulación sin colisiones.
    Boris debería conservar energía mejor con dt pequeño.
    """
    os.makedirs(carpeta, exist_ok=True)
    filas = []
    curvas = {}

    for i, dt in enumerate(dt_vals):
        drift, rms, E_norm = _corrida_sin_colisiones(B0, dt, pasos)
        filas.append([dt, drift, rms, pasos, B0])
        curvas[dt] = E_norm

    arr = np.array(filas)
    np.savetxt(
        os.path.join(carpeta, "deriva_energia.csv"),
        arr,
        delimiter=",",
        header="dt_s,deriva_total_pct,rms_fluct_pct,pasos,B0_T",
        comments="",
    )

    # Guardar curvas E(t)/E0 para cada dt
    max_len = max(len(v) for v in curvas.values())
    mat = np.full((max_len, len(dt_vals) + 1), np.nan)
    mat[:, 0] = np.arange(max_len)
    for j, dt in enumerate(dt_vals):
        e = curvas[dt]
        mat[: len(e), j + 1] = e
    header = "paso," + ",".join(f"E_norm_dt={d:.1e}" for d in dt_vals)
    np.savetxt(
        os.path.join(carpeta, "deriva_energia_curvas.csv"),
        mat,
        delimiter=",",
        header=header,
        comments="",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax0 = axes[0]
    for dt in dt_vals:
        e = curvas[dt]
        t = np.arange(len(e)) * dt
        ax0.plot(t * 1e6, e, lw=0.8, label=f"dt={dt*1e9:.0f} ns")
    ax0.axhline(1.0, color="k", ls=":", lw=0.5)
    ax0.set_xlabel("t [µs]")
    ax0.set_ylabel("E / E₀")
    ax0.set_title("Energía cinética normalizada (sin colisiones)")
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.bar(
        range(len(dt_vals)),
        arr[:, 1],
        tick_label=[f"{d*1e9:.0f}" for d in dt_vals],
        color="steelblue", alpha=0.85,
    )
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_xlabel("dt [ns]")
    ax1.set_ylabel("Deriva total [%]")
    ax1.set_title("Deriva de energía vs. paso temporal")

    fig.savefig(os.path.join(carpeta, "deriva_energia.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Deriva energía: |deriva| máx = {np.max(np.abs(arr[:, 1])):.3f}%")
    return arr


# ══════════════════════════════════════════════════════════════
#  EJECUCIÓN INTEGRAL
# ══════════════════════════════════════════════════════════════

def ejecutar_validacion_integral(carpeta=GUARDAR_DIR):
    print("=== Semana 14: validación física integral ===")
    os.makedirs(carpeta, exist_ok=True)

    larmor = validar_larmor(carpeta=carpeta)
    T_teo, T_num, err_tc = validar_ciclotron(carpeta=carpeta)
    tau_arr = validar_tau_vs_bohm(carpeta=carpeta)
    deriva = analisis_deriva_energia(carpeta=carpeta)

    resumen = {
        "larmor_error_max_pct": float(larmor[:, 4].max()),
        "ciclotron_error_pct": float(err_tc) if np.isfinite(err_tc) else None,
        "tau_vs_bohm_ratio_medio": float(np.nanmean(tau_arr[:, 3])),
        "deriva_energia_max_pct": float(np.max(np.abs(deriva[:, 1]))),
        "deriva_energia_rms_max_pct": float(np.max(deriva[:, 2])),
        "notas": (
            "τ_sim > τ_Bohm es esperado: Bohm es cota inferior pesimista. "
            "Deriva de E debe decrecer al reducir dt si Boris está bien resuelto."
        ),
    }
    with open(os.path.join(carpeta, "resumen_validacion.json"), "w",
              encoding="utf-8") as f:
        json.dump(resumen, f, indent=2)

    print("\n--- Resumen ---")
    for k, v in resumen.items():
        if k != "notas":
            print(f"  {k}: {v}")
    print(f"\nTodos los datos en: {carpeta}/")
    return resumen


if __name__ == "__main__":
    ejecutar_validacion_integral()
