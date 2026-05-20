"""
barrido_B0.py
=============
Semana 11 del cronograma: anlisis paramétrico del campo magnético.

Corre la simulación (motor_lite) para una lista de valores de B_0 y genera:
  - Gráfica t(B_0)  con barras de error (std)
  - Gráfica de fracción de partículas confinadas vs. B_0
  - CSV con todos los resultados
  - (Opcional) gráfica de energía cinética para cada B_0

Uso
---
    python barrido_B0.py

Parámetros configurables en la sección CONFIGURACIÓN más abajo.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # backend no-interactivo: guarda PNG sin bloquear
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(__file__))

from particulas   import Particula
from colisiones   import ColisionEstocastica, velocidad_inicial_mb
from contenedor   import ContenedorCilindrico
import montecarlo as mc
from motor_lite   import motor_lite, campo_B_solenoide_vec, campo_E_cero_vec


# ══════════════════════════════════════════════════════════════
#  CONFIGURACIÓN DEL BARRIDO  (calibrada con _diagnostico_escalas.py)
# ══════════════════════════════════════════════════════════════

# Valores de B a explorar [T]
B0_VALS = [0.005, 0.008, 0.012, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

# Parámetros de simulación
N_PARTICULAS  = 40          
PASOS         = 10_000      
                            
DT            = 1e-8        
N_SEMILLAS    = 2           

# Parámetros físicos  (PROTÓN a T baja)
M_PARTICULA   = 1.673e-27   # masa protón [kg]
Q_PARTICULA   = 1.602e-19   # carga protón [C]
T_PLASMA      = 1e4         # T=1e4 K → v_th ≈ 9084 m/s
NU_COLISION   = 500.0       # [Hz]  P_col/paso = 5e-6 (muy baja)

# Geometría  (cilindro de 1 cm de radio)
RADIO         = 0.01        # [m]  = 1 cm
ALTURA        = 0.02        # [m]  = 2 cm

# Salida
ESCALA_KEY    = "us"        # microsegundos
GUARDAR_DIR   = os.path.join(os.path.dirname(__file__), "..", "data",
                             "barrido_B0")

ESCALAS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}



# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def _construir_particulas(contenedor, rng, seed_base):
    """Crea N_PARTICULAS con posiciones y velocidades MB."""
    particulas, motores = [], []
    for i in range(N_PARTICULAS):
        x0  = contenedor.posicion_aleatoria(rng)
        v0  = velocidad_inicial_mb(M_PARTICULA, T_PLASMA, rng=rng)
        p   = Particula(i, q=Q_PARTICULA, m=M_PARTICULA, x0=x0, v0=v0)
        col = ColisionEstocastica(
            nu=NU_COLISION, m=M_PARTICULA, T=T_PLASMA,
            dt=DT, seed=seed_base + i
        )
        particulas.append(p)
        motores.append(col)
    return particulas, motores


def _una_corrida(B0, seed):
    """Corre motor_lite para un B₀ y una semilla. Devuelve stats + E_cin."""
    rng = np.random.default_rng(seed)
    cont = ContenedorCilindrico(radio=RADIO, altura=ALTURA)
    particulas, motores = _construir_particulas(cont, rng, seed * 1000)

    # Campos VECTORIZADOS: reciben X(N,3) → devuelven (N,3)  (sin loop Python)
    fn_E = campo_E_cero_vec
    fn_B = lambda X: campo_B_solenoide_vec(X, B0=B0, radio=RADIO)

    tiempos_escape, E_cin_hist = motor_lite(
        pasos             = PASOS,
        particulas        = particulas,
        motores_colision  = motores,
        fn_E              = fn_E,
        fn_B              = fn_B,
        dt                = DT,
        contenedor        = cont,
        registrar_energia = True,
        verbose           = False,
    )
    stats = mc.calcular_tau(tiempos_escape, N_PARTICULAS, DT, PASOS)
    return stats, E_cin_hist


# ══════════════════════════════════════════════════════════════
#  BARRIDO PRINCIPAL
# ══════════════════════════════════════════════════════════════

def correr_barrido():
    """
    Para cada B0 en B0_VALS: corre N_SEMILLAS veces y promedia.
    Devuelve (B0s, tau_medios, tau_stds, frac_confs, E_cin_por_B0, te_por_B0)
    """
    tau_medios   = []
    tau_stds     = []
    frac_confs   = []
    E_cin_ultimo = {}   # {B0: E_cin_historia de la última semilla}
    te_ultimo    = {}   # {B0: tiempos_escape dict de la última semilla}

    total = len(B0_VALS) * N_SEMILLAS
    done  = 0

    print(f"\n  Barrido de B0: {len(B0_VALS)} valores x {N_SEMILLAS} "
          f"semillas = {total} corridas")
    print(f"  Particulas: {N_PARTICULAS}  |  Pasos: {PASOS}  |  "
          f"dt: {DT:.1e} s\n")

    for B0 in B0_VALS:
        taus_semilla  = []
        fracs_semilla = []

        for s in range(N_SEMILLAS):
            done += 1
            print(f"  [{done:>3}/{total}] B0={B0:.3f} T  semilla={s} ...",
                  end=" ", flush=True)
            stats, E_cin_hist = _una_corrida(B0, seed=s)
            taus_semilla.append(stats["tau_medio"])
            fracs_semilla.append(stats["fraccion_esc"])
            E_cin_ultimo[B0] = E_cin_hist
            # Reconstruir tiempos_escape a partir de t_escape_arr
            # (guardamos solo de la última semilla para el panel de decaimiento)
            te_ultimo[B0] = {
                i: float(stats["t_escape_arr"][i])
                for i in range(N_PARTICULAS)
                if stats["t_escape_arr"][i] < PASOS * DT
            }
            n_esc = stats["n_escaparon"]
            print(f"tau={stats['tau_medio']/ESCALAS[ESCALA_KEY]:.2f} {ESCALA_KEY}  "
                  f"esc={n_esc}/{N_PARTICULAS}")

        tau_medios.append(np.mean(taus_semilla))
        tau_stds.append(np.std(taus_semilla))
        frac_confs.append(np.mean(fracs_semilla))

    return (np.array(B0_VALS), np.array(tau_medios),
            np.array(tau_stds), np.array(frac_confs),
            E_cin_ultimo, te_ultimo)


# ══════════════════════════════════════════════════════════════
#  GRÁFICAS
# ══════════════════════════════════════════════════════════════

def graficar_barrido(B0s, tau_medios, tau_stds, frac_confs,
                     E_cin_por_B0, te_por_B0, guardar_dir):
    """
    Figura 2x2:
      [0,0] tau(B0) lineal  - solo datos simulados (sin Bohm), ylim correcto
      [0,1] tau(B0) log-log - con ajuste de ley de potencia + Bohm
      [1,0] Curvas N_confinadas(t) solapadas para cada B0 (reemplaza fraccion=1)
      [1,1] Energia cinetica normalizada vs. tiempo por B0
    """
    factor  = ESCALAS[ESCALA_KEY]
    BG      = "#0d0d1a"
    AX      = "#12122a"
    PALETA  = plt.cm.plasma(np.linspace(0.15, 0.9, len(B0s)))

    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    fig.suptitle(f"Efecto del Campo Magnético B0 sobre el Confinamiento\n"
                 f"N={N_PARTICULAS} partículas | {PASOS} pasos | "
                 f"dt={DT:.1e} s | T={T_PLASMA:.0e} K",
                 color="white", fontsize=11, y=0.98)

    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    def _ax_style(ax, titulo, xlabel, ylabel):
        ax.set_facecolor(AX)
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.set_title(titulo, color="white", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    tau_plot = tau_medios / factor
    std_plot = tau_stds   / factor

    # ── Panel 0,0 — tau(B0) lineal (solo datos, sin Bohm) ────
    ax0 = fig.add_subplot(gs[0, 0])
    _ax_style(ax0, f"Tiempo de confinamiento t vs. B0",
              "B₀ [T]", f"τ [{ESCALA_KEY}]")
    ax0.errorbar(B0s, tau_plot, yerr=std_plot,
                 fmt='o-', color="#4a9eff", linewidth=1.8,
                 markersize=7, capsize=4, ecolor="#ff6b6b",
                 label="t medio ± σ")

    # Ajuste ley de potencia
    mask = tau_medios > 0
    if mask.sum() >= 2:
        log_B  = np.log(B0s[mask])
        log_t  = np.log(tau_medios[mask])
        coefs  = np.polyfit(log_B, log_t, 1)
        exp_B  = coefs[0]
        B_fit  = np.linspace(B0s[0], B0s[-1], 200)
        t_fit  = np.exp(coefs[1]) * B_fit**exp_B / factor
        ax0.plot(B_fit, t_fit, '--', color="#50fa7b", linewidth=1.3,
                 label=f"Ajuste: τ ∝ B₀^{exp_B:.2f}")

    # ylim basado en DATOS, no en Bohm
    y_max = float(np.max(tau_plot + std_plot)) * 1.35
    ax0.set_ylim(0, max(y_max, 1e-3))
    ax0.set_xlim(left=0)
    ax0.legend(fontsize=8, facecolor="#1a1a2e",
               labelcolor="white", framealpha=0.5)

    # ── Panel 0,1 — τ(B₀) log-log + Bohm ──────────────
    ax1 = fig.add_subplot(gs[0, 1])
    _ax_style(ax1, "τ(B₀) en escala log-log",
              "B₀ [T]", f"τ [{ESCALA_KEY}]")
    ax1.errorbar(B0s, tau_plot, yerr=std_plot,
                 fmt='o', color="#4a9eff", markersize=7,
                 capsize=4, ecolor="#ff6b6b", label="tau simulado")
    if mask.sum() >= 2:
        ax1.plot(B_fit, t_fit, '--', color="#50fa7b", linewidth=1.3,
                 label=f"tau ~ B0^{exp_B:.2f}")

    # Bohm SOLO en log-log (escala compatible, no destruye el eje)
    tau_bohm_ll = np.array([
        mc.tau_bohm(RADIO, float(b), T_PLASMA, abs(Q_PARTICULA))
        for b in B0s
    ]) / factor
    ax1.plot(B0s, tau_bohm_ll, ':', color="#ffb86c", linewidth=1.3,
             label="tau_Bohm (ref.)")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend(fontsize=8, facecolor="#1a1a2e",
               labelcolor="white", framealpha=0.5)

    # ── Panel 1,0 — Curvas N_confinadas(t) por B0 ──────────────
    ax2 = fig.add_subplot(gs[1, 0])
    _ax_style(ax2, "Decaimiento N_confinadas(t) por B0",
              f"Tiempo [{ESCALA_KEY}]", "N confinadas / N total")

    for idx, B0 in enumerate(B0s):
        te = te_por_B0.get(float(B0), {})
        t_arr_b, N_arr_b = mc.curva_decaimiento(te, N_PARTICULAS, DT, PASOS)
        t_norm = t_arr_b / factor
        N_norm = N_arr_b / N_PARTICULAS
        # Recortar hasta que llegue a 0 o al final
        ultimo = np.where(N_norm > 0)[0]
        fin    = int(ultimo[-1]) + 2 if len(ultimo) else len(t_norm)
        ax2.plot(t_norm[:fin], N_norm[:fin],
                 color=PALETA[idx], linewidth=1.6,
                 label=f"B0={B0:.3f}T")

    ax2.axhline(np.exp(-1), color="#aaaacc", linestyle='--',
                linewidth=0.9, alpha=0.7, label="1/e")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=7, facecolor="#1a1a2e",
               labelcolor="white", framealpha=0.5, ncol=2)

    ax3 = fig.add_subplot(gs[1, 1])
    _ax_style(ax3, "Energía cinética normalizada E(t)/E0",
              f"Tiempo [{ESCALA_KEY}]", "E_cin(t) / E_cin(0)")

    t_arr = np.arange(PASOS) * DT / factor
    for idx, B0 in enumerate(B0s):
        E_hist = E_cin_por_B0.get(B0)
        if E_hist and E_hist[0] > 0:
            E_arr  = np.array(E_hist)
            E_norm = E_arr / E_arr[0]
            ax3.plot(t_arr[:len(E_norm)], E_norm,
                     color=PALETA[idx], linewidth=0.9,
                     alpha=0.85, label=f"B0={B0:.2f}T")

    ax3.axhline(1.0, color="#555577", linestyle='--', linewidth=0.8)
    ax3.legend(fontsize=6, facecolor="#1a1a2e", labelcolor="white",
               framealpha=0.5, ncol=2)

    # ── Guardar ───────────────────────────────────────────────
    os.makedirs(guardar_dir, exist_ok=True)
    ruta = os.path.join(guardar_dir, "barrido_B0.png")
    fig.savefig(ruta, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)   # liberar memoria, no bloquear
    print(f"\n  [Barrido] Figura guardada -> {ruta}")


# ══════════════════════════════════════════════════════════════
#  GUARDAR CSV
# ══════════════════════════════════════════════════════════════

def guardar_csv(B0s, tau_medios, tau_stds, frac_confs, guardar_dir):
    """Exporta los resultados del barrido a CSV."""
    import csv
    os.makedirs(guardar_dir, exist_ok=True)
    ruta = os.path.join(guardar_dir, "barrido_B0_resultados.csv")
    with open(ruta, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["B0_T", "tau_medio_s", "tau_std_s",
                    "fraccion_escapada", "tau_bohm_s"])
        for i, B0 in enumerate(B0s):
            tau_b = mc.tau_bohm(RADIO, B0, T_PLASMA, abs(Q_PARTICULA))
            w.writerow([
                round(B0, 4),
                round(tau_medios[i], 6),
                round(tau_stds[i],   6),
                round(frac_confs[i], 4),
                round(tau_b, 6),
            ])
    print(f"  [Barrido] CSV guardado    -> {ruta}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n==========================================")
    print("  BARRIDO PARAMETRICO: tau(B0)")
    print("==========================================")

    B0s, tau_medios, tau_stds, frac_confs, E_cin_por_B0, te_por_B0 = correr_barrido()

    guardar_csv(B0s, tau_medios, tau_stds, frac_confs, GUARDAR_DIR)

    print("\n  Resultados:")
    factor = ESCALAS[ESCALA_KEY]
    print(f"  {'B0 [T]':>8}  {'tau ['+ESCALA_KEY+']':>12}  "
          f"{'std':>10}  {'frac_esc':>10}")
    print(f"  {'-'*50}")
    for i, B0 in enumerate(B0s):
        print(f"  {B0:>8.3f}  "
              f"{tau_medios[i]/factor:>12.4f}  "
              f"{tau_stds[i]/factor:>10.4f}  "
              f"{frac_confs[i]:>10.3f}")

    graficar_barrido(B0s, tau_medios, tau_stds, frac_confs,
                     E_cin_por_B0, te_por_B0, GUARDAR_DIR)
