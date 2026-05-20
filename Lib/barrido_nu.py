"""
barrido_nu.py
=============
Semana 12 del cronograma: análisis de sensibilidad - variación de la
frecuencia de colisión nu_colision y su efecto en el tiempo de confinamiento.

Genera figura 2x2:
  [0,0] tau vs. nu  (escala lineal)
  [0,1] tau vs. nu  (log-log + ajuste potencia)
  [1,0] Curvas de decaimiento N(t) por nu
  [1,1] Fracción de partículas que escapan vs. nu
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(__file__))
from particulas  import Particula
from colisiones  import ColisionEstocastica, velocidad_inicial_mb
from contenedor  import ContenedorCilindrico
import montecarlo as mc
from motor_lite  import motor_lite, campo_B_solenoide_vec, campo_E_cero_vec

# ══════════════════════════════════════════════════
#  CONFIGURACIÓN
# ══════════════════════════════════════════════════
NU_VALS       = [10, 50, 100, 500, 1000, 5000, 10000, 50000]  # Hz
B0            = 0.1       # T  (campo fijo, el del barrido anterior)
N_PARTICULAS  = 40
PASOS         = 10_000
DT            = 1e-8
N_SEMILLAS    = 2
M_PAR = 1.673e-27; Q_PAR = 1.602e-19; T_PL = 1e4
RADIO = 0.01; ALTURA = 0.02
GUARDAR_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "barrido_nu")

BG   = "#0d1117"
CMAP = plt.cm.plasma

plt.rcParams.update({"font.family":"DejaVu Sans","axes.facecolor":BG,
    "figure.facecolor":BG,"axes.edgecolor":"#8888aa",
    "axes.labelcolor":"white","xtick.color":"#8888aa","ytick.color":"#8888aa",
    "text.color":"white","grid.color":"#2a2a4a","grid.alpha":0.5})

def _build(cont, rng, seed_base, nu):
    parts, motors = [], []
    for i in range(N_PARTICULAS):
        x0 = cont.posicion_aleatoria(rng)
        v0 = velocidad_inicial_mb(M_PAR, T_PL, rng=rng)
        p  = Particula(i, q=Q_PAR, m=M_PAR, x0=x0, v0=v0)
        c  = ColisionEstocastica(nu=nu, m=M_PAR, T=T_PL, dt=DT,
                                  seed=seed_base+i)
        parts.append(p); motors.append(c)
    return parts, motors

def _corrida(nu, seed):
    rng  = np.random.default_rng(seed)
    cont = ContenedorCilindrico(radio=RADIO, altura=ALTURA)
    parts, motors = _build(cont, rng, seed*1000, nu)
    fn_B = lambda X: campo_B_solenoide_vec(X, B0=B0, radio=RADIO)
    te, _ = motor_lite(PASOS, parts, motors, campo_E_cero_vec, fn_B,
                       DT, cont, registrar_energia=False, verbose=False)
    return mc.calcular_tau(te, N_PARTICULAS, DT, PASOS)

def curva_decaimiento_nu(stats_list, nu_val):
    """Reconstruye N_conf(t) desde t_escape_arr de una lista de stats."""
    dt_us = DT * 1e6
    t_max_us = PASOS * dt_us
    t_arr = np.arange(0, t_max_us, dt_us)
    N_conf = np.zeros(len(t_arr))
    total  = 0
    for s in stats_list:
        te = np.array(s.get("t_escape_arr", []))
        total += N_PARTICULAS
        for t_step in range(len(t_arr)):
            t_us = t_step * dt_us
            N_conf[t_step] += np.sum(te > t_step)
    return t_arr, N_conf / total

def barrido():
    os.makedirs(GUARDAR_DIR, exist_ok=True)
    print("\n" + "="*46)
    print("  BARRIDO PARAMÉTRICO: tau(nu_colision)")
    print("="*46)
    print(f"\n  nu: {len(NU_VALS)} valores x {N_SEMILLAS} semillas\n")

    resultados = []
    stats_por_nu = {}

    total = len(NU_VALS) * N_SEMILLAS
    idx   = 0
    for nu in NU_VALS:
        taus, stats_l = [], []
        for seed in range(N_SEMILLAS):
            idx += 1
            print(f"  [{idx:3d}/{total}] nu={nu:.0f} Hz  semilla={seed} ...",
                  end="", flush=True)
            s = _corrida(nu, seed)
            taus.append(s["tau"])
            stats_l.append(s)
            print(f" tau={s['tau']*1e6:.2f} us  esc={s['n_escapadas']}/{N_PARTICULAS}")
        tau_m = np.mean(taus)
        tau_s = np.std(taus)
        frac  = np.mean([s["n_escapadas"]/N_PARTICULAS for s in stats_l])
        resultados.append({"nu": nu, "tau": tau_m, "std": tau_s, "frac": frac})
        stats_por_nu[nu] = stats_l

    # CSV
    ruta_csv = os.path.join(GUARDAR_DIR, "barrido_nu_resultados.csv")
    with open(ruta_csv, "w") as f:
        f.write("nu_Hz,tau_medio_s,tau_std_s,fraccion_escapada\n")
        for r in resultados:
            f.write(f"{r['nu']},{r['tau']:.6e},{r['std']:.6e},{r['frac']:.4f}\n")
    print(f"\n  CSV -> {ruta_csv}")

    # Datos para graficar
    nu_arr  = np.array([r["nu"]  for r in resultados])
    tau_arr = np.array([r["tau"] for r in resultados]) * 1e6  
    std_arr = np.array([r["std"] for r in resultados]) * 1e6
    frac_arr= np.array([r["frac"] for r in resultados])


    try:
        log_nu  = np.log10(nu_arr)
        log_tau = np.log10(tau_arr)
        coeffs  = np.polyfit(log_nu, log_tau, 1)
        alpha, log_A = coeffs
        A = 10**log_A
        nu_fit = np.logspace(np.log10(nu_arr.min()), np.log10(nu_arr.max()), 200)
        tau_fit = A * nu_fit**alpha
        fit_ok  = True
    except Exception:
        fit_ok = False

    # ═══════════════════════════════════════════════════
    #  FIGURA
    # ═══════════════════════════════════════════════════
    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    fig.suptitle(
        f"Efecto de la Frecuencia de Colisión ν sobre el Confinamiento\n"
        f"N={N_PARTICULAS} partículas | B₀={B0} T | {PASOS} pasos | dt={DT:.0e} s",
        color="white", fontsize=13, fontweight="bold", y=0.99)

    gs = GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.38)

    # [0,0] Lineal
    ax00 = fig.add_subplot(gs[0,0], facecolor=BG)
    ax00.errorbar(nu_arr, tau_arr, yerr=std_arr,
                  fmt="o-", color="#00bfff", ecolor="#ff6b6b",
                  lw=1.5, ms=6, capsize=4, label="τ medio ± σ")
    if fit_ok:
        ax00.plot(nu_fit, tau_fit, "--", color="#aaffaa", lw=1.5,
                  label=f"Ajuste: τ ∝ ν^{alpha:.2f}")
    ax00.set_xlabel("ν_colisión [Hz]")
    ax00.set_ylabel("τ [μs]")
    ax00.set_title("Tiempo de confinamiento τ vs. ν", color="white")
    ax00.legend(fontsize=8, framealpha=0.3, facecolor=BG,
                edgecolor="#8888aa", labelcolor="white")
    ax00.grid(True); ax00.tick_params(colors="#8888aa")
    y_lo = max(0, tau_arr.min()-std_arr.max()-0.5)
    y_hi = tau_arr.max()+std_arr.max()+0.5
    ax00.set_ylim(y_lo, y_hi)

    # [0,1] Log-log
    ax01 = fig.add_subplot(gs[0,1], facecolor=BG)
    ax01.errorbar(nu_arr, tau_arr, yerr=std_arr,
                  fmt="o", color="#00bfff", ecolor="#ff6b6b",
                  ms=6, capsize=4, label="tau simulado")
    if fit_ok:
        ax01.plot(nu_fit, tau_fit, "--", color="#aaffaa", lw=1.5,
                  label=f"tau ~ ν^{alpha:.2f}")
    ax01.set_xscale("log"); ax01.set_yscale("log")
    ax01.set_xlabel("ν_colisión [Hz]")
    ax01.set_ylabel("τ [μs]")
    ax01.set_title("τ(ν) en escala log-log", color="white")
    ax01.legend(fontsize=8, framealpha=0.3, facecolor=BG,
                edgecolor="#8888aa", labelcolor="white")
    ax01.grid(True, which="both"); ax01.tick_params(colors="#8888aa")

    # [1,0] Decaimiento N(t) por nu (usar semilla 0 únicamente)
    ax10 = fig.add_subplot(gs[1,0], facecolor=BG)
    colors_nu = CMAP(np.linspace(0.15, 0.95, len(NU_VALS)))
    dt_us = DT * 1e6
    t_arr = np.arange(0, PASOS * dt_us, dt_us)
    for i_nu, nu in enumerate(NU_VALS):
        s_list = stats_por_nu[nu]
        # Solo semilla 0
        te_all = []
        for s in s_list:
            te_all.extend(s.get("t_escape_arr", []))
        te_all = np.array(te_all)
        N_total = N_PARTICULAS * len(s_list)
        N_conf  = np.array([np.sum(te_all > k) for k in range(len(t_arr))])
        ax10.plot(t_arr, N_conf / N_total,
                  color=colors_nu[i_nu], lw=1.2, alpha=0.85,
                  label=f"ν={nu:.0f} Hz")
    ax10.axhline(1/np.e, color="white", lw=0.8, ls="--", alpha=0.4, label="1/e")
    ax10.set_xlabel("Tiempo [μs]")
    ax10.set_ylabel("N confinadas / N total")
    ax10.set_title("Decaimiento N_conf(t) por ν", color="white")
    ax10.legend(fontsize=6, framealpha=0.3, facecolor=BG,
                edgecolor="#8888aa", labelcolor="white", ncol=2)
    ax10.grid(True); ax10.tick_params(colors="#8888aa")

    # [1,1] Fracción escapada vs nu
    ax11 = fig.add_subplot(gs[1,1], facecolor=BG)
    ax11.semilogx(nu_arr, frac_arr*100, "s-", color="#ffd166", lw=1.5, ms=7)
    ax11.axhline(100, color="white", ls="--", lw=0.8, alpha=0.4)
    ax11.set_xlabel("ν_colisión [Hz]")
    ax11.set_ylabel("Fracción escapada [%]")
    ax11.set_title("Partículas que escapan vs. ν", color="white")
    ax11.set_ylim(0, 110)
    ax11.grid(True, which="both"); ax11.tick_params(colors="#8888aa")

    ruta = os.path.join(GUARDAR_DIR, "barrido_nu.png")
    fig.savefig(ruta, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  [Barrido nu] Figura -> {ruta}")

if __name__ == "__main__":
    barrido()
