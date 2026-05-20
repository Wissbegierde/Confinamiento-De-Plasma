"""
montecarlo.py
=============
Semanas 9–11 del cronograma: análisis estadístico Monte Carlo del confinamiento.

Funciones principales
---------------------
calcular_tau          : media, mediana, std, percentiles de tiempos de escape
fraccion_confinadas   : qué % sobrevivió hasta el final
curva_decaimiento     : N_confinadas(t) paso a paso
tau_bohm              : tiempo de confinamiento teórico de Bohm (referencia)
graficar_resultados   : figura de 4 paneles (decaimiento, histograma, energía,
                        impactos en pared)
guardar_resultados    : exporta estadísticas a CSV
"""

import os
import numpy as np
import matplotlib
from contenedor import ContenedorTokamak, ContenedorCilindrico, ContenedorEsferico
# Elegir un backend que exista de verdad.
# 'matplotlib.use("Qt5Agg")' puede "funcionar" aunque no tengas PyQt/PySide,
# y fallar recién cuando se crea la primera figura. Preferimos TkAgg (GUI) o Agg (headless).
for _b in ("TkAgg", "Agg"):
    try:
        matplotlib.use(_b, force=True)
        break
    except Exception:
        continue
import matplotlib.pyplot as plt
from scipy.stats import expon


# ══════════════════════════════════════════════════════════════
#  1. ESTADÍSTICAS DE CONFINAMIENTO
# ══════════════════════════════════════════════════════════════

def calcular_tau(tiempos_escape: dict, n_total: int, dt: float, pasos: int
                 ) -> dict:
    """
    Calcula las estadísticas de confinamiento a partir de los tiempos de escape.

    Las partículas que NO aparecen en tiempos_escape sobrevivieron hasta el
    final; se les asigna t_escape = pasos * dt (cota inferior de su τ real).

    Parámetros
    ----------
    tiempos_escape : dict {id_particula: t_escape [s]}  ← devuelto por motor
    n_total        : número total de partículas simuladas
    dt             : paso de tiempo [s]
    pasos          : pasos totales de la simulación

    Retorna
    -------
    stats : dict con claves:
        n_total, n_escaparon, n_confinadas,
        t_escape_arr  (array con todos los tiempos, censurados incluidos),
        tau_medio, tau_mediana, tau_std,
        p25, p75, p95   (percentiles)
    """
    t_max = pasos * dt

    # Construir array completo: escapadas + censuradas (confinadas al final)
    t_arr = np.array([
        tiempos_escape.get(i, t_max)   # censuradas reciben t_max
        for i in range(n_total)
    ])

    n_esc = len(tiempos_escape)
    n_con = n_total - n_esc

    return {
        "n_total"      : n_total,
        "n_escaparon"  : n_esc,
        "n_confinadas" : n_con,
        "t_escape_arr" : t_arr,
        "t_esc_solo"   : np.array(list(tiempos_escape.values()), dtype=float),
        "ids_escapados": set(tiempos_escape.keys()),
        "tau_medio"    : t_arr.mean(),
        "tau_mediana"  : np.median(t_arr),
        "tau_std"      : t_arr.std(),
        "p25"          : np.percentile(t_arr, 25),
        "p75"          : np.percentile(t_arr, 75),
        "p95"          : np.percentile(t_arr, 95),
        "fraccion_esc" : n_esc / n_total,
        "tau_esc_medio": float(np.mean(list(tiempos_escape.values()))) if n_esc else float("nan"),
        "tau_cens"     : tau_exponencial_censurado(tiempos_escape, n_total, t_max),
        "frac_censurada": n_con / n_total,
    }


def fraccion_confinadas(tiempos_escape: dict, n_total: int) -> float:
    """Fracción de partículas que NO escaparon."""
    return 1.0 - len(tiempos_escape) / n_total


def tau_exponencial_censurado(tiempos_escape: dict, n_total: int, t_max: float) -> float:
    """
    MLE de τ con censura por derecha (partículas que no escaparon hasta t_max).
    λ = n_esc / (Σ t_i + n_cens · t_max),  τ = 1/λ
    Coincide con N(t) = N₀ exp(-t/τ) cuando la pérdida es exponencial.
    """
    n_esc = len(tiempos_escape)
    if n_esc == 0:
        return float("inf")
    t_sum = float(sum(tiempos_escape.values()))
    n_cens = n_total - n_esc
    lam = n_esc / (t_sum + n_cens * t_max)
    return 1.0 / lam if lam > 0 else float("inf")


# ══════════════════════════════════════════════════════════════
#  2. CURVA DE DECAIMIENTO  N_confinadas(t)
# ══════════════════════════════════════════════════════════════

def curva_decaimiento(tiempos_escape: dict, n_total: int,
                      dt: float, pasos: int) -> tuple:
    """
    Construye N_confinadas(t) paso a paso.

    Retorna
    -------
    t_arr      : np.ndarray de tiempos [s]  (longitud = pasos+1)
    N_arr      : np.ndarray de partículas confinadas en cada instante
    """
    t_arr = np.arange(pasos + 1) * dt
    N_arr = np.zeros(pasos + 1, dtype=int)
    N_arr[0] = n_total

    # Ordenar escapes por tiempo
    escapes_sorted = sorted(tiempos_escape.values())
    idx_t = 0
    n_restantes = n_total

    for paso in range(1, pasos + 1):
        t = paso * dt
        while idx_t < len(escapes_sorted) and escapes_sorted[idx_t] <= t:
            n_restantes -= 1
            idx_t += 1
        N_arr[paso] = n_restantes

    return t_arr, N_arr


# ══════════════════════════════════════════════════════════════
#  3. MODELO TEÓRICO DE BOHM
# ══════════════════════════════════════════════════════════════

def tau_bohm(radio: float, B0: float, T_plasma: float,
             q: float = 1.602e-19) -> float:
    """
    Tiempo de confinamiento de Bohm (difusión de Bohm).

        D_Bohm = (1/16) · kT / (q · B)   [m²/s]
        τ_Bohm = a² / D_Bohm              [s]

    donde a es el radio del contenedor.

    Este es el límite inferior (peor caso) del confinamiento magnético clásico.
    El confinamiento clásico (Spitzer) puede ser órdenes de magnitud mayor.

    Parámetros
    ----------
    radio    : radio del contenedor [m]
    B0       : campo magnético [T]
    T_plasma : temperatura del plasma [K]
    q        : carga de la partícula [C]

    Retorna
    -------
    tau [s]
    """
    kB = 1.380649e-23
    D_bohm = (kB * T_plasma) / (16.0 * q * B0)   # m²/s
    return radio**2 / D_bohm


def radio_larmor(m: float, v_perp: float, q: float, B0: float) -> float:
    """Radio de Larmor: r_L = m·v_⊥ / (|q|·B)"""
    return m * v_perp / (abs(q) * B0)


# ══════════════════════════════════════════════════════════════
#  4. GRÁFICAS
# ══════════════════════════════════════════════════════════════

def _n_bins_histograma(n_esc: int, n_frames: int) -> int:
    """
    Bins legibles según cuántas partículas escaparon.
    Con pocos escapes → pocos bins (~√N); con muchos → hasta 1000.
    """
    if n_esc <= 0:
        return 20
    por_eventos = max(20, min(1000, int(np.sqrt(n_esc) * 6), n_esc // 2))
    por_tiempo = min(1000, max(20, (n_frames - 1) // 50))
    return int(min(por_tiempo, por_eventos))


def _posiciones_impacto(particulas, ids_escapados):
    """Posición final de partículas que escaparon (registradas en tiempos_escape)."""
    posiciones = []
    for p in particulas:
        if p.id in ids_escapados:
            posiciones.append(np.asarray(p.x, dtype=float))
    return posiciones


def graficar_resultados(stats: dict, t_arr: np.ndarray, N_arr: np.ndarray,
                        E_cin_historia: list = None,
                        particulas=None, contenedor=None,
                        tiempos_escape: dict = None,
                        escala_key: str = "us",
                        tau_ref: float = None,
                        guardar_dir: str = None,
                        mostrar: bool = None):
    """
    Genera una figura de 2×2 paneles:
      [0,0] Curva N_confinadas(t) + ajuste exponencial
      [0,1] Histograma de tiempos de escape
      [1,0] Energía cinética total vs. tiempo  (si E_cin_historia)
      [1,1] Mapa de impactos en la pared       (si particulas + contenedor)

    Parámetros
    ----------
    stats          : dict devuelto por calcular_tau()
    t_arr          : array de tiempos [s]
    N_arr          : array de N_confinadas en cada instante
    E_cin_historia : lista de E_cin por paso (opcional)
    particulas     : lista de Particula (para el mapa de impactos, opcional)
    contenedor     : objeto ContenedorXxx (opcional)
    escala_key     : 's' | 'ms' | 'us' | 'ns'
    tau_ref        : τ de Bohm teórico para marcar en la gráfica (opcional)
    guardar_dir    : si se proporciona, guarda la figura en ese directorio
    mostrar        : si False, no llama plt.show() (default: True solo sin guardar_dir)
    """
    if mostrar is None:
        mostrar = guardar_dir is None

    # Si no vamos a mostrar ventana, forzar backend no-GUI.
    # Esto evita depender de Qt y permite guardar figuras desde hilos.
    if not mostrar:
        try:
            plt.switch_backend("Agg")
        except Exception:
            pass

    if "t_escape_arr" not in stats:
        raise KeyError(
            "stats debe incluir 't_escape_arr' (usar calcular_tau(), no solo CSV)"
        )

    ESCALAS = {
        "s":  ("s",   1.0),
        "ms": ("ms",  1e-3),
        "us": ("μs",  1e-6),
        "ns": ("ns",  1e-9),
    }
    unidad, factor = ESCALAS.get(escala_key, ("s", 1.0))
    t_plot = t_arr / factor

    n_total   = stats["n_total"]
    n_esc     = stats["n_escaparon"]
    tau_med   = stats["tau_medio"]
    t_escape  = stats["t_escape_arr"]

    # ── Color scheme oscuro ────────────────────────────────────
    BG   = "#0d0d1a"
    AX   = "#12122a"
    CLR1 = "#4a9eff"   # azul
    CLR2 = "#ff6b6b"   # rojo
    CLR3 = "#50fa7b"   # verde
    CLR4 = "#ffb86c"   # naranja

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=BG)
    fig.suptitle("Análisis Monte Carlo — Confinamiento de Plasma",
                 color="white", fontsize=13, y=0.98)

    for ax in axes.flat:
        ax.set_facecolor(AX)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    # ── Panel 0,0 — Curva de decaimiento ──────────────────────
    ax = axes[0, 0]
    ax.plot(t_plot, N_arr, color=CLR1, linewidth=1.8, label="$N_{conf}(t)$")
    ax.set_xlabel(f"Tiempo [{unidad}]")
    ax.set_ylabel("Partículas confinadas")
    ax.set_title("Decaimiento de confinamiento")

    # Ajuste N(t)=N₀ exp(-t/τ) con censura (coincide mejor con la curva azul)
    if n_esc > 0:
        tau_fit = stats.get("tau_cens")
        if (tau_fit is None or not np.isfinite(tau_fit)) and tiempos_escape:
            tau_fit = tau_exponencial_censurado(tiempos_escape, n_total, t_arr[-1])
        if n_esc > 0 and tau_fit is not None and np.isfinite(tau_fit):
            t_fit = np.linspace(0, t_arr[-1], 300)
            N_fit = n_total * np.exp(-t_fit / tau_fit)
            ax.plot(t_fit / factor, N_fit,
                    '--', color=CLR2, linewidth=1.4,
                    label=f"Exp. censurado τ={tau_fit/factor:.3g} {unidad}")
        frac_cens = stats.get("frac_censurada", 0)
        if frac_cens > 0.15:
            ax.text(
                0.02, 0.05,
                f"{100*frac_cens:.0f}% aún confinadas al final",
                transform=ax.transAxes, ha="left", va="bottom",
                color="#aaaacc", fontsize=8,
            )

    t_max_plot = float(t_plot[-1]) if len(t_plot) else 1.0
    ax.set_xlim(0, t_max_plot * 1.05)

    if tau_ref is not None:
        tau_ref_plot = tau_ref / factor
        if tau_ref_plot <= t_max_plot * 2:
            ax.axvline(tau_ref_plot, color=CLR4, linestyle=':', linewidth=1.2,
                       label=f"τ Bohm={tau_ref_plot:.3g} {unidad}")
        else:
            ax.text(
                0.98, 0.95,
                f"τ Bohm = {tau_ref_plot:.2e} {unidad}\n(fuera de escala del eje)",
                transform=ax.transAxes, ha="right", va="top",
                color=CLR4, fontsize=8,
            )

    ax.axhline(n_total * np.exp(-1), color="#555577", linestyle='--',
               linewidth=0.8, alpha=0.7, label="N₀/e")
    ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white",
              framealpha=0.5)
    ax.set_ylim(bottom=0)

      # ── Panel 0,1 — Histograma de tiempos de escape ─────────
    ax = axes[0, 1]
    if n_esc > 0:
        t_esc_solo = stats.get("t_esc_solo")
        if t_esc_solo is None or len(t_esc_solo) == 0:
            if tiempos_escape:
                t_esc_solo = np.array(list(tiempos_escape.values()), dtype=float)
            else:
                t_esc_solo = np.array([
                    v for v in t_escape if v < t_arr[-1] * (1 - 1e-12)
                ])
        t_esc_plot = np.asarray(t_esc_solo, dtype=float) / factor
        t_fin = t_arr[-1] / factor
        n_bins = _n_bins_histograma(n_esc, len(t_arr))
        bin_edges = np.linspace(0.0, t_fin, n_bins + 1)

        counts, bins, patches = ax.hist(
            t_esc_plot, bins=bin_edges,
            color=CLR1, alpha=0.75, edgecolor="#0d0d1a",
            label=f"Tiempos de escape (N={len(t_esc_plot)})"
        )
        if len(t_esc_plot) > 3:
            _, scale_esc = expon.fit(t_esc_plot, floc=0)
            x_fit = np.linspace(bins[0], bins[-1], 200)
            bin_w = bins[1] - bins[0] if len(bins) > 1 else (bins[-1] - bins[0])
            y_fit = len(t_esc_plot) * bin_w * expon.pdf(x_fit, scale=scale_esc)
            ax.plot(x_fit, y_fit, color=CLR2, linewidth=1.8,
                    label=f"PDF exp. solo escapadas (τ={scale_esc:.3g} {unidad})")
            tau_c = stats.get("tau_cens")
            if tau_c is not None and np.isfinite(tau_c):
                ax.text(
                    0.98, 0.95,
                    f"τ censurado = {tau_c/factor:.3g} {unidad}\n"
                    f"(ajusta N_conf, no este histograma)",
                    transform=ax.transAxes, ha="right", va="top",
                    color="#aaaacc", fontsize=7,
                )
        ymax = max(counts) if len(counts) else 1
        ax.set_ylim(0, ymax * 1.15 + 0.5)
    else:
        ax.text(0.5, 0.5, "Ninguna partícula escapó",
                ha="center", va="center", color="white",
                transform=ax.transAxes, fontsize=11)

    ax.set_xlabel(f"Tiempo de escape [{unidad}]")
    ax.set_ylabel("Número de partículas")
    ax.set_title("Histograma de tiempos de escape")
    ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white",
              framealpha=0.5)

    # ── Panel 1,0 — Energía cinética total ──────────────────
    ax = axes[1, 0]
    if E_cin_historia:
        E_arr   = np.array(E_cin_historia)
        E_norm  = E_arr / E_arr[0]
        t_E     = np.arange(len(E_arr)) * (t_arr[-1] / len(t_arr)) / factor

        # Detectar si los datos son sintéticos (oscilan >5% sin tendencia)
        diffs   = np.diff(E_norm)
        monoton = np.mean(diffs <= 0)          # fracción de pasos que decrece
        es_real = monoton > 0.6 or E_norm[-1] < 0.95  # real si decrece claramente

        # Curva bruta (semitransparente) + suavizado
        ventana  = max(1, len(E_arr) // 80)    # ventana media móvil ~1.25%
        E_smooth = np.convolve(E_norm,
                               np.ones(ventana) / ventana, mode="valid")
        t_smooth = t_E[:len(E_smooth)]

        ax.plot(t_E, E_norm, color=CLR3, linewidth=0.6, alpha=0.35)
        ax.plot(t_smooth, E_smooth, color=CLR3, linewidth=1.8,
                label="E_cin (suavizado)")
        ax.axhline(1.0, color="#555577", linestyle="--", linewidth=0.8)

        # Tendencia lineal para ver la deriva
        if len(t_E) > 5:
            coef    = np.polyfit(t_E, E_norm, 1)
            y_trend = np.polyval(coef, [t_E[0], t_E[-1]])
            ax.plot([t_E[0], t_E[-1]], y_trend,
                    color=CLR4, linewidth=1.2, linestyle=":",
                    label="Tendencia lineal")

        ax.set_xlabel(f"Tiempo [{unidad}]")
        ax.set_ylabel("$E_{cin}(t) / E_{cin}(0)$")
        ax.set_title("Energía cinética normalizada")

        drift_pct = 100 * (E_norm[-1] - 1.0)
        color_drift = CLR2 if abs(drift_pct) > 5 else CLR3
        ax.text(0.98, 0.05,
                f"Deriva total: {drift_pct:+.2f}%",
                ha="right", va="bottom", color=color_drift, fontsize=9,
                transform=ax.transAxes)

        if not es_real:
            ax.text(0.5, 0.55,
                    "(datos sintéticos / test)",
                    ha="center", va="center", color="#666688", fontsize=8,
                    style="italic", transform=ax.transAxes)

        ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white",
                  framealpha=0.5)
    else:
        ax.text(0.5, 0.5,
                "Activa registrar_energia=True\nen motor_simulacion()",
                ha="center", va="center", color="#888888",
                transform=ax.transAxes, fontsize=10)
        ax.set_title("Energía cinética (no registrada)")

    # ── Panel 1,1 — Mapa de impactos en la pared ──────────────
    ax = axes[1, 1]
    if particulas is not None and contenedor is not None:
        ids_esc = set(tiempos_escape.keys()) if tiempos_escape else stats.get(
            "ids_escapados", set()
        )
        posiciones = _posiciones_impacto(particulas, ids_esc)

        if posiciones:
            P = np.array(posiciones)
            if isinstance(contenedor, ContenedorTokamak):
                u = np.sqrt(P[:, 0] ** 2 + P[:, 1] ** 2) * 1e3
                v = P[:, 2] * 1e3
                u_edges = np.linspace(
                    max(0, (contenedor.R - contenedor.a) * 1e3),
                    (contenedor.R + contenedor.a) * 1e3,
                    31,
                )
                v_edges = np.linspace(
                    -contenedor.a * 1e3, contenedor.a * 1e3, 25,
                )
                h = ax.hist2d(u, v, bins=[u_edges, v_edges], cmap="hot")
                ax.set_xlabel("z [mm]")
                ax.set_ylabel("R_xy [mm]")
                tit_imp = "Impactos en pared (R_xy–z)"
            else:
                lim_xy = getattr(contenedor, "radio", None)
                if lim_xy is None and hasattr(contenedor, "lim"):
                    lim_xy = float(contenedor.lim.max())
                elif lim_xy is None:
                    lim_xy = np.max(np.abs(P[:, :2])) * 1.1 or 0.01
                h = ax.hist2d(
                    P[:, 0] * 1e3, P[:, 1] * 1e3,
                    bins=25,
                    range=[[-lim_xy * 1e3, lim_xy * 1e3], [-lim_xy * 1e3, lim_xy * 1e3]],
                    cmap="hot",
                )
                ax.set_xlabel("X [mm]")
                ax.set_ylabel("Y [mm]")
                tit_imp = "Mapa de impactos en la pared (XY)"
            cb = plt.colorbar(h[3], ax=ax)
            cb.set_label("N impactos")
            cb.ax.yaxis.label.set_color("white")
            ax.set_title(tit_imp)
        else:
            ax.text(0.5, 0.5, "Sin datos de impacto\n(pasa tiempos_escape)",
                    ha="center", va="center", color="white",
                    transform=ax.transAxes)
            ax.set_title("Mapa de impactos en la pared")
    else:
        ax.text(0.5, 0.5,
                "Pasa particulas y contenedor\npara ver el mapa de impactos",
                ha="center", va="center", color="#888888",
                transform=ax.transAxes, fontsize=10)
        ax.set_title("Mapa de impactos en la pared")

    # ── Cuadro de estadísticas ─────────────────────────────────
    tau_esc = stats.get("tau_esc_medio", float("nan"))
    tau_cens = stats.get("tau_cens", float("nan"))
    txt = (f"N total:  {n_total}\n"
           f"Escaparon: {n_esc} ({100*stats['fraccion_esc']:.1f}%)\n"
           f"τ medio (c/confinadas): {tau_med/factor:.4g} {unidad}\n"
           f"τ escapadas: {tau_esc/factor:.4g} {unidad}\n"
           f"τ exp. censurado: {tau_cens/factor:.4g} {unidad}\n"
           f"σ: {stats['tau_std']/factor:.4g} {unidad}")
    fig.text(0.01, 0.01, txt, color="#aaaacc", fontsize=8,
             family="monospace", va="bottom")

    plt.tight_layout(rect=[0, 0.07, 1, 0.96])

    if guardar_dir:
        os.makedirs(guardar_dir, exist_ok=True)
        ruta = os.path.join(guardar_dir, "analisis_montecarlo.png")
        fig.savefig(ruta, dpi=300, bbox_inches="tight", facecolor=BG)
        print(f"  [MC] Figura guardada -> {ruta}")

    if mostrar:
        plt.show()
    else:
        plt.close(fig)


# ══════════════════════════════════════════════════════════════
#  5. EXPORTAR ESTADÍSTICAS A CSV
# ══════════════════════════════════════════════════════════════

def guardar_resultados(stats: dict, t_arr: np.ndarray, N_arr: np.ndarray,
                       carpeta: str = "data"):
    """
    Guarda dos archivos CSV:
    - montecarlo_stats.csv   : estadísticas globales (una fila)
    - montecarlo_decaimiento.csv : curva N_confinadas(t)
    """
    os.makedirs(carpeta, exist_ok=True)

    # Estadísticas globales
    import csv
    ruta_stats = os.path.join(carpeta, "montecarlo_stats.csv")
    with open(ruta_stats, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_total", "n_escaparon", "n_confinadas",
                    "fraccion_esc", "tau_medio_s", "tau_mediana_s",
                    "tau_std_s", "p25_s", "p75_s", "p95_s"])
        w.writerow([
            stats["n_total"], stats["n_escaparon"], stats["n_confinadas"],
            round(stats["fraccion_esc"], 6),
            stats["tau_medio"],  stats["tau_mediana"],
            stats["tau_std"],    stats["p25"],
            stats["p75"],        stats["p95"],
        ])
    print(f"  [MC] Estadisticas -> {ruta_stats}")

    # Curva de decaimiento
    ruta_decay = os.path.join(carpeta, "montecarlo_decaimiento.csv")
    datos = np.column_stack((t_arr, N_arr))
    np.savetxt(ruta_decay, datos, delimiter=",",
               header="t_s,N_confinadas", comments="")
    print(f"  [MC] Decaimiento  -> {ruta_decay}")


# ══════════════════════════════════════════════════════════════
#  6. RESUMEN EN CONSOLA
# ══════════════════════════════════════════════════════════════

def imprimir_resumen(stats: dict, escala_key: str = "us",
                     tau_ref: float = None):
    """Imprime un cuadro de texto con las estadísticas principales."""
    ESCALAS = {"s": ("s", 1.0), "ms": ("ms", 1e-3),
               "us": ("μs", 1e-6), "ns": ("ns", 1e-9)}
    unidad, factor = ESCALAS.get(escala_key, ("s", 1.0))

    print("\n╔══════════════════════════════════════════╗")
    print("║     RESULTADOS MONTE CARLO               ║")
    print("╚══════════════════════════════════════════╝")
    print(f"  Partículas totales   : {stats['n_total']}")
    print(f"  Escaparon            : {stats['n_escaparon']} "
          f"({100*stats['fraccion_esc']:.1f}%)")
    print(f"  Confinadas al final  : {stats['n_confinadas']}")
    print(f"  τ medio              : {stats['tau_medio']/factor:.4g} {unidad}")
    print(f"  τ mediana            : {stats['tau_mediana']/factor:.4g} {unidad}")
    print(f"  Desv. estándar       : {stats['tau_std']/factor:.4g} {unidad}")
    print(f"  Percentil 25 %       : {stats['p25']/factor:.4g} {unidad}")
    print(f"  Percentil 75 %       : {stats['p75']/factor:.4g} {unidad}")
    print(f"  Percentil 95 %       : {stats['p95']/factor:.4g} {unidad}")
    if tau_ref is not None:
        print(f"  τ Bohm (teórico)     : {tau_ref/factor:.4g} {unidad}")
        ratio = stats['tau_medio'] / tau_ref if tau_ref > 0 else float('nan')
        print(f"  τ_sim / τ_Bohm       : {ratio:.3f}")
    print()
