"""
analisis_error_completo.py
==========================
Módulo para el análisis detallado de errores del simulador:
1. Convergencia Temporal (Boris Step): trayectoria y energía en campo E×B cruzados.
2. Estabilidad Numérica: efectos de aumentar dt más allá del límite ciclotrónico.
3. Convergencia Espacial (Resolvedor de Poisson): comparación contra solución analítica.
4. Teoría vs Experimental: trayectoria de Boris paso a paso, error acumulado por iteración.
5. Convergencia de Distribuciones: muestras → PDF teórica con N creciente
   (Exponencial/Poisson, Maxwell-Boltzmann, Uniforme/Rechazo).

Guarda los resultados y figuras en data/validacion/.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from integradores import boris_step
from interacciones import ResolvedorPoisson

GUARDAR_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "validacion"
)
os.makedirs(GUARDAR_DIR, exist_ok=True)

# Constantes del protón de referencia
M_P = 1.673e-27
Q_P = 1.602e-19
EPS0 = 8.854e-12


# ══════════════════════════════════════════════════════════════
#  1. ANALISIS DE CONVERGENCIA TEMPORAL (INTEGRADOR DE BORIS)
# ══════════════════════════════════════════════════════════════

def analizar_convergencia_temporal():
    print("--- 1. Analizando Convergencia Temporal del Integrador Boris ---")
    
    # Parámetros físicos: E x B cruzados (más general que solo B)
    B0 = 0.5  # Tesla
    Ex = 1e3  # V/m
    B = np.array([0.0, 0.0, B0])
    E = np.array([Ex, 0.0, 0.0])
    
    v_perp = 1e4  # m/s
    v0 = np.array([0.0, -v_perp, 0.0])
    x0 = np.array([0.0, 0.0, 0.0])
    
    omega_c = Q_P * B0 / M_P
    T_c = 2 * np.pi / omega_c
    v_D = -Ex / B0  # Velocidad de deriva E x B en y
    
    # Tiempo de integración: ~1.5 ciclos ciclotrónicos
    T_fin = 1.5 * T_c
    
    # Pasos de tiempo a ensayar (en segundos)
    dt_vals = np.array([5e-11, 1e-10, 2e-10, 5e-10, 1e-9, 2e-9])
    
    errores_pos = []
    errores_ene = []
    
    # Solución analítica en t
    def sol_analitica(t):
        # x(t) = (v_perp + v_D)/omega_c * (cos(omega_c * t) - 1)
        # y(t) = v_D * t - (v_perp + v_D)/omega_c * sin(omega_c * t)
        # z(t) = 0
        amp = (v_perp + v_D) / omega_c
        x = amp * (np.cos(omega_c * t) - 1.0)
        y = v_D * t - amp * np.sin(omega_c * t)
        z = np.zeros_like(t)
        return np.column_stack((x, y, z))

    print(f"  Ciclo ciclotrónico T_c = {T_c*1e9:.2f} ns. Velocidad de deriva v_D = {v_D:.1f} m/s.")
    
    for dt in dt_vals:
        n_pasos = int(T_fin / dt)
        t_arr = np.arange(n_pasos + 1) * dt
        
        # Simulación con Staggering de Velocidad para conservar el 2do orden de convergencia
        # v_staggered = v0 - (dt/2) * (q/m) * (E + v0 x B)
        v_staggered = v0 - (dt / 2.0) * (Q_P / M_P) * (E + np.cross(v0, B))
        
        x = x0.copy()
        
        hist_x = [x.copy()]
        hist_v_half = [v_staggered.copy()]
        
        for _ in range(n_pasos):
            x, v_staggered = boris_step(x, v_staggered, E, B, Q_P, M_P, dt)
            hist_x.append(x.copy())
            hist_v_half.append(v_staggered.copy())
            
        hist_x = np.array(hist_x)
        
        # Reconstruir velocidades en pasos enteros t_k para el cálculo de energía y comparación
        v_int = []
        for k in range(n_pasos):
            vk = (hist_v_half[k] + hist_v_half[k+1]) / 2.0
            v_int.append(vk)
        v_int = np.array(v_int)
        
        # Comparación con solución analítica en cada paso (omitimos el último para coincidir con v_int)
        pos_teo = sol_analitica(t_arr[:-1])
        errs = np.linalg.norm(hist_x[:-1] - pos_teo, axis=1)
        max_err_pos = np.max(errs)
        
        # Conservación de la energía total (Cinética + Potencial electrostática)
        # E_total = 0.5 * m * v^2 - q * E_x * x
        E_pot = - Q_P * Ex * hist_x[:-1, 0]
        E_cin = 0.5 * M_P * np.sum(v_int**2, axis=1)
        E_total = E_cin + E_pot
        
        E_cin_teo = 0.5 * M_P * np.sum(v0**2)  # energía inicial de referencia
        max_err_ene = np.max(np.abs(E_total - E_total[0])) / E_cin_teo
        
        errores_pos.append(max_err_pos)
        errores_ene.append(max_err_ene)
        
        print(f"    dt = {dt*1e9:5.1f} ns | Pasos = {n_pasos:6d} | Max Err Pos = {max_err_pos*1e6:7.3f} um | Max Err Ene = {max_err_ene*100:8.5f}%")

    errores_pos = np.array(errores_pos)
    errores_ene = np.array(errores_ene)
    
    # Calcular orden de convergencia (pendiente en escala log-log)
    p_pos = np.polyfit(np.log(dt_vals), np.log(errores_pos), 1)[0]
    p_ene = np.polyfit(np.log(dt_vals), np.log(errores_ene), 1)[0]
    
    print(f"  --> Orden de convergencia en posición (teórico ~ 2): {p_pos:.3f}")
    print(f"  --> Orden de convergencia en energía (teórico ~ 2):  {p_ene:.3f}")
    
    # Guardar datos en CSV
    csv_path = os.path.join(GUARDAR_DIR, "convergencia_temporal.csv")
    np.savetxt(
        csv_path,
        np.column_stack((dt_vals, errores_pos, errores_ene)),
        delimiter=",",
        header="dt_s,error_pos_m,error_energia_rel",
        comments=""
    )
    print(f"  Datos guardados en {csv_path}")
    
    # Graficar en dos paneles para evitar mezclar escalas incompatibles
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor("white")
    
    # Panel izquierdo: Error en Posición
    ax1.loglog(dt_vals * 1e9, errores_pos * 1e6, "o-", color="firebrick", label=f"Error Posición (Pendiente = {p_pos:.3f})")
    dt_ref = np.linspace(dt_vals.min(), dt_vals.max(), 100)
    C_pos = (errores_pos[-1] * 1e6) / (dt_vals[-1] * 1e9)**2
    ax1.loglog(dt_ref * 1e9, C_pos * (dt_ref * 1e9)**2, ":", color="gray", label="Guía teórica O(dt²)")
    ax1.set_xlabel("Paso de tiempo dt [ns]")
    ax1.set_ylabel("Error Máximo en Posición [µm]")
    ax1.set_title("Convergencia en Posición")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()
    
    # Panel derecho: Error en Energía
    ax2.loglog(dt_vals * 1e9, errores_ene * 100, "s--", color="royalblue", label=f"Error Energía (Pendiente = {p_ene:.3f})")
    C_ene = (errores_ene[-1] * 100) / (dt_vals[-1] * 1e9)**2
    ax2.loglog(dt_ref * 1e9, C_ene * (dt_ref * 1e9)**2, ":", color="gray", label="Guía teórica O(dt²)")
    ax2.set_xlabel("Paso de tiempo dt [ns]")
    ax2.set_ylabel("Error Relativo Máximo en Energía [%]")
    ax2.set_title("Conservación de Energía")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()
    
    fig.suptitle("Análisis de Convergencia Temporal (Boris Step)", fontsize=13, y=0.98)
    plt.tight_layout()
    
    fig_path = os.path.join(GUARDAR_DIR, "convergencia_temporal.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gráfica guardada en {fig_path}\n")


# ══════════════════════════════════════════════════════════════
#  2. ANALISIS DE ESTABILIDAD NUMÉRICA
# ══════════════════════════════════════════════════════════════

def analizar_estabilidad():
    print("--- 2. Analizando Estabilidad Numérica (Límite Ciclotrónico) ---")
    
    # Para Boris en campo B constante, el esquema es incondicionalmente estable en energía
    # (la velocidad nunca crece sin límites porque la rotación conserva la norma),
    # pero la trayectoria se distorsiona catastróficamente si dt > T_c / 2 (es decir, omega_c * dt > pi).
    # Si omega_c * dt = 2, el ángulo de giro numérico es theta = 2 * arctan(1) = pi/2 por paso.
    # Si omega_c * dt es muy grande, theta se acerca a pi por paso, perdiendo toda la física.
    
    B0 = 0.5
    B = np.array([0.0, 0.0, B0])
    v_perp = 1e4
    v0 = np.array([0.0, -v_perp, 0.0])
    x0 = np.array([0.0, 0.0, 0.0])
    
    omega_c = Q_P * B0 / M_P
    T_c = 2 * np.pi / omega_c
    
    # Tres casos de pasos de tiempo:
    # 1. dt_estable = T_c / 20 (resolución excelente de la órbita)
    # 2. dt_limite = T_c / 4  (órbita poligonal burda pero estable)
    # 3. dt_inestable = T_c * 0.8 (excede el límite físico de muestreo de Nyquist, la órbita se destruye)
    dts = {
        "Estable (Tc/20)": T_c / 20.0,
        "Coarse (Tc/4)": T_c / 4.0,
        "Inestable (Tc * 0.8)": T_c * 0.8
    }
    
    T_sim = 2.0 * T_c
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Trayectoria analítica
    theta = np.linspace(0, 2*np.pi * 2, 200)
    r_L = v_perp / omega_c
    x_analitico = r_L * (np.cos(theta) - 1.0)
    y_analitico = -r_L * np.sin(theta)
    
    for ax, (label, dt) in zip(axes, dts.items()):
        n_pasos = int(T_sim / dt)
        x = x0.copy()
        v = v0.copy()
        
        hist_x = [x.copy()]
        for _ in range(n_pasos):
            # En campo puro B sin E
            x, v = boris_step(x, v, np.zeros(3), B, Q_P, M_P, dt)
            hist_x.append(x.copy())
        hist_x = np.array(hist_x)
        
        ax.plot(x_analitico * 1e3, y_analitico * 1e3, "--", color="gray", label="Teórica")
        ax.plot(hist_x[:, 0] * 1e3, hist_x[:, 1] * 1e3, "o-", markersize=4, label="Numérica")
        ax.set_aspect("equal")
        ax.set_title(f"{label}\ndt = {dt*1e9:.1f} ns")
        ax.set_xlabel("x [mm]")
        if ax == axes[0]:
            ax.set_ylabel("y [mm]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
    fig.suptitle("Efecto del Paso de Tiempo dt en la Estabilidad de la Órbita", fontsize=14, y=1.05)
    fig_path = os.path.join(GUARDAR_DIR, "estabilidad_orbita.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gráfica de estabilidad guardada en {fig_path}\n")


# ══════════════════════════════════════════════════════════════
#  3. ANALISIS DE CONVERGENCIA ESPACIAL (RESOLVEDOR DE POISSON)
# ══════════════════════════════════════════════════════════════

def analizar_convergencia_espacial():
    print("--- 3. Analizando Convergencia Espacial del Resolvedor de Poisson ---")
    
    # Definimos una caja cúbica grounded (Dirichlet = 0 en las caras)
    Lx, Ly, Lz = 0.1, 0.1, 0.1  # Metros
    rho_0 = 1e-6  # C/m^3 (densidad de carga de prueba)
    
    # Resoluciones a ensayar
    resoluciones = [8, 12, 16, 20, 24, 30]
    
    errores_L2 = []
    grid_spacings = []
    
    # Solución analítica para una densidad sinusoidal:
    # rho(x,y,z) = rho_0 * sin(pi*x/Lx) * sin(pi*y/Ly) * sin(pi*z/Lz)
    # phi_teo(x,y,z) = phi_0 * sin(pi*x/Lx) * sin(pi*y/Ly) * sin(pi*z/Lz)
    # donde phi_0 = rho_0 / (eps0 * pi^2 * (1/Lx^2 + 1/Ly^2 + 1/Lz^2))
    factor_k2 = np.pi**2 * (1.0/Lx**2 + 1.0/Ly**2 + 1.0/Lz**2)
    phi_0 = rho_0 / (EPS0 * factor_k2)
    
    def obtener_analitico(X, Y, Z):
        return phi_0 * np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly) * np.sin(np.pi * Z / Lz)

    # Función que define la frontera de la caja de simulación
    def frontera_caja(X, Y, Z):
        # Los nodos frontera en los bordes exactos del resolvedor (usando tolerancia)
        tol = 1e-9
        return (X <= tol) | (X >= Lx - tol) | (Y <= tol) | (Y >= Ly - tol) | (Z <= tol) | (Z >= Lz - tol)

    for N in resoluciones:
        # Instanciar resolvedor de Poisson
        res = ResolvedorPoisson(dimensiones=[Lx, Ly, Lz], resolucion=(N, N, N), epsilon_0=EPS0)
        
        # Configurar frontera
        res.definir_frontera_vectorizada(frontera_caja)
        
        # Cargar densidad de carga sinusoidal en los nodos internos
        # (en los nodos frontera es 0)
        xs = np.arange(N) * res.dx
        ys = np.arange(N) * res.dy
        zs = np.arange(N) * res.dz
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        
        rho_grid = rho_0 * np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly) * np.sin(np.pi * Z / Lz)
        # Aplicar máscara libre (solo interior tiene carga, frontera es Dirichlet = 0)
        rho_grid[~res.mask_libre] = 0.0
        res.rho = rho_grid
        
        # Resolver
        res.precalcular_matriz()
        res.resolver_con_matriz()
        
        # Evaluar solución exacta
        phi_exacto = obtener_analitico(X, Y, Z)
        phi_exacto[~res.mask_libre] = 0.0
        
        # Calcular error L2 relativo en los nodos libres (interiores)
        nodos_libres = res.mask_libre
        err_abs = res.phi[nodos_libres] - phi_exacto[nodos_libres]
        err_l2 = np.sqrt(np.sum(err_abs**2) / np.sum(phi_exacto[nodos_libres]**2))
        
        errores_L2.append(err_l2)
        grid_spacings.append(res.dx)
        
        print(f"    Res = {N:2d}³ | h = {res.dx*1e3:6.3f} mm | Error L2 Rel = {err_l2*100:8.5f}%")

    grid_spacings = np.array(grid_spacings)
    errores_L2 = np.array(errores_L2)
    
    # Calcular orden de convergencia espacial (pendiente en escala log-log)
    p_esp = np.polyfit(np.log(grid_spacings), np.log(errores_L2), 1)[0]
    print(f"  --> Orden de convergencia espacial (teórico ~ 2): {p_esp:.3f}")
    
    # Guardar en CSV
    csv_path = os.path.join(GUARDAR_DIR, "convergencia_espacial.csv")
    np.savetxt(
        csv_path,
        np.column_stack((grid_spacings, errores_L2)),
        delimiter=",",
        header="h_m,error_L2_rel",
        comments=""
    )
    print(f"  Datos guardados en {csv_path}")
    
    # Graficar
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(grid_spacings * 1e3, errores_L2 * 100, "o-", color="darkgreen", label=f"Error L2 Relativo (Pendiente = {p_esp:.2f})")
    
    # Guía O(h^2)
    h_ref = np.linspace(grid_spacings.min(), grid_spacings.max(), 100)
    ax.loglog(h_ref * 1e3, 1e4 * (h_ref)**2, ":", color="gray", label="Guía teórica O(h²)")
    
    ax.set_xlabel("Espaciamiento de grilla h [mm]")
    ax.set_ylabel("Error L2 Relativo [%]")
    ax.set_title("Análisis de Convergencia Espacial (Poisson Solver)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    
    fig_path = os.path.join(GUARDAR_DIR, "convergencia_espacial.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gráfica guardada en {fig_path}\n")


# ══════════════════════════════════════════════════════════════
#  4. TEORIA VS EXPERIMENTAL: TRAYECTORIA BORIS PASO A PASO
# ══════════════════════════════════════════════════════════════

def analizar_teo_vs_exp():
    """Compara la trayectoria numérica de Boris contra la solución analítica
    exacta paso a paso, mostrando la evolución del error con el número de
    iteraciones para tres valores de dt."""
    print("--- 4. Analizando Teoría vs Experimental (Trayectoria Boris) ---")

    B0   = 0.5          # T
    Ex   = 1e3          # V/m
    B    = np.array([0.0, 0.0, B0])
    E    = np.array([Ex, 0.0, 0.0])
    v_perp = 1e4        # m/s
    v0   = np.array([0.0, -v_perp, 0.0])
    x0   = np.array([0.0,  0.0,   0.0])

    omega_c = Q_P * B0 / M_P
    T_c     = 2 * np.pi / omega_c
    v_D     = -Ex / B0             # deriva E×B teórica
    T_fin   = 1.5 * T_c

    # Solución analítica continua
    def sol_analitica(t_arr):
        amp = (v_perp + v_D) / omega_c
        x   = amp * (np.cos(omega_c * t_arr) - 1.0)
        y   = v_D * t_arr - amp * np.sin(omega_c * t_arr)
        return np.column_stack((x, y, np.zeros_like(t_arr)))

    # Tres resoluciones temporales
    casos = [
        ("dt fino  (Tc/60)",  T_c / 60,  "#4a9eff"),
        ("dt medio (Tc/20)",  T_c / 20,  "#50fa7b"),
        ("dt grueso (Tc/6)",  T_c /  6,  "#ff6b6b"),
    ]

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0d0d1a")

    # --- columna izquierda: trayectorias XY superpuestas ---
    ax_xy = fig.add_subplot(2, 3, (1, 4))   # ocupa filas 1-2, columna 1
    ax_xy.set_facecolor("#12122a")
    ax_xy.tick_params(colors="white")
    for sp in ax_xy.spines.values(): sp.set_edgecolor("#333355")
    ax_xy.xaxis.label.set_color("white")
    ax_xy.yaxis.label.set_color("white")
    ax_xy.title.set_color("white")

    # panel de error por iteración (fila 1, cols 2 y 3)
    ax_err1 = fig.add_subplot(2, 3, 2)
    ax_err2 = fig.add_subplot(2, 3, 3)
    # panel de error acumulado (fila 2, cols 2 y 3)
    ax_err3 = fig.add_subplot(2, 3, 5)
    ax_err4 = fig.add_subplot(2, 3, 6)

    for ax in [ax_err1, ax_err2, ax_err3, ax_err4]:
        ax.set_facecolor("#12122a")
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#333355")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    # Trayectoria analítica de referencia (200 puntos)
    t_ref = np.linspace(0, T_fin, 600)
    xy_ref = sol_analitica(t_ref)
    ax_xy.plot(xy_ref[:, 0]*1e3, xy_ref[:, 1]*1e3,
               "--", color="white", lw=1.4, alpha=0.6, label="Analítica")

    for label, dt, color in casos:
        n_pasos = int(T_fin / dt)
        t_arr   = np.arange(n_pasos + 1) * dt

        # staggered init
        v_st = v0 - (dt / 2.0) * (Q_P / M_P) * (E + np.cross(v0, B))
        x    = x0.copy()
        hist_x    = [x.copy()]
        hist_v_st = [v_st.copy()]

        for _ in range(n_pasos):
            x, v_st = boris_step(x, v_st, E, B, Q_P, M_P, dt)
            hist_x.append(x.copy())
            hist_v_st.append(v_st.copy())

        hist_x = np.array(hist_x)          # (n+1, 3)

        # Velocidades en pasos enteros
        v_int = [(hist_v_st[k] + hist_v_st[k+1]) / 2.0 for k in range(n_pasos)]
        v_int = np.array(v_int)

        # Posición analítica en cada paso
        pos_teo = sol_analitica(t_arr[:-1])
        err_pos = np.linalg.norm(hist_x[:-1] - pos_teo, axis=1)   # m

        # Energía total: E_cin + E_pot electrostático
        E_pot   = -Q_P * Ex * hist_x[:-1, 0]
        E_cin   = 0.5 * M_P * np.sum(v_int**2, axis=1)
        E_tot   = E_cin + E_pot
        err_ene = np.abs(E_tot - E_tot[0]) / (0.5 * M_P * np.sum(v0**2))

        pasos_arr = np.arange(n_pasos)

        # Trayectoria XY
        ax_xy.plot(hist_x[:, 0]*1e3, hist_x[:, 1]*1e3,
                   "-", color=color, lw=0.9, alpha=0.85, label=label)

        # Error posición vs iteración (escala lineal)
        ax_err1.plot(pasos_arr, err_pos*1e6, color=color, lw=0.9, alpha=0.85, label=label)
        # Error posición vs iteración (escala log)
        ax_err2.semilogy(pasos_arr, err_pos + 1e-30, color=color, lw=0.9, alpha=0.85, label=label)
        # Error energía vs iteración (lineal)
        ax_err3.plot(pasos_arr, err_ene*100, color=color, lw=0.9, alpha=0.85, label=label)
        # Error energía vs iteración (log)
        ax_err4.semilogy(pasos_arr, err_ene + 1e-30, color=color, lw=0.9, alpha=0.85, label=label)

    ax_xy.set_xlabel("x [mm]"); ax_xy.set_ylabel("y [mm]")
    ax_xy.set_title("Trayectoria Ciclodal — Teoría vs Experimental", pad=8)
    ax_xy.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", framealpha=0.6)
    ax_xy.grid(True, alpha=0.2)

    for ax, title, ylabel in [
        (ax_err1, "Error Posición vs Iteración (Lineal)", "Error [μm]"),
        (ax_err2, "Error Posición vs Iteración (Log)",    "Error [m]"),
        (ax_err3, "Error Energía vs Iteración (Lineal)",  "Err. E [%]"),
        (ax_err4, "Error Energía vs Iteración (Log)",     "Err. E (rel)"),
    ]:
        ax.set_xlabel("Iteración (paso)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9, pad=5)
        ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white", framealpha=0.6)
        ax.grid(True, alpha=0.2, which="both")

    fig.suptitle(
        "Análisis Teoría vs Experimental — Integrador de Boris (campos E×B)",
        color="white", fontsize=13, y=1.01
    )
    plt.tight_layout()
    fig_path = os.path.join(GUARDAR_DIR, "teo_vs_exp_boris.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close(fig)
    print(f"  Gráfica guardada en {fig_path}\n")


# ══════════════════════════════════════════════════════════════
#  5. CONVERGENCIA DE DISTRIBUCIONES ALEATORIAS -> PDF TEÓRICA
# ══════════════════════════════════════════════════════════════

def analizar_convergencia_distribuciones():
    """Demuestra que al aumentar N las muestras convergen a las PDFs teóricas
    de las tres distribuciones usadas en la simulación (definidas en
    distribuciones_aleatorias.tex):

      A) Exponencial  – tiempos entre colisiones (proceso de Poisson)
      B) Maxwell-Boltzmann – velocidades térmicas de las partículas
      C) Uniforme (rechazo) – posiciones dentro de un cilindro
    """
    print("--- 5. Convergencia de Distribuciones Aleatorias -> PDF Teorica ---")

    rng    = np.random.default_rng(42)
    N_vals = [50, 500, 5_000, 50_000]   # muestras a comparar
    colores = ["#ff6b6b", "#ffb86c", "#50fa7b", "#4a9eff"]

    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    fig.patch.set_facecolor("#0d0d1a")

    for ax in axes.flat:
        ax.set_facecolor("#12122a")
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#333355")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    # ── A) Distribución Exponencial (colisiones de Poisson) ────────────
    nu   = 200.0         # frecuencia de colisión [Hz]
    dt_s = 1e-8          # paso de tiempo [s]   → P_col = 1 - exp(-nu*dt)
    lam  = nu            # parámetro de la exponencial
    x_exp = np.linspace(0, 5.0 / lam, 400)
    pdf_exp = lam * np.exp(-lam * x_exp)   # f(t) = λ e^{-λt}

    for col_idx, N in enumerate(N_vals):
        ax = axes[0, col_idx]
        muestras = rng.exponential(scale=1.0/lam, size=N)
        # histograma como densidad
        counts, edges = np.histogram(muestras, bins=max(10, min(80, N//10)), density=True)
        mids = (edges[:-1] + edges[1:]) / 2
        ax.bar(mids*1e3, counts/1e3, width=(edges[1]-edges[0])*1e3,
               color=colores[col_idx], alpha=0.65, edgecolor="#0d0d1a", lw=0.4,
               label=f"Muestras N={N:,}")
        ax.plot(x_exp*1e3, pdf_exp/1e3, color="white", lw=2,
                linestyle="--", label=r"Teórica: $\lambda e^{-\lambda t}$")
        ax.set_xlabel("Tiempo entre colisiones [ms]")
        ax.set_ylabel("Densidad de probabilidad" if col_idx == 0 else "")
        ax.set_title(f"Exponencial  N={N:,}", fontsize=9)
        ax.legend(fontsize=6.5, facecolor="#1a1a2e", labelcolor="white", framealpha=0.6)
        ax.grid(True, alpha=0.2)

        # Error KS (kolmogorov-smirnov como proxy de convergencia)
        from scipy.stats import kstest
        D, p = kstest(muestras, "expon", args=(0, 1.0/lam))
        ax.text(0.98, 0.95, f"KS D={D:.3f}\np={p:.3f}",
                transform=ax.transAxes, ha="right", va="top",
                color="#aaaacc", fontsize=7, family="monospace")

    # ── B) Maxwell-Boltzmann (velocidades térmicas) ────────────────────
    T_plasma = 1e4        # K
    m_p      = 1.673e-27  # kg
    kB       = 1.380649e-23
    sigma_v  = np.sqrt(kB * T_plasma / m_p)  # desv. estándar de cada componente

    # La rapidez |v| sigue Maxwell: f(v) = 4π (m/2πkT)^{3/2} v² exp(-mv²/2kT)
    a_mb = sigma_v   # parámetro de escala de Maxwell
    v_max = 6 * sigma_v
    x_mb  = np.linspace(0, v_max, 500)
    # f(v) de Maxwell-Boltzmann:
    pdf_mb = (np.sqrt(2/np.pi) * x_mb**2 / a_mb**3
              * np.exp(-x_mb**2 / (2 * a_mb**2)))

    for col_idx, N in enumerate(N_vals):
        ax = axes[1, col_idx]
        # Generamos 3 componentes gaussianas independientes y calculamos |v|
        vx = rng.normal(0, sigma_v, N)
        vy = rng.normal(0, sigma_v, N)
        vz = rng.normal(0, sigma_v, N)
        rapidez = np.sqrt(vx**2 + vy**2 + vz**2)

        counts, edges = np.histogram(rapidez, bins=max(10, min(80, N//10)), density=True)
        mids = (edges[:-1] + edges[1:]) / 2
        ax.bar(mids/1e3, counts*1e3, width=(edges[1]-edges[0])/1e3,
               color=colores[col_idx], alpha=0.65, edgecolor="#0d0d1a", lw=0.4,
               label=f"Muestras N={N:,}")
        ax.plot(x_mb/1e3, pdf_mb*1e3, color="white", lw=2,
                linestyle="--", label="Teórica: Maxwell-Boltzmann")
        ax.set_xlabel("|v| [km/s]")
        ax.set_ylabel("Densidad de probabilidad" if col_idx == 0 else "")
        ax.set_title(f"Maxwell-Boltzmann  N={N:,}", fontsize=9)
        ax.legend(fontsize=6.5, facecolor="#1a1a2e", labelcolor="white", framealpha=0.6)
        ax.grid(True, alpha=0.2)

        # Error cuadrático medio entre histograma y PDF teórica en los mismos bins
        pdf_teo_bins = (np.sqrt(2/np.pi) * mids**2 / a_mb**3
                        * np.exp(-mids**2 / (2 * a_mb**2)))
        rmse = np.sqrt(np.mean((counts - pdf_teo_bins)**2)) / (pdf_teo_bins.max() + 1e-30)
        ax.text(0.98, 0.95, f"RMSE rel.\n= {rmse:.4f}",
                transform=ax.transAxes, ha="right", va="top",
                color="#aaaacc", fontsize=7, family="monospace")

    # ── C) Uniforme (método de rechazo en un círculo 2D) ──────────────
    # Radio del cilindro de referencia (proyección XY)
    R_cil = 0.01   # metros
    # PDF de r = distancia al eje en disco uniforme: f(r) = 2r/R^2
    x_r   = np.linspace(0, R_cil, 400)
    pdf_r = 2 * x_r / R_cil**2

    for col_idx, N in enumerate(N_vals):
        ax = axes[2, col_idx]
        # Simulamos el método de rechazo: generar en caja [-R,R]x[-R,R], aceptar en disco
        aceptadas = []
        intentos  = 0
        while len(aceptadas) < N:
            bx = rng.uniform(-R_cil, R_cil, N * 5)
            by = rng.uniform(-R_cil, R_cil, N * 5)
            dentro = bx**2 + by**2 <= R_cil**2
            for px, py, ok in zip(bx, by, dentro):
                if ok:
                    aceptadas.append(np.sqrt(px**2 + py**2))
                intentos += 1
                if len(aceptadas) >= N:
                    break
        radios = np.array(aceptadas[:N])

        counts, edges = np.histogram(radios, bins=max(10, min(60, N//10)), density=True)
        mids = (edges[:-1] + edges[1:]) / 2
        ax.bar(mids*1e3, counts/1e3, width=(edges[1]-edges[0])*1e3,
               color=colores[col_idx], alpha=0.65, edgecolor="#0d0d1a", lw=0.4,
               label=f"Muestras N={N:,}")
        ax.plot(x_r*1e3, pdf_r/1e3, color="white", lw=2,
                linestyle="--", label=r"Teórica: $f(r) = 2r/R^2$")
        ax.set_xlabel("Distancia al eje r [mm]")
        ax.set_ylabel("Densidad de probabilidad" if col_idx == 0 else "")
        ax.set_title(f"Rechazo (Uniforme disco)  N={N:,}", fontsize=9)
        ax.legend(fontsize=6.5, facecolor="#1a1a2e", labelcolor="white", framealpha=0.6)
        ax.grid(True, alpha=0.2)

        # Error L1 normalizado
        pdf_teo_bins = 2 * mids / R_cil**2
        L1 = np.mean(np.abs(counts - pdf_teo_bins)) / (pdf_teo_bins.max() + 1e-30)
        ax.text(0.98, 0.95, f"L1 rel.\n= {L1:.4f}",
                transform=ax.transAxes, ha="right", va="top",
                color="#aaaacc", fontsize=7, family="monospace")

    # Etiquetas de fila a la izquierda
    for fila, nombre in enumerate(["A) Exponencial\n(Colisiones Poisson)",
                                    "B) Maxwell-Boltzmann\n(Velocidades Térmicas)",
                                    "C) Uniforme/Rechazo\n(Posiciones en Cilindro)"]):
        axes[fila, 0].set_ylabel(nombre + "\n\nDensidad de probabilidad",
                                  color="white", fontsize=9)

    fig.suptitle(
        "Convergencia de Numeros Aleatorios -> PDF Teorica (N creciente)",
        color="white", fontsize=14, y=1.01
    )
    plt.tight_layout()
    fig_path = os.path.join(GUARDAR_DIR, "convergencia_distribuciones.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close(fig)
    print(f"  Gráfica guardada en {fig_path}\n")


# ══════════════════════════════════════════════════════════════
#  EJECUCIÓN GENERAL
# ══════════════════════════════════════════════════════════════

def main():
    print("==========================================================")
    print(" INICIANDO ANÁLISIS COMPLETO DE ERRORES Y CONVERGENCIA")
    print("==========================================================\n")

    analizar_convergencia_temporal()
    analizar_estabilidad()
    analizar_convergencia_espacial()
    analizar_teo_vs_exp()
    analizar_convergencia_distribuciones()

    print("==========================================================")
    print(" ANÁLISIS DE ERRORES FINALIZADO.")
    print(f" Gráficas y datos CSV exportados a: {GUARDAR_DIR}")
    print("==========================================================")


if __name__ == "__main__":
    main()
