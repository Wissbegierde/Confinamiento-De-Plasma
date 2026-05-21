"""
analisis_error_iterativo.py
===========================
Módulo complementario para el análisis de error del resolvedor de Poisson.
Implementa un Resolvedor de Poisson mediante el método iterativo de Jacobi
y mide el decaimiento exponencial del error (residual y error L2 relativo)
paso a paso hasta 1000 iteraciones.

Guarda los resultados y figuras en data/validacion/.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GUARDAR_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "validacion"
)
os.makedirs(GUARDAR_DIR, exist_ok=True)

# Parámetros físicos e inicialización
Lx, Ly, Lz = 0.1, 0.1, 0.1  # Metros
rho_0 = 1e-6  # C/m^3
EPS0 = 8.854e-12
N = 16  # Grilla 16x16x16

def analizar_error_iterativo():
    print("==========================================================")
    print(" INICIANDO ANÁLISIS DE ERROR POISSON ITERATIVO (JACOBI)")
    print("==========================================================\n")
    
    # 1. Definición de la grilla
    h = Lx / (N - 1)
    xs = np.arange(N) * h
    ys = np.arange(N) * h
    zs = np.arange(N) * h
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    
    # 2. Solución analítica exacta (sinusoidal)
    factor_k2 = np.pi**2 * (1.0/Lx**2 + 1.0/Ly**2 + 1.0/Lz**2)
    phi_0 = rho_0 / (EPS0 * factor_k2)
    phi_exacto = phi_0 * np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly) * np.sin(np.pi * Z / Lz)
    
    # Densidad de carga
    rho = rho_0 * np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly) * np.sin(np.pi * Z / Lz)
    
    # Máscara de nodos interiores libres
    mask_libre = (X > 1e-9) & (X < Lx - 1e-9) & (Y > 1e-9) & (Y < Ly - 1e-9) & (Z > 1e-9) & (Z < Lz - 1e-9)
    phi_exacto[~mask_libre] = 0.0
    rho[~mask_libre] = 0.0
    
    # Precalcular el término derecho discretizado del paso de Jacobi: h^2 * rho / eps_0
    rhs = h**2 * rho / EPS0
    
    # Inicializar el potencial en cero (error inicial = 100%)
    phi = np.zeros((N, N, N))
    
    n_iteraciones = 1000
    historial_iteraciones = []
    historial_error_L2 = []
    historial_residual = []
    
    # Norma del término derecho para normalizar el residual
    norma_rhs = np.sqrt(np.sum(rhs[mask_libre]**2))
    
    print(f"  Ejecutando Jacobi para grilla de {N}x{N}x{N} ({N**3} nodos) durante {n_iteraciones} pasos...")
    
    for m in range(n_iteraciones + 1):
        # Calcular error L2 relativo actual
        err_l2 = np.sqrt(np.sum((phi[mask_libre] - phi_exacto[mask_libre])**2) / np.sum(phi_exacto[mask_libre]**2))
        
        # Calcular residual local: R = (6*phi - vecinos - rhs) en los nodos libres
        # Para evitar problemas en las fronteras, calculamos R en el interior [1:-1, 1:-1, 1:-1]
        vecinos = (
            phi[2:, 1:-1, 1:-1] + phi[:-2, 1:-1, 1:-1] +
            phi[1:-1, 2:, 1:-1] + phi[1:-1, :-2, 1:-1] +
            phi[1:-1, 1:-1, 2:] + phi[1:-1, 1:-1, :-2]
        )
        res_interior = 6.0 * phi[1:-1, 1:-1, 1:-1] - vecinos - rhs[1:-1, 1:-1, 1:-1]
        err_res = np.sqrt(np.sum(res_interior**2)) / norma_rhs
        
        historial_iteraciones.append(m)
        historial_error_L2.append(err_l2)
        historial_residual.append(err_res)
        
        if m % 100 == 0:
            print(f"    Paso = {m:4d} | Error L2 Relativo = {err_l2*100:8.5f}% | Residual Relativo = {err_res*100:8.5f}%")
        
        # Actualización de Jacobi para la siguiente iteración
        phi_new = np.zeros_like(phi)
        phi_new[1:-1, 1:-1, 1:-1] = (
            phi[2:, 1:-1, 1:-1] + phi[:-2, 1:-1, 1:-1] +
            phi[1:-1, 2:, 1:-1] + phi[1:-1, :-2, 1:-1] +
            phi[1:-1, 1:-1, 2:] + phi[1:-1, 1:-1, :-2] +
            rhs[1:-1, 1:-1, 1:-1]
        ) / 6.0
        
        phi = phi_new

    historial_iteraciones = np.array(historial_iteraciones)
    historial_error_L2 = np.array(historial_error_L2)
    historial_residual = np.array(historial_residual)
    
    # Guardar en CSV
    csv_path = os.path.join(GUARDAR_DIR, "convergencia_iterativa.csv")
    np.savetxt(
        csv_path,
        np.column_stack((historial_iteraciones, historial_error_L2, historial_residual)),
        delimiter=",",
        header="iteracion,error_L2_rel,residual_rel",
        comments=""
    )
    print(f"\n  Datos iterativos guardados en {csv_path}")
    
    # ── GRAFICACIÓN ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Gráfica en escala lineal (La curva exponencial clásica)
    ax1.plot(historial_iteraciones, historial_error_L2 * 100, color="darkviolet", linewidth=2, label="Error L2 Relativo")
    ax1.plot(historial_iteraciones, historial_residual * 100, color="darkorange", linewidth=1.5, linestyle="--", label="Residual Relativo")
    ax1.set_xlabel("Iteración (Paso)")
    ax1.set_ylabel("Error / Residual [%]")
    ax1.set_title("Decaimiento Exponencial del Error (Escala Lineal)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Gráfica en escala semilogarítmica (Para comprobar la tasa constante)
    ax2.semilogy(historial_iteraciones, historial_error_L2, color="darkviolet", linewidth=2, label="Error L2 Relativo")
    ax2.semilogy(historial_iteraciones, historial_residual, color="darkorange", linewidth=1.5, linestyle="--", label="Residual Relativo")
    ax2.set_xlabel("Iteración (Paso)")
    ax2.set_ylabel("Error / Residual (Adimensional)")
    ax2.set_title("Decaimiento del Error (Escala Semi-Logarítmica)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()
    
    fig.suptitle("Convergencia del Resolvedor Iterativo de Poisson (Jacobi)", fontsize=14, y=1.02)
    fig_path = os.path.join(GUARDAR_DIR, "convergencia_iterativa.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gráfica guardada en {fig_path}")
    
    print("\n==========================================================")
    print(" ANÁLISIS ITERATIVO FINALIZADO.")
    print("==========================================================\n")

if __name__ == "__main__":
    analizar_error_iterativo()
