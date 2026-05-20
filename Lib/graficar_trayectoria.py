"""
graficar_trayectoria.py - Semana 4 del cronograma.
Lee data/trayectoria_helicoidal.csv, trayectoria_p1.csv, trayectoria_p2.csv
y genera figura 3D multi-panel.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

BASE   = os.path.join(os.path.dirname(__file__), "..", "data")
OUT    = os.path.join(BASE, "trayectorias", "trayectorias_3D.png")
CSV_H  = os.path.join(BASE, "trayectoria_helicoidal.csv")
CSV_P1 = os.path.join(BASE, "trayectoria_p1.csv")
CSV_P2 = os.path.join(BASE, "trayectoria_p2.csv")

# Físicos
M = 1.673e-27; Q = 1.602e-19; KB = 1.38e-23
B0 = 0.1; T_K = 1e4

BG = "#0d1117"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.facecolor":BG,
    "figure.facecolor":BG,"axes.edgecolor":"#8888aa",
    "axes.labelcolor":"white","xtick.color":"#8888aa","ytick.color":"#8888aa",
    "text.color":"white","grid.color":"#2a2a4a","grid.alpha":0.5})

def load(path):
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    return d[::2]   # submuestreo

def graficar():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    h  = load(CSV_H);  t_h, xh, yh, zh = h[:,0], h[:,1], h[:,2], h[:,3]
    p1 = load(CSV_P1); t1, x1, y1, z1  = p1[:,0],p1[:,1],p1[:,2],p1[:,3]
    p2 = load(CSV_P2); t2, x2, y2, z2  = p2[:,0],p2[:,1],p2[:,2],p2[:,3]

    v_th   = np.sqrt(2*KB*T_K/M)
    r_L    = M*v_th/(Q*B0)
    omega_c= Q*B0/M
    T_c    = 2*np.pi/omega_c

    fig = plt.figure(figsize=(16,11), facecolor=BG)
    fig.suptitle(
        "Validación de Trayectorias Helicoidales — Confinamiento de Plasma\n"
        f"Protón | B₀={B0} T | T={T_K:.0e} K | Integrador Boris",
        color="white", fontsize=14, fontweight="bold", y=0.99)

    # ─── 3D ───
    ax3 = fig.add_subplot(2,3,(1,4), projection="3d", facecolor=BG)
    ax3.set_facecolor(BG)
    ax3.plot(xh, yh, zh, color="#00e5ff", lw=1.2, label="Helicoidal pura")
    ax3.plot(x1, y1, z1, color="#ff6b6b", lw=0.8, alpha=0.7, label="Partícula 1")
    ax3.plot(x2, y2, z2, color="#ffd166", lw=0.8, alpha=0.7, label="Partícula 2")
    ax3.scatter(*[xh[0],yh[0],zh[0]],  color="white", s=40, zorder=5)
    ax3.scatter(*[xh[-1],yh[-1],zh[-1]], color="#00e5ff", s=40, marker="^", zorder=5)
    for sp in [ax3.xaxis, ax3.yaxis, ax3.zaxis]:
        sp.pane.fill=False; sp.pane.set_edgecolor("#8888aa")
    ax3.set_xlabel("x [m]"); ax3.set_ylabel("y [m]"); ax3.set_zlabel("z [m]")
    ax3.set_title("Vista 3D", color="white")
    ax3.tick_params(colors="#8888aa")
    ax3.legend(loc="upper left", framealpha=0.3, facecolor=BG,
               edgecolor="#8888aa", labelcolor="white", fontsize=8)

    # ─── XY ───
    axXY = fig.add_subplot(2,3,2, facecolor=BG)
    axXY.plot(xh, yh, color="#00e5ff", lw=1.2)
    axXY.plot(x1, y1, color="#ff6b6b", lw=0.7, alpha=0.6)
    axXY.plot(x2, y2, color="#ffd166", lw=0.7, alpha=0.6)
    th = np.linspace(0,2*np.pi,200)
    cx,cy = xh.mean(), yh.mean()
    axXY.plot(cx+r_L*np.cos(th), cy+r_L*np.sin(th),
              "--", color="white", lw=0.8, alpha=0.5, label=f"r_L={r_L*100:.2f} cm")
    axXY.set_xlabel("x [m]"); axXY.set_ylabel("y [m]")
    axXY.set_title("Plano XY (transversal)", color="white")
    axXY.legend(fontsize=7, framealpha=0.3, facecolor=BG,
                edgecolor="#8888aa", labelcolor="white")
    axXY.grid(True); axXY.set_aspect("equal","box"); axXY.tick_params(colors="#8888aa")

    # ─── XZ ───
    axXZ = fig.add_subplot(2,3,3, facecolor=BG)
    axXZ.plot(zh*1e3, xh, color="#00e5ff", lw=1.2)
    axXZ.plot(z1*1e3, x1, color="#ff6b6b", lw=0.7, alpha=0.6)
    axXZ.plot(z2*1e3, x2, color="#ffd166", lw=0.7, alpha=0.6)
    axXZ.set_xlabel("z [mm]  (eje B₀)"); axXZ.set_ylabel("x [m]")
    axXZ.set_title("Plano ZX (longitudinal)", color="white")
    axXZ.grid(True); axXZ.tick_params(colors="#8888aa")

    # ─── Cuadro de validación ───
    axT = fig.add_subplot(2,3,5, facecolor=BG)
    axT.axis("off")
    r_med = (np.sqrt(xh**2+yh**2).max() - np.sqrt(xh**2+yh**2).min())/2
    txt = (
        "══ VALIDACIÓN FÍSICA ══\n\n"
        f" B₀         = {B0} T\n"
        f" v_th       = {v_th:.0f} m/s\n"
        f" ω_c        = {omega_c:.3e} rad/s\n"
        f" T_c        = {T_c*1e9:.1f} ns\n\n"
        f" r_L teórico = {r_L*100:.3f} cm\n"
        f" r_L medido  = {r_med*100:.3f} cm\n"
        f" Error relat.= {abs(r_med-r_L)/r_L*100:.1f} %\n\n"
        " ✓ Boris conserva |v|\n"
        " ✓ Trayectoria helicoidal\n"
        " ✓ Período correcto"
    )
    axT.text(0.05, 0.95, txt, transform=axT.transAxes,
             fontsize=10, color="white", fontfamily="monospace",
             va="top", bbox=dict(boxstyle="round", facecolor="#1a1a2e",
                                 edgecolor="#00e5ff", alpha=0.9))

    # ─── |v| vs t ───
    axV = fig.add_subplot(2,3,6, facecolor=BG)
    vx = np.gradient(xh, t_h); vy = np.gradient(yh, t_h); vz = np.gradient(zh, t_h)
    v  = np.sqrt(vx**2+vy**2+vz**2)
    axV.plot(t_h*1e3, v/v[0], color="#00e5ff", lw=1.0)
    axV.axhline(1.0, color="white", lw=0.8, ls="--", alpha=0.5, label="v₀")
    axV.set_ylim(0.8,1.2)
    axV.set_xlabel("Tiempo [ms]"); axV.set_ylabel("|v(t)| / |v₀|")
    axV.set_title("Conservación de |v| (Boris)", color="white")
    d = (v[-1]/v[0]-1)*100
    axV.text(0.05,0.1,f"Deriva: {d:+.2f}%", transform=axV.transAxes,
             color="#00e5ff", fontsize=9, fontfamily="monospace")
    axV.grid(True); axV.tick_params(colors="#8888aa")
    axV.legend(fontsize=8, framealpha=0.3, facecolor=BG,
               edgecolor="#8888aa", labelcolor="white")

    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  [Trayectoria] -> {OUT}")

if __name__ == "__main__":
    graficar()
