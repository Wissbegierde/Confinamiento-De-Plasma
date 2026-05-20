"""
visualizacion.py
================
Módulo de visualización 3D para el simulador PIC.
- Contorno del recipiente
- Flechas de campo E y B (campo total)
- Animación interactiva sin RecursionError
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

from contenedor import (
    ContenedorCilindrico, ContenedorEsferico,
    ContenedorCaja, ContenedorPlacasParalelas, ContenedorTokamak
)

ESPECIES = {
    "electron":  {"color": "cyan",    "label": "e⁻"},
    "proton":    {"color": "red",     "label": "p⁺"},
    "hidrogeno": {"color": "orange",  "label": "H⁺"},
    "helio":     {"color": "yellow",  "label": "He²⁺"},
    "helio3":    {"color": "green",   "label": "He3²⁺"},
    "deuterio":  {"color": "magenta", "label": "D⁺"},
}

ESCALAS = {
    "s":  ("segundos",      1.0),
    "ms": ("milisegundos",  1e-3),
    "us": ("microsegundos", 1e-6),
    "ns": ("nanosegundos",  1e-9),
}


# ══════════════════════════════════════════════════════════════
#  CONTORNO DEL RECIPIENTE
# ══════════════════════════════════════════════════════════════

def _dibujar_contenedor(ax, contenedor, color="#444466", alpha=0.18):
    """Dibuja el contorno 3D del recipiente."""

    if isinstance(contenedor, ContenedorCilindrico):
        R = contenedor.radio
        H = contenedor.altura
        theta = np.linspace(0, 2*np.pi, 60)
        z_bot = np.full_like(theta, -H/2)
        z_top = np.full_like(theta,  H/2)

        # Círculos superior e inferior
        ax.plot(R*np.cos(theta), R*np.sin(theta), z_bot,
                color=color, linewidth=1.2, alpha=0.7)
        ax.plot(R*np.cos(theta), R*np.sin(theta), z_top,
                color=color, linewidth=1.2, alpha=0.7)

        # Superficie lateral translúcida
        theta2 = np.linspace(0, 2*np.pi, 40)
        z2     = np.linspace(-H/2, H/2, 20)
        TH, ZZ = np.meshgrid(theta2, z2)
        XX = R * np.cos(TH)
        YY = R * np.sin(TH)
        ax.plot_surface(XX, YY, ZZ, alpha=alpha,
                        color=color, linewidth=0, antialiased=True)

        # Líneas verticales
        for ang in np.linspace(0, 2*np.pi, 8, endpoint=False):
            ax.plot([R*np.cos(ang), R*np.cos(ang)],
                    [R*np.sin(ang), R*np.sin(ang)],
                    [-H/2, H/2], color=color, linewidth=0.8, alpha=0.5)

    elif isinstance(contenedor, ContenedorEsferico):
        R = contenedor.radio
        u = np.linspace(0, 2*np.pi, 40)
        v = np.linspace(0,   np.pi, 30)
        X = R * np.outer(np.cos(u), np.sin(v))
        Y = R * np.outer(np.sin(u), np.sin(v))
        Z = R * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(X, Y, Z, alpha=alpha,
                        color=color, linewidth=0, antialiased=True)
        # Ecuador y meridianos
        theta = np.linspace(0, 2*np.pi, 60)
        ax.plot(R*np.cos(theta), R*np.sin(theta),
                np.zeros_like(theta), color=color, linewidth=1.0, alpha=0.7)

    elif isinstance(contenedor, ContenedorCaja):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        Lx, Ly, Lz = contenedor.lim * 2
        hx, hy, hz = Lx/2, Ly/2, Lz/2

        # Caras (6) — para verse como los otros sólidos (con relleno translúcido)
        v000 = (-hx, -hy, -hz)
        v001 = (-hx, -hy,  hz)
        v010 = (-hx,  hy, -hz)
        v011 = (-hx,  hy,  hz)
        v100 = ( hx, -hy, -hz)
        v101 = ( hx, -hy,  hz)
        v110 = ( hx,  hy, -hz)
        v111 = ( hx,  hy,  hz)

        caras = [
            [v000, v100, v110, v010],  # z = -hz
            [v001, v011, v111, v101],  # z = +hz
            [v000, v010, v011, v001],  # x = -hx
            [v100, v101, v111, v110],  # x = +hx
            [v000, v001, v101, v100],  # y = -hy
            [v010, v110, v111, v011],  # y = +hy
        ]

        poly = Poly3DCollection(
            caras,
            facecolors=color,
            edgecolors=color,
            linewidths=0.6,
            alpha=alpha,
        )
        ax.add_collection3d(poly)

        # 12 aristas de la caja
        aristas = [
            [[-hx,hx],[-hy,-hy],[-hz,-hz]],
            [[-hx,hx],[ hy, hy],[-hz,-hz]],
            [[-hx,hx],[-hy,-hy],[ hz, hz]],
            [[-hx,hx],[ hy, hy],[ hz, hz]],
            [[-hx,-hx],[-hy,hy],[-hz,-hz]],
            [[ hx, hx],[-hy,hy],[-hz,-hz]],
            [[-hx,-hx],[-hy,hy],[ hz, hz]],
            [[ hx, hx],[-hy,hy],[ hz, hz]],
            [[-hx,-hx],[-hy,-hy],[-hz,hz]],
            [[ hx, hx],[-hy,-hy],[-hz,hz]],
            [[-hx,-hx],[ hy, hy],[-hz,hz]],
            [[ hx, hx],[ hy, hy],[-hz,hz]],
        ]
        for a in aristas:
            ax.plot(a[0], a[1], a[2], color=color, linewidth=1.2, alpha=0.7)

    elif isinstance(contenedor, ContenedorPlacasParalelas):
        L  = contenedor.L
        d  = contenedor.d
        hl = L/2
        # Placa inferior y superior
        xx, yy = np.meshgrid([-hl, hl], [-hl, hl])
        for z in [-d/2, d/2]:
            ax.plot_surface(xx, yy, np.full_like(xx, z),
                            alpha=alpha*1.5, color=color, linewidth=0)
        # Borde
        theta = np.array([-hl, hl, hl, -hl, -hl])
        phi   = np.array([-hl,-hl, hl,  hl, -hl])
        for z in [-d/2, d/2]:
            ax.plot(theta, phi, np.full(5, z),
                    color=color, linewidth=1.2, alpha=0.7)

    elif isinstance(contenedor, ContenedorTokamak):
        R, a = contenedor.R, contenedor.a
        # Toro paramétrico
        phi   = np.linspace(0, 2*np.pi, 60)
        theta = np.linspace(0, 2*np.pi, 30)
        PH, TH = np.meshgrid(phi, theta)
        X = (R + a*np.cos(TH)) * np.cos(PH)
        Y = (R + a*np.cos(TH)) * np.sin(PH)
        Z = a * np.sin(TH)
        ax.plot_surface(X, Y, Z, alpha=alpha,
                        color=color, linewidth=0, antialiased=True)
        # Círculo central
        ax.plot(R*np.cos(phi), R*np.sin(phi),
                np.zeros_like(phi), color=color, linewidth=1.0, alpha=0.6)


# ══════════════════════════════════════════════════════════════
#  LÍMITES DE VISTA Y MUESTREO
# ══════════════════════════════════════════════════════════════

def _limites_vista(contenedor):
    """Límites del eje 3D en metros (semi-ejes x,y y semi-altura z)."""
    if isinstance(contenedor, ContenedorCilindrico):
        return contenedor.radio * 1.15, contenedor.altura / 2 * 1.15
    if isinstance(contenedor, ContenedorTokamak):
        return (contenedor.R + contenedor.a) * 1.2, contenedor.a * 1.25
    if isinstance(contenedor, ContenedorEsferico):
        return contenedor.radio * 1.15, contenedor.radio * 1.15
    if isinstance(contenedor, ContenedorCaja):
        lim = float(contenedor.lim.max())
        return lim * 1.1, lim * 1.1
    if isinstance(contenedor, ContenedorPlacasParalelas):
        return contenedor.L / 2 * 1.1, contenedor.d / 2 * 1.1
    return 0.02, 0.02


def _puntos_muestreo_campo(contenedor, n_flechas):
    """Puntos dentro del volumen para dibujar flechas E/B."""
    lim_xy, lim_z = _limites_vista(contenedor)
    xs = np.linspace(-lim_xy, lim_xy, n_flechas)
    ys = np.linspace(-lim_xy, lim_xy, n_flechas)
    zs = np.linspace(-lim_z, lim_z, n_flechas)
    puntos = []
    for x in xs:
        for y in ys:
            for z in zs:
                pos = np.array([x, y, z], dtype=float)
                if contenedor.esta_dentro(pos):
                    puntos.append(pos)
    return puntos, min(lim_xy, lim_z) * 0.25


# ══════════════════════════════════════════════════════════════
#  FLECHAS DE CAMPO
# ══════════════════════════════════════════════════════════════

def _dibujar_campos(ax, contenedor, fn_E, fn_B, n_flechas=4):
    """
    Dibuja flechas de E (naranja) y B (azul) en puntos dentro del contenedor.
    n_flechas: puntos por eje en la caja de muestreo.
    """
    puntos, flecha_len = _puntos_muestreo_campo(contenedor, n_flechas)
    if not puntos:
        print("  [3D] Sin puntos interiores para flechas E/B (revisa geometría).")
        return []

    puntos_E = []
    puntos_B = []
    for pos in puntos:
        E = fn_E(pos)
        B = fn_B(pos)
        if np.linalg.norm(E) > 1e-30:
            puntos_E.append((pos, E))
        if np.linalg.norm(B) > 1e-30:
            puntos_B.append((pos, B))

    def _normalizar_flechas(lista, escala):
        """Normaliza todas las flechas a la misma longitud visual."""
        if not lista: return
        vecs = np.array([v for _, v in lista])
        mags = np.linalg.norm(vecs, axis=1, keepdims=True)
        mags = np.where(mags < 1e-30, 1.0, mags)
        vecs_norm = vecs / mags * escala
        return vecs_norm

    # Campo E — naranja
    if puntos_E:
        vecs = _normalizar_flechas(puntos_E, flecha_len)
        for idx, (pos, _) in enumerate(puntos_E):
            v = vecs[idx]
            ax.quiver(pos[0], pos[1], pos[2],
                      v[0], v[1], v[2],
                      color='#FF8C00', alpha=0.75,
                      linewidth=1.2, arrow_length_ratio=0.35)

    # Campo B — azul claro
    if puntos_B:
        vecs = _normalizar_flechas(puntos_B, flecha_len)
        for idx, (pos, _) in enumerate(puntos_B):
            v = vecs[idx]
            ax.quiver(pos[0], pos[1], pos[2],
                      v[0], v[1], v[2],
                      color='#00BFFF', alpha=0.65,
                      linewidth=1.2, arrow_length_ratio=0.35)

    # Leyenda de campos
    e_patch = Line2D([0],[0], color='#FF8C00', linewidth=2,
                     marker='>', markersize=7, label='E total')
    b_patch = Line2D([0],[0], color='#00BFFF', linewidth=2,
                     marker='>', markersize=7, label='B total')
    return [e_patch, b_patch]


# ══════════════════════════════════════════════════════════════
#  VISUALIZACIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════

def lanzar_visualizacion(particulas, colores, contenedor,
                          dt, escala_key, fn_E, fn_B,
                          n_flechas=4):
    """Abre ventana 3D interactiva. Requiere backend TkAgg o Qt5Agg (no Agg)."""
    import matplotlib
    backend = matplotlib.get_backend().lower()
    if "agg" in backend:
        for b in ("TkAgg", "Qt5Agg", "WxAgg"):
            try:
                matplotlib.use(b, force=True)
                print(f"  [3D] Backend interactivo: {b}")
                break
            except Exception:
                continue

    """
    Parámetros
    ----------
    particulas  : lista de objetos Particula (con historia_x)
    colores     : lista de str, un color por partícula
    contenedor  : objeto ContenedorXxx
    dt          : paso de tiempo de la simulación
    escala_key  : 's' | 'ms' | 'us' | 'ns'
    fn_E        : función pos → E_ext (np.array 3,)
    fn_B        : función pos → B_ext (np.array 3,)
    n_flechas   : densidad de flechas por eje (3-5 recomendado)
    """
    _, factor = ESCALAS[escala_key]
    n_frames  = min(len(p.historia_x) for p in particulas)
    if n_frames < 2:
        print("Pocos frames para animar."); return

    historias = [np.array(p.historia_x) for p in particulas]

    # ── Figura ────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 8), facecolor="#0d0d1a")
    fig.suptitle("Simulación de Plasma — PIC 3D",
                 color="white", fontsize=13, y=0.98)

    gs   = gridspec.GridSpec(3, 2, figure=fig,
                             height_ratios=[8, 0.6, 0.6],
                             hspace=0.38, wspace=0.3)
    ax3d = fig.add_subplot(gs[0, :], projection='3d')
    ax3d.set_facecolor("#0d0d1a")
    ax3d.tick_params(colors='white')
    for p in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        p.label.set_color('white')
    ax3d.set_xlabel('X [m]'); ax3d.set_ylabel('Y [m]'); ax3d.set_zlabel('Z [m]')

    lim_xy, lim_z = _limites_vista(contenedor)
    ax3d.set_xlim(-lim_xy, lim_xy)
    ax3d.set_ylim(-lim_xy, lim_xy)
    ax3d.set_zlim(-lim_z, lim_z)
    try:
        ax3d.set_box_aspect((lim_xy, lim_xy, lim_z))
    except Exception:
        pass

    # ── Contenedor ────────────────────────────────────────────
    _dibujar_contenedor(ax3d, contenedor)

    # ── Flechas de campo ──────────────────────────────────────
    campo_handles = _dibujar_campos(ax3d, contenedor, fn_E, fn_B, n_flechas=5)

    # ── Partículas ────────────────────────────────────────────
    TRAIL = 80
    puntos, trazas = [], []
    for c in colores:
        pt, = ax3d.plot([], [], [], 'o', color=c, markersize=7, zorder=5)
        tr, = ax3d.plot([], [], [], '-', color=c, alpha=0.35, linewidth=0.9)
        puntos.append(pt); trazas.append(tr)

    # ── Leyenda ───────────────────────────────────────────────
    vistos = {}
    for c in colores:
        for esp in ESPECIES.values():
            if esp["color"] == c and c not in vistos:
                vistos[c] = esp["label"]
    part_handles = [
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor=c, markersize=7, label=l)
        for c, l in vistos.items()
    ]
    contenedor_handle = Line2D([0],[0], color='#444466',
                                linewidth=1.5, label='Contenedor')
    ax3d.legend(
        handles=part_handles + (campo_handles or []) + [contenedor_handle],
        loc='upper left', framealpha=0.25,
        labelcolor='white', facecolor='#1a1a2e', fontsize=8
    )

    # ── Slider ────────────────────────────────────────────────
    ax_sl  = fig.add_subplot(gs[1, :])
    slider = Slider(ax_sl, '', 0, n_frames-1,
                    valinit=0, valstep=1, color='#4a9eff')
    ax_sl.set_facecolor('#1a1a2e')
    slider.label.set_color('white'); slider.valtext.set_color('white')

    # ── Botones ───────────────────────────────────────────────
    btn_r = Button(fig.add_subplot(gs[2, 0]), '⏮  Reiniciar',
                   color='#1a1a2e', hovercolor='#2a2a4e')
    btn_p = Button(fig.add_subplot(gs[2, 1]), '▶  Play',
                   color='#1a1a2e', hovercolor='#2a2a4e')
    btn_r.label.set_color('white'); btn_p.label.set_color('white')

    st = {"f": 0, "pausado": True, "vel": 1, "busy": False}

    def titulo(f):
        t = f * dt / factor
        return (f"t = {t:.4f} {escala_key}  |  "
                f"frame {f}/{n_frames-1}  |  "
                f"vel ×{st['vel']}   "
                f"[Espacio=play  ←→=frame  +/-=vel  rueda=zoom]")

    def ir_a(f):
        if st["busy"]: return
        st["busy"] = True
        f = int(np.clip(f, 0, n_frames-1))
        st["f"] = f
        for i, h in enumerate(historias):
            pos = h[f]
            puntos[i].set_data([pos[0]], [pos[1]])
            puntos[i].set_3d_properties([pos[2]])
            i0 = max(0, f - TRAIL)
            trazas[i].set_data(h[i0:f+1, 0], h[i0:f+1, 1])
            trazas[i].set_3d_properties(h[i0:f+1, 2])
        ax3d.set_title(titulo(f), color='white', fontsize=8, pad=6)
        slider.eventson = False
        slider.set_val(f)
        slider.eventson = True
        fig.canvas.draw_idle()
        st["busy"] = False

    slider.on_changed(lambda val: (
        st.__setitem__("pausado", True),
        btn_p.label.set_text('▶  Play'),
        ir_a(int(val))
    ) if not st["busy"] else None)

    def on_play(e):
        st["pausado"] = not st["pausado"]
        btn_p.label.set_text('⏸  Pausa' if not st["pausado"] else '▶  Play')

    btn_p.on_clicked(on_play)
    btn_r.on_clicked(lambda e: (
        st.__setitem__("pausado", True),
        btn_p.label.set_text('▶  Play'),
        ir_a(0)
    ))

    def on_tecla(e):
        if e.key == ' ':      on_play(e)
        elif e.key == 'right':
            st["pausado"] = True; btn_p.label.set_text('▶  Play')
            ir_a(st["f"] + st["vel"])
        elif e.key == 'left':
            st["pausado"] = True; btn_p.label.set_text('▶  Play')
            ir_a(st["f"] - st["vel"])
        elif e.key == '+':
            st["vel"] = min(st["vel"]*2, 128)
            ax3d.set_title(titulo(st["f"]), color='white', fontsize=8, pad=6)
            fig.canvas.draw_idle()
        elif e.key == '-':
            st["vel"] = max(st["vel"]//2, 1)
            ax3d.set_title(titulo(st["f"]), color='white', fontsize=8, pad=6)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_tecla)

    def _tick(event):
        if not st["pausado"]:
            ir_a((st["f"] + st["vel"]) % n_frames)

    timer = fig.canvas.new_timer(interval=30)
    timer.add_callback(_tick, None)
    timer.start()

    ir_a(0)
    plt.show()
    timer.stop()
