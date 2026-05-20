"""
gui_principal.py
================
Interfaz gráfica para el simulador PIC de plasma.

Layout:
  ┌────────────────┬──────────────────────────┬──────────────────┐
  │  IZQUIERDA     │         CENTRO           │    DERECHA       │
  │  Configuración │   Vista 3D / Progreso    │ Stats MC / Charts│
  │  Parámetros    │   Paso actual / Choques  │ Decaimiento      │
  │  Especies ±    │   Consola de log         │ Energía cinética │
  │  Campos        │   [INICIAR/PAUSAR/STOP]  │                  │
  └────────────────┴──────────────────────────┴──────────────────┘

Uso:
    cd Lib
    python gui_principal.py

Vista 3D (panel central):
    • Rueda del ratón           → zoom (solo con el cursor sobre el gráfico)
    • Clic izquierdo + arrastra → rotar (elev./azim.; permite ver desde arriba/abajo)
    • Clic central + arrastra   → rotar (alternativa)
    • Clic derecho + arrastra   → desplazar (pan)

Módulos requeridos (misma carpeta):
    main.py, Aplicaciones.py, motor.py, motor_lite.py,
    visualizacion.py, montecarlo.py, contenedor.py,
    campos.py, gestor_corridas.py, tools.py, particulas.py
"""

import sys
import os
import threading
import queue
import io
import time
import traceback

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from matplotlib.lines import Line2D



# ═══════════════════════════════════════════════════════════════════
#  PALETAS — Dark plasma / Light solar
# ═══════════════════════════════════════════════════════════════════
THEMES = {
    "dark": {
        "bg":          "#06080f",
        "panel":       "#0b0e1e",
        "card":        "#0f1428",
        "inp":         "#0c1020",
        "border":      "#1a2b50",
        "cyan":        "#00e5ff",   # más brillante — acento principal
        "blue":        "#4d9fff",   # azul claro legible
        "green":       "#00f080",   # verde neón visible
        "orange":      "#ffaa44",   # naranja cálido
        "red":         "#ff4466",   # rojo vivo
        "purple":      "#bb88ff",   # violeta suave
        "yellow":      "#ffe066",   # amarillo legible sobre oscuro
        "hi":          "#eef4ff",   # texto principal — casi blanco frío
        "mid":         "#99b8e8",   # texto secundario — azul claro
        "lo":          "#4a6a9a",   # texto terciario
        "muted":       "#182040",
        "plot":        "#0d1020",
        "grid":        "#1a2540",
        "btn_start_fg":"#06080f",
        "escaped":     "#888888",   # gris visible sobre fondo oscuro
    },
    "light": {
        "bg":          "#f0f4fb",
        "panel":       "#e2e9f5",
        "card":        "#d4ddf0",
        "inp":         "#ffffff",
        "border":      "#9ab0d0",
        "cyan":        "#0055bb",
        "blue":        "#1144cc",
        "green":       "#006633",
        "orange":      "#cc4400",
        "red":         "#bb0022",
        "purple":      "#5500bb",
        "yellow":      "#7a5500",
        "hi":          "#08101e",
        "mid":         "#203055",
        "lo":          "#4a6080",
        "muted":       "#bccce0",
        "plot":        "#ffffff",
        "grid":        "#c4d0e8",
        "btn_start_fg":"#ffffff",
        "escaped":     "#111111",   # casi negro, visible sobre fondo claro
    },
}

T = dict(THEMES["dark"])   # tema activo — se muta en _apply_theme()

FM  = ("Courier New", 9)
FMB = ("Courier New", 9,  "bold")
FT  = ("Courier New", 11, "bold")
FS  = ("Courier New", 8)
FSB = ("Courier New", 8,  "bold")

ESP_CFG = {
    "electron":  {"color": T["cyan"],   "label": "e⁻",    "abbr": "e⁻"},
    "proton":    {"color": T["red"],    "label": "p⁺",    "abbr": "p⁺"},
    "hidrogeno": {"color": T["orange"], "label": "H⁺",    "abbr": "H⁺"},
    "helio":     {"color": T["yellow"], "label": "He²⁺",  "abbr": "He²⁺"},
    "helio3":    {"color": T["green"],  "label": "He3²⁺", "abbr": "He3²⁺"},
    "deuterio":  {"color": T["purple"], "label": "D⁺",    "abbr": "D⁺"},
}

LOG_TAGS = {
    "[PIC]":   T["cyan"],
    "[Lite]":  T["blue"],
    "[MC]":    T["green"],
    "[3D]":    T["purple"],
    "[AVISO]": T["orange"],
    "ERROR":   T["red"],
    "===":     T["cyan"],
    "✓":       T["green"],
    "▶":       T["cyan"],
}


# ═══════════════════════════════════════════════════════════════════
#  ESTADO GLOBAL COMPARTIDO entre GUI y threads de simulación
# ═══════════════════════════════════════════════════════════════════
class _SimState:
    def __init__(self):
        self.running      = threading.Event()
        self.paused       = threading.Event()   # set = pausado
        self.stop_req     = threading.Event()
        self.step         = 0
        self.total_steps  = 1
        self.particulas   = []
        self.colores      = []
        self.contenedor   = None
        self.escaped_ids  = set()
        self.tiempos_escape = {}
        self.E_cin_historia = []
        self.stats        = {}
        self.last_3d_ts   = 0.0
        self.last_mc_ts   = 0.0
        self.fn_E         = None   # función de campo eléctrico activa
        self.fn_B         = None   # función de campo magnético activa
        self._lock        = threading.Lock()

    def reset(self):
        with self._lock:
            self.step = 0
            self.particulas = []
            self.colores = []
            self.contenedor = None
            self.escaped_ids = set()
            self.tiempos_escape = {}
            self.E_cin_historia = []
            self.stats = {}
            self.last_3d_ts = 0.0
            self.last_mc_ts = 0.0
            self.fn_E = None
            self.fn_B = None


G      = _SimState()
LOG_Q  = queue.Queue()
STAT_Q = queue.Queue()


# ═══════════════════════════════════════════════════════════════════
#  CAPTURA DE STDOUT — envía a la consola Y a LOG_Q
# ═══════════════════════════════════════════════════════════════════
class _Tee(io.TextIOBase):
    def __init__(self, orig):
        self._orig = orig

    def write(self, s):
        try:
            self._orig.write(s)
            self._orig.flush()
        except Exception:
            pass
        line = s.rstrip()
        if line:
            LOG_Q.put(line)
        return len(s)

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
#  VENTANA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════
class SimuladorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("⚛  PIC Plasma Simulator")
        root.configure(bg=T["bg"])
        root.geometry("1680x940")
        root.minsize(1280, 720)

        self._vars     = {}   # tk Variables del formulario
        self._esp_vars = {}   # tk.IntVar por especie
        self._mc_vars  = {}   # tk.StringVar para tabla MC

        # ── Reproducción temporal (slider play/pause) ────────────
        self._playback_active   = False
        self._speed_idx         = 4   # índice en _speeds → x1
        self._speeds            = [0.2, 0.25, 1/3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
        self._speed_labels      = ["x1/5","x1/4","x1/3","x1/2","x1","x2","x3","x4","x5"]
        self._playback_after_id = None

        self._setup_styles()
        self._build_header()
        self._build_main()
        self._build_footer()

        self.root.after(250, self._poll)

    # ─── Estilos TTK ──────────────────────────────────────────────
    def _setup_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        base = dict(background=T["bg"], foreground=T["hi"],
                    font=FM, fieldbackground=T["inp"],
                    troughcolor=T["panel"], bordercolor=T["border"],
                    darkcolor=T["panel"], lightcolor=T["panel"],
                    relief="flat")
        s.configure(".",              **base)
        s.configure("TFrame",         background=T["bg"])
        s.configure("Card.TFrame",    background=T["card"], relief="flat")
        s.configure("TLabel",         background=T["bg"],   foreground=T["hi"])
        s.configure("Card.TLabel",    background=T["card"], foreground=T["hi"])
        s.configure("TEntry",
                    fieldbackground=T["inp"], foreground=T["hi"],
                    insertcolor=T["cyan"], relief="flat")
        s.configure("TCombobox",
                    fieldbackground=T["inp"], foreground=T["hi"],
                    selectbackground=T["border"], relief="flat")
        s.map("TCombobox",
              fieldbackground=[("readonly", T["inp"])],
              foreground=[("readonly", T["hi"])])
        s.configure("TScrollbar",
                    background=T["border"], troughcolor=T["panel"],
                    arrowcolor=T["mid"])
        s.configure("Plasma.Horizontal.TProgressbar",
                    background=T["cyan"], troughcolor=T["muted"],
                    bordercolor=T["border"],
                    lightcolor=T["cyan"], darkcolor=T["cyan"])

    # ─── Cambio de tema ───────────────────────────────────────────
    def _apply_theme(self, name: str):
        """Muta T con los colores del tema y repinta toda la UI."""
        T.update(THEMES[name])
        self._theme_name.set(name)

        # Actualizar estilos TTK
        self._setup_styles()

        # Repintar la raíz y reconstruir secciones visuales
        self.root.configure(bg=T["bg"])

        # Repintar todos los widgets recursivamente
        def _repaint(w):
            cls = w.winfo_class()
            try:
                if cls in ("Frame", "Labelframe"):
                    bg = T["panel"] if w.master and w.master.winfo_class() in ("Frame","Tk") else T["bg"]
                    # Intentar detectar por color actual
                    cur = w.cget("bg")
                    for k in ("bg", "panel", "card", "muted", "plot"):
                        for th in THEMES.values():
                            if cur == th[k]:
                                w.configure(bg=T[k])
                                break
                elif cls == "Label":
                    cur_bg = w.cget("bg")
                    cur_fg = w.cget("fg")
                    for k in ("bg","panel","card","muted","plot","inp"):
                        for th in THEMES.values():
                            if cur_bg == th[k]:
                                w.configure(bg=T[k])
                    for k in ("hi","mid","lo","cyan","green","red","orange","yellow","blue","purple"):
                        for th in THEMES.values():
                            if cur_fg == th[k]:
                                w.configure(fg=T[k])
                elif cls == "Text":
                    cur_bg = w.cget("bg")
                    for k in ("plot","inp","panel","card","bg"):
                        for th in THEMES.values():
                            if cur_bg == th[k]:
                                w.configure(bg=T[k], fg=T["hi"],
                                            insertbackground=T["cyan"])
                elif cls == "Entry":
                    w.configure(bg=T["inp"], fg=T["hi"],
                                insertbackground=T["cyan"],
                                highlightbackground=T["border"],
                                highlightcolor=T["cyan"])
                elif cls == "Button":
                    cur_bg = w.cget("bg")
                    cur_fg = w.cget("fg")
                    for k in ("cyan","card","muted","border","bg","panel"):
                        for th in THEMES.values():
                            if cur_bg == th[k]:
                                w.configure(bg=T[k])
                    for k in ("bg","hi","mid","lo","cyan","red","yellow","green","btn_start_fg"):
                        for th in THEMES.values():
                            if cur_fg == th.get(k, ""):
                                w.configure(fg=T.get(k, T["hi"]))
                elif cls == "Checkbutton" or cls == "Radiobutton":
                    w.configure(bg=T["bg"], fg=T["hi"],
                                activebackground=T["bg"],
                                selectcolor=T["inp"])
                elif cls == "Canvas":
                    cur = w.cget("bg")
                    for k in ("bg","panel","plot","card"):
                        for th in THEMES.values():
                            if cur == th[k]:
                                w.configure(bg=T[k])
            except Exception:
                pass
            for child in w.winfo_children():
                _repaint(child)

        _repaint(self.root)

        # Actualizar botones de tema explícitamente
        try:
            if name == "dark":
                self._btn_dark.configure(bg=T["cyan"], fg=T["btn_start_fg"])
                self._btn_light.configure(bg=T["card"], fg=T["mid"])
            else:
                self._btn_light.configure(bg=T["cyan"], fg=T["btn_start_fg"])
                self._btn_dark.configure(bg=T["card"], fg=T["mid"])
        except Exception:
            pass

        # Repintar start button
        try:
            self.btn_start.configure(bg=T["cyan"], fg=T["btn_start_fg"],
                                     activebackground=T["cyan"])
        except Exception:
            pass

        # Repintar botones de reproducción temporal
        try:
            is_playing = self._playback_active
            self.btn_pb_play.configure(
                fg=T["orange"] if is_playing else T["green"],
                bg=T["card"])
            for b in (self.btn_pb_back, self.btn_pb_fwd,
                      self.btn_pb_slower, self.btn_pb_faster):
                b.configure(bg=T["card"], fg=T["mid"])
            self._speed_lbl.configure(
                bg=T["panel"], fg=T["cyan"])
            self._pb_frame.configure(bg=T["panel"])
        except Exception:
            pass

        # Repintar matplotlib
        try:
            self.fig3d.patch.set_facecolor(T["plot"])
            self.ax3d.set_facecolor(T["plot"])
            self._style_3d(self.ax3d)
            self.canvas3d.draw_idle()
        except Exception:
            pass
        try:
            self.fig_mc.patch.set_facecolor(T["plot"])
            self._style_mc_axes()
            self.canvas_mc.draw_idle()
        except Exception:
            pass

        # Actualizar tags del log
        try:
            for tag, color in LOG_TAGS.items():
                pass  # LOG_TAGS usa T refs — ya actualizado
            for tag, color in {
                "[PIC]": T["cyan"], "[Lite]": T["blue"], "[MC]": T["green"],
                "[3D]": T["purple"], "[AVISO]": T["orange"],
                "ERROR": T["red"], "===": T["cyan"], "✓": T["green"], "▶": T["cyan"],
                "dim": T["lo"], "ok": T["green"], "err": T["red"],
            }.items():
                self.log_txt.tag_config(tag, foreground=color)
        except Exception:
            pass

    # ─── Header ───────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self.root, bg=T["panel"], height=54)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="⚛  PLASMA PIC SIMULATOR",
                 bg=T["panel"], fg=T["cyan"],
                 font=("Courier New", 14, "bold")).pack(side="left", padx=16, pady=10)

        tk.Label(hdr,
                 text="Simulación de partículas cargadas en campos confinados",
                 bg=T["panel"], fg=T["mid"], font=FS).pack(side="left", padx=4)

        # ── Selector de tema ────────────────────────────────────
        self._theme_name = tk.StringVar(value="dark")
        theme_frame = tk.Frame(hdr, bg=T["panel"])
        theme_frame.pack(side="right", padx=(0, 6), pady=10)

        tk.Label(theme_frame, text="Tema:", bg=T["panel"], fg=T["mid"],
                 font=FS).pack(side="left", padx=(0, 4))

        self._btn_dark = tk.Button(
            theme_frame, text="🌙 Oscuro",
            command=lambda: self._apply_theme("dark"),
            bg=T["cyan"], fg=T["bg"],
            relief="flat", font=FS, padx=8, pady=3, cursor="hand2")
        self._btn_dark.pack(side="left", padx=2)

        self._btn_light = tk.Button(
            theme_frame, text="☀ Claro",
            command=lambda: self._apply_theme("light"),
            bg=T["card"], fg=T["mid"],
            relief="flat", font=FS, padx=8, pady=3, cursor="hand2")
        self._btn_light.pack(side="left", padx=2)

        # ── Botones de control en el header ─────────────────────
        btn_frame = tk.Frame(hdr, bg=T["panel"])
        btn_frame.pack(side="right", padx=12, pady=8)

        btn_kw = dict(relief="flat", font=FMB, padx=10, pady=5, cursor="hand2")

        self.btn_stop = tk.Button(
            btn_frame, text="⏹  STOP", command=self._stop_sim,
            bg=T["card"], fg=T["red"],
            activebackground=T["border"],
            state="disabled", **btn_kw)
        self.btn_stop.pack(side="right", padx=(2,0))

        self.btn_pause = tk.Button(
            btn_frame, text="⏸  PAUSAR", command=self._pause_sim,
            bg=T["card"], fg=T["yellow"],
            activebackground=T["border"],
            state="disabled", **btn_kw)
        self.btn_pause.pack(side="right", padx=2)

        self.btn_start = tk.Button(
            btn_frame, text="▶  INICIAR", command=self._start_sim,
            bg=T["cyan"], fg=T["bg"],
            activebackground="#00aace", **btn_kw)
        self.btn_start.pack(side="right", padx=(0,2))

        tk.Frame(self.root, bg=T["border"], height=1).pack(fill="x")

    # ─── Layout 3 columnas ────────────────────────────────────────
    def _build_main(self):
        main = tk.Frame(self.root, bg=T["bg"])
        main.pack(fill="both", expand=True)

        self.left   = tk.Frame(main, bg=T["bg"], width=330)
        self.center = tk.Frame(main, bg=T["bg"])
        self.right  = tk.Frame(main, bg=T["bg"], width=390)

        self.left.pack(  side="left",  fill="y",    padx=(6,3), pady=6)
        self.center.pack(side="left",  fill="both", expand=True, padx=3, pady=6)
        self.right.pack( side="left",  fill="y",    padx=(3,6), pady=6)

        self.left.pack_propagate(False)
        self.right.pack_propagate(False)

        self._build_left()
        self._build_center()
        self._build_right()

    # ══════════════════════════════════════════════════════════════
    #  COLUMNA IZQUIERDA — Configuración con secciones colapsables
    # ══════════════════════════════════════════════════════════════
    def _build_left(self):
        canvas = tk.Canvas(self.left, bg=T["bg"], highlightthickness=0)
        sb = ttk.Scrollbar(self.left, orient="vertical", command=canvas.yview)
        self._cfg_frame = tk.Frame(canvas, bg=T["bg"])
        self._cfg_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._cfg_frame, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        def _scroll(e):
            canvas.yview_scroll(-1*(e.delta//120), "units")

        for w in (canvas, self._cfg_frame):
            w.bind("<MouseWheel>", _scroll)

        f = self._cfg_frame

        # ── Helper: sección colapsable ───────────────────────────
        def collapsible_section(parent, title, color=T["cyan"], open_default=True):
            """Retorna body_frame. El wrapper siempre ocupa su lugar en el padre."""
            state = {"open": open_default}

            # wrapper siempre empaquetado — mantiene la posición relativa
            wrapper = tk.Frame(parent, bg=T["bg"])
            wrapper.pack(fill="x", padx=0, pady=(4, 0))
            wrapper.bind("<MouseWheel>", _scroll)

            hdr = tk.Frame(wrapper, bg=T["muted"], cursor="hand2")
            hdr.pack(fill="x")
            hdr.bind("<MouseWheel>", _scroll)

            arrow_var = tk.StringVar(value="▼" if open_default else "▶")
            arrow_lbl = tk.Label(hdr, textvariable=arrow_var,
                                 bg=T["muted"], fg=color, font=FSB, width=2)
            arrow_lbl.pack(side="left", padx=(6, 0))
            tk.Label(hdr, text=title, bg=T["muted"], fg=color,
                     font=FSB, anchor="w").pack(side="left", padx=4, pady=4)

            # body siempre hijo del wrapper — pack/pack_forget solo afecta dentro del wrapper
            body = tk.Frame(wrapper, bg=T["bg"])
            if open_default:
                body.pack(fill="x", padx=0, pady=(0, 2))

            def toggle(e=None):
                if state["open"]:
                    body.pack_forget()
                    arrow_var.set("▶")
                    state["open"] = False
                else:
                    body.pack(fill="x", padx=0, pady=(0, 2))
                    arrow_var.set("▼")
                    state["open"] = True
                canvas.update_idletasks()
                canvas.configure(scrollregion=canvas.bbox("all"))

            hdr.bind("<Button-1>", toggle)
            arrow_lbl.bind("<Button-1>", toggle)
            for child in hdr.winfo_children():
                child.bind("<Button-1>", toggle)
                child.bind("<MouseWheel>", _scroll)

            return body

        def field(parent, label, key, default="", width=11):
            fr = tk.Frame(parent, bg=T["bg"])
            fr.pack(fill="x", padx=8, pady=2)
            fr.bind("<MouseWheel>", _scroll)
            tk.Label(fr, text=label, bg=T["bg"], fg=T["mid"],
                     font=FS, width=16, anchor="w").pack(side="left")
            v = tk.StringVar(value=str(default))
            self._vars[key] = v
            e = tk.Entry(fr, textvariable=v, font=FM,
                         bg=T["inp"], fg=T["hi"],
                         insertbackground=T["cyan"],
                         relief="flat", bd=1,
                         highlightthickness=1,
                         highlightbackground=T["border"],
                         highlightcolor=T["cyan"],
                         width=width)
            e.pack(side="left", fill="x", expand=True)
            return v

        def dropdown(parent, label, key, values, default=None):
            fr = tk.Frame(parent, bg=T["bg"])
            fr.pack(fill="x", padx=8, pady=2)
            fr.bind("<MouseWheel>", _scroll)
            tk.Label(fr, text=label, bg=T["bg"], fg=T["mid"],
                     font=FS, width=16, anchor="w").pack(side="left")
            v = tk.StringVar(value=default or values[0])
            self._vars[key] = v
            cb = ttk.Combobox(fr, textvariable=v, values=values,
                              font=FM, state="readonly", width=11)
            cb.pack(side="left")
            return v

        def checkbox(parent, label, key, default=True):
            fr = tk.Frame(parent, bg=T["bg"])
            fr.pack(fill="x", padx=8, pady=1)
            fr.bind("<MouseWheel>", _scroll)
            v = tk.BooleanVar(value=default)
            self._vars[key] = v
            tk.Checkbutton(fr, text=label, variable=v,
                           bg=T["bg"], fg=T["hi"],
                           activebackground=T["bg"],
                           selectcolor=T["inp"],
                           font=FS).pack(side="left")
            return v

        # ── Identificación ──────────────────────────────────────
        sec_corrida = collapsible_section(f, "CORRIDA", T["mid"], open_default=True)
        field(sec_corrida, "Etiqueta", "etiqueta", "sim_01")

        # ── Motor ───────────────────────────────────────────────
        sec_motor = collapsible_section(f, "MOTOR DE SIMULACIÓN", T["blue"], open_default=True)
        fr_mot = tk.Frame(sec_motor, bg=T["bg"])
        fr_mot.pack(fill="x", padx=10, pady=2)
        fr_mot.bind("<MouseWheel>", _scroll)
        self._vars["motor"] = tk.StringVar(value="pic")
        for val, txt, color in [
            ("pic",  "PIC completo  (Poisson + colisiones)", T["cyan"]),
            ("lite", "Lite vectorizado  (más rápido)",       T["blue"]),
        ]:
            rb = tk.Radiobutton(
                fr_mot, text=txt, value=val,
                variable=self._vars["motor"],
                bg=T["bg"], fg=color,
                activebackground=T["bg"],
                selectcolor=T["inp"],
                font=FS, anchor="w")
            rb.pack(fill="x", pady=1)
            rb.bind("<MouseWheel>", _scroll)

        # ── Tiempo ──────────────────────────────────────────────
        sec_tiempo = collapsible_section(f, "TIEMPO", T["cyan"])
        field(sec_tiempo, "dt  [s]",   "dt",    "1e-9")
        field(sec_tiempo, "Pasos",     "pasos", "1000")
        dropdown(sec_tiempo, "Escala", "escala", ["s","ms","us","ns"], "us")

        # ── Geometría ───────────────────────────────────────────
        sec_geo = collapsible_section(f, "GEOMETRÍA", T["cyan"])

        # Inicializar variables de geometría con defaults
        for key, val in [("radio","0.01"),("altura","0.02"),
                         ("Lx","0.02"),("Ly","0.02"),("Lz","0.02"),
                         ("separacion","0.01"),("L_placas","0.05"),
                         ("R_mayor","0.05"),("a_menor","0.01")]:
            if key not in self._vars:
                self._vars[key] = tk.StringVar(value=val)

        fr_geo_sel = tk.Frame(sec_geo, bg=T["bg"])
        fr_geo_sel.pack(fill="x", padx=8, pady=2)
        fr_geo_sel.bind("<MouseWheel>", _scroll)
        tk.Label(fr_geo_sel, text="Forma", bg=T["bg"], fg=T["mid"],
                 font=FS, width=16, anchor="w").pack(side="left")
        self._vars["geometria"] = tk.StringVar(value="cilindro")
        cb_geo = ttk.Combobox(fr_geo_sel, textvariable=self._vars["geometria"],
                              values=["cilindro","esfera","caja","placas","tokamak"],
                              font=FM, state="readonly", width=11)
        cb_geo.pack(side="left")

        # Frame dinámico de parámetros geométricos
        self._frame_geo_params = tk.Frame(sec_geo, bg=T["bg"])
        self._frame_geo_params.pack(fill="x")

        # Info de geometría
        self._lbl_geo_info = tk.Label(sec_geo, text="",
                                      bg=T["bg"], fg=T["lo"], font=FS,
                                      wraplength=290, justify="left", anchor="w")
        self._lbl_geo_info.pack(fill="x", padx=10, pady=(0,2))

        GEO_INFO = {
            "cilindro": "Cilindro recto con eje en Z. Radio = radio de la sección, Altura = longitud total.",
            "esfera":   "Esfera centrada en el origen. Solo requiere Radio.",
            "caja":     "Paralelepípedo (caja rectangular). Lx, Ly, Lz son las longitudes totales de cada eje.",
            "placas":   "Dos placas paralelas separadas una distancia d (eje Z). L_lateral define el área de inicialización.",
            "tokamak":  "Toro (donut). R = radio mayor (centro al centro del tubo), a = radio menor (radio del tubo).",
        }

        # Definición de campos por geometría: lista de (label, key, default)
        GEO_CAMPOS = {
            "cilindro": [("Radio  [m]",  "radio",   "0.01"),
                         ("Altura [m]",  "altura",  "0.02")],
            "esfera":   [("Radio  [m]",  "radio",   "0.01")],
            "caja":     [("Lx  [m]",     "Lx",      "0.02"),
                         ("Ly  [m]",     "Ly",      "0.02"),
                         ("Lz  [m]",     "Lz",      "0.02")],
            "placas":   [("Separación d [m]", "separacion", "0.01"),
                         ("L lateral  [m]",   "L_placas",   "0.05")],
            "tokamak":  [("R mayor [m]", "R_mayor",  "0.05"),
                         ("a menor [m]", "a_menor",  "0.01")],
        }

        def _on_geo_change(*_):
            geo = self._vars["geometria"].get()
            self._lbl_geo_info.config(text=GEO_INFO.get(geo, ""))
            # Limpiar frame dinámico
            for w in self._frame_geo_params.winfo_children():
                w.destroy()
            # Crear campos correspondientes
            for lbl, key, default in GEO_CAMPOS.get(geo, []):
                if key not in self._vars:
                    self._vars[key] = tk.StringVar(value=default)
                fr_p = tk.Frame(self._frame_geo_params, bg=T["bg"])
                fr_p.pack(fill="x", padx=8, pady=2)
                fr_p.bind("<MouseWheel>", _scroll)
                tk.Label(fr_p, text=lbl, bg=T["bg"], fg=T["mid"],
                         font=FS, width=16, anchor="w").pack(side="left")
                tk.Entry(fr_p, textvariable=self._vars[key], font=FM,
                         bg=T["inp"], fg=T["hi"],
                         insertbackground=T["cyan"],
                         relief="flat", bd=1,
                         highlightthickness=1,
                         highlightbackground=T["border"],
                         highlightcolor=T["cyan"],
                         width=11).pack(side="left", fill="x", expand=True)
            canvas.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))
            # Actualizar vista previa 3D del contenedor cuando cambia la forma
            self._update_preview_contenedor()

        self._vars["geometria"].trace_add("write", _on_geo_change)
        _on_geo_change()

        # Cuando el usuario modifica parámetros geométricos, refrescar la vista previa.
        def _on_geo_param_change(*_):
            self._update_preview_contenedor()

        for _k in ("radio", "altura", "Lx", "Ly", "Lz",
                   "separacion", "L_placas", "R_mayor", "a_menor"):
            var = self._vars.get(_k)
            if var is not None:
                var.trace_add("write", _on_geo_param_change)

        # ══ CAMPOS ELECTROMAGNÉTICOS — con selección de tipo ════
        sec_campos = collapsible_section(f, "CAMPOS ELECTROMAGNÉTICOS", T["cyan"])

        # ── Campo Magnético ──────────────────────────────────────
        tk.Label(sec_campos, text="  ▸ CAMPO MAGNÉTICO",
                 bg=T["bg"], fg=T["blue"], font=FSB).pack(fill="x", padx=8, pady=(6,1))

        TIPOS_B = [
            "Solenoide (constante axial)",
            "Tokamak (toroidal + poloidal)",
            "Dipolo magnético",
            "Cuadrupolo magnético",
            "Espejo magnético",
            "Cero (sin campo B)",
        ]
        fr_b1 = tk.Frame(sec_campos, bg=T["bg"])
        fr_b1.pack(fill="x", padx=8, pady=2)
        fr_b1.bind("<MouseWheel>", _scroll)
        tk.Label(fr_b1, text="Tipo B:", bg=T["bg"], fg=T["mid"],
                 font=FS, width=16, anchor="w").pack(side="left")
        self._vars["tipo_B"] = tk.StringVar(value=TIPOS_B[0])
        cb_b = ttk.Combobox(fr_b1, textvariable=self._vars["tipo_B"],
                             values=TIPOS_B, font=FS, state="readonly", width=22)
        cb_b.pack(side="left", fill="x", expand=True)

        # Inicializar variables de B con defaults
        for key, val in [("B0","1.0"),("B_radio_sol","0.5"),
                         ("B_R_tok","1.0"),("B_pol_tok","0.1"),
                         ("Bm_espejo","3.0"),("L_espejo","1.0"),
                         ("G_cuad_B","1.0")]:
            if key not in self._vars:
                self._vars[key] = tk.StringVar(value=val)

        # Frame dinámico de parámetros de B
        self._frame_B_params = tk.Frame(sec_campos, bg=T["bg"])
        self._frame_B_params.pack(fill="x")

        # info del tipo B seleccionado
        self._lbl_B_info = tk.Label(sec_campos, text="",
                                    bg=T["bg"], fg=T["lo"], font=FS,
                                    wraplength=290, justify="left", anchor="w")
        self._lbl_B_info.pack(fill="x", padx=10, pady=(0,4))

        B_DESCRIPCIONES = {
            TIPOS_B[0]: "B uniforme a lo largo del eje Z dentro del radio del solenoide.",
            TIPOS_B[1]: "Campo toroidal B∝1/R más componente poloidal (0.1·B₀). Usar con geometría tokamak.",
            TIPOS_B[2]: "Campo dipolar: B∝1/r³. Útil para simular magnetósferas o trampas de partículas.",
            TIPOS_B[3]: "Campo cuadrupolar: componentes lineales en X,Y. Enfoca haces de partículas.",
            TIPOS_B[4]: "Espejo: B₀·(1 + (Bm−1)·(z/L)²). Confinamiento axial por gradiente de campo.",
            TIPOS_B[5]: "Sin campo magnético externo.",
        }

        # Campos por tipo B: (label, key, default)
        B_CAMPOS = {
            TIPOS_B[0]: [("B₀  [T]",       "B0",          "1.0"),
                         ("Radio sol. [m]", "B_radio_sol", "0.5")],
            TIPOS_B[1]: [("B₀  [T]",        "B0",          "1.0"),
                         ("R mayor [m]",    "B_R_tok",     "1.0"),
                         ("B poloidal [T]", "B_pol_tok",   "0.1")],
            TIPOS_B[2]: [("B₀  [T]",        "B0",          "1.0")],
            TIPOS_B[3]: [("Gradiente G",    "G_cuad_B",    "1.0")],
            TIPOS_B[4]: [("B₀  [T]",        "B0",          "1.0"),
                         ("Bm (espejo)",    "Bm_espejo",   "3.0"),
                         ("L espejo [m]",   "L_espejo",    "1.0")],
            TIPOS_B[5]: [],
        }

        def _on_tipo_B_change(*_):
            tipo = self._vars["tipo_B"].get()
            self._lbl_B_info.config(text=B_DESCRIPCIONES.get(tipo, ""))
            # Limpiar frame dinámico
            for w in self._frame_B_params.winfo_children():
                w.destroy()
            campos = B_CAMPOS.get(tipo, [])
            if not campos:
                self._frame_B_params.pack_forget()
            else:
                for lbl, key, default in campos:
                    if key not in self._vars:
                        self._vars[key] = tk.StringVar(value=default)
                    fr_p = tk.Frame(self._frame_B_params, bg=T["bg"])
                    fr_p.pack(fill="x", padx=8, pady=2)
                    fr_p.bind("<MouseWheel>", _scroll)
                    tk.Label(fr_p, text=lbl, bg=T["bg"], fg=T["mid"],
                             font=FS, width=16, anchor="w").pack(side="left")
                    tk.Entry(fr_p, textvariable=self._vars[key], font=FM,
                             bg=T["inp"], fg=T["hi"],
                             insertbackground=T["cyan"],
                             relief="flat", bd=1,
                             highlightthickness=1,
                             highlightbackground=T["border"],
                             highlightcolor=T["cyan"],
                             width=11).pack(side="left", fill="x", expand=True)
                self._frame_B_params.pack(fill="x")
            canvas.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))

        self._vars["tipo_B"].trace_add("write", _on_tipo_B_change)
        _on_tipo_B_change()

        tk.Frame(sec_campos, bg=T["border"], height=1).pack(fill="x", padx=8, pady=(4,0))

        # ── Campo Eléctrico ───────────────────────────────────────
        tk.Label(sec_campos, text="  ▸ CAMPO ELÉCTRICO",
                 bg=T["bg"], fg=T["orange"], font=FSB).pack(fill="x", padx=8, pady=(6,1))

        TIPOS_E = [
            "Cero (sin campo E)",
            "Constante uniforme",
            "Radial centrífugo",
            "Radial centrípeto (confinante)",
            "Oscilante (onda plana)",
            "Cuadrupolo eléctrico",
        ]
        fr_e1 = tk.Frame(sec_campos, bg=T["bg"])
        fr_e1.pack(fill="x", padx=8, pady=2)
        fr_e1.bind("<MouseWheel>", _scroll)
        tk.Label(fr_e1, text="Tipo E:", bg=T["bg"], fg=T["mid"],
                 font=FS, width=16, anchor="w").pack(side="left")
        self._vars["tipo_E"] = tk.StringVar(value=TIPOS_E[0])
        cb_e = ttk.Combobox(fr_e1, textvariable=self._vars["tipo_E"],
                             values=TIPOS_E, font=FS, state="readonly", width=22)
        cb_e.pack(side="left", fill="x", expand=True)

        # Inicializar variables de E con defaults
        for key, val in [("E0x","0.0"),("E0y","0.0"),("E0z","0.0"),
                         ("E0_mag","1.0"),("E_omega","1e6"),("E_k_quad","1.0")]:
            if key not in self._vars:
                self._vars[key] = tk.StringVar(value=val)

        # Parámetros de E (frame dinámico)
        self._frame_E_params = tk.Frame(sec_campos, bg=T["bg"])

        self._lbl_E_info = tk.Label(sec_campos, text="",
                                    bg=T["bg"], fg=T["lo"], font=FS,
                                    wraplength=290, justify="left", anchor="w")
        self._lbl_E_info.pack(fill="x", padx=10, pady=(0,4))

        E_DESCRIPCIONES = {
            TIPOS_E[0]: "Sin campo eléctrico externo.",
            TIPOS_E[1]: "E uniforme especificado por componentes (Ex, Ey, Ez).",
            TIPOS_E[2]: "E = E₀ · r̂ (apunta hacia afuera desde el eje). E₀ es la magnitud.",
            TIPOS_E[3]: "E = −E₀ · r̂ (apunta hacia el eje). Confina partículas radialmente.",
            TIPOS_E[4]: "E oscilante: Ez = E₀·cos(ω·t). Requiere amplitud E₀ y frecuencia ω.",
            TIPOS_E[5]: "Cuadrupolo: E = k·(x, −y, 0). k es la constante de gradiente.",
        }

        # Campos por tipo E: (label, key, default)
        E_CAMPOS = {
            TIPOS_E[0]: [],
            TIPOS_E[1]: [("E₀ x [V/m]", "E0x", "0.0"),
                         ("E₀ y [V/m]", "E0y", "0.0"),
                         ("E₀ z [V/m]", "E0z", "0.0")],
            TIPOS_E[2]: [("E₀  [V/m]",  "E0_mag", "1.0")],
            TIPOS_E[3]: [("E₀  [V/m]",  "E0_mag", "1.0")],
            TIPOS_E[4]: [("E₀  [V/m]",  "E0_mag", "1.0"),
                         ("ω  [rad/s]", "E_omega", "1e6")],
            TIPOS_E[5]: [("k  [V/m²]",  "E_k_quad", "1.0")],
        }

        def _on_tipo_E_change(*_):
            tipo = self._vars["tipo_E"].get()
            self._lbl_E_info.config(text=E_DESCRIPCIONES.get(tipo, ""))
            # Limpiar frame dinámico
            for w in self._frame_E_params.winfo_children():
                w.destroy()
            campos = E_CAMPOS.get(tipo, [])
            if not campos:
                self._frame_E_params.pack_forget()
            else:
                for lbl, key, default in campos:
                    if key not in self._vars:
                        self._vars[key] = tk.StringVar(value=default)
                    fr_p = tk.Frame(self._frame_E_params, bg=T["bg"])
                    fr_p.pack(fill="x", padx=8, pady=2)
                    fr_p.bind("<MouseWheel>", _scroll)
                    tk.Label(fr_p, text=lbl, bg=T["bg"], fg=T["mid"],
                             font=FS, width=16, anchor="w").pack(side="left")
                    tk.Entry(fr_p, textvariable=self._vars[key], font=FM,
                             bg=T["inp"], fg=T["hi"],
                             insertbackground=T["cyan"],
                             relief="flat", bd=1,
                             highlightthickness=1,
                             highlightbackground=T["border"],
                             highlightcolor=T["cyan"],
                             width=11).pack(side="left", fill="x", expand=True)
                self._frame_E_params.pack(fill="x")
            canvas.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))

        self._vars["tipo_E"].trace_add("write", _on_tipo_E_change)
        _on_tipo_E_change()

        # ── Plasma ──────────────────────────────────────────────
        sec_plasma = collapsible_section(f, "PLASMA", T["cyan"])
        field(sec_plasma, "Temperatura [K]",   "T_plasma", "1e4")
        field(sec_plasma, "Colisiones ν [Hz]", "nu",       "500")

        # ── Especies ────────────────────────────────────────────
        sec_esp = collapsible_section(f, "ESPECIES DE PARTÍCULAS", T["green"])
        tk.Label(sec_esp, text="  Cantidad por especie  (0 = omitir):",
                 bg=T["bg"], fg=T["lo"], font=FS).pack(fill="x", padx=8)

        for esp, cfg_esp in ESP_CFG.items():
            fr_e = tk.Frame(sec_esp, bg=T["bg"])
            fr_e.pack(fill="x", padx=8, pady=2)
            fr_e.bind("<MouseWheel>", _scroll)

            tk.Label(fr_e, text="●", fg=cfg_esp["color"],
                     bg=T["bg"], font=FMB).pack(side="left", padx=(0,2))
            tk.Label(fr_e, text=f"{cfg_esp['abbr']:7}", fg=cfg_esp["color"],
                     bg=T["bg"], font=FS, width=8).pack(side="left")

            v = tk.IntVar(value=0)
            self._esp_vars[esp] = v

            def _dec(x=v):  x.set(max(0, x.get() - 1))
            def _inc(x=v):  x.set(x.get() + 1)

            ebtn_kw = dict(relief="flat", bd=0, font=FMB,
                           padx=6, pady=1,
                           activebackground=T["border"],
                           cursor="hand2")
            tk.Button(fr_e, text="−", command=_dec,
                      bg=T["card"], fg=T["mid"], **ebtn_kw).pack(side="left")
            e_cnt = tk.Entry(fr_e, textvariable=v, width=4,
                             bg=T["inp"], fg=T["hi"],
                             insertbackground=T["cyan"],
                             relief="flat", bd=0, font=FM,
                             justify="center")
            e_cnt.pack(side="left")
            tk.Button(fr_e, text="+", command=_inc,
                      bg=T["card"], fg=cfg_esp["color"], **ebtn_kw).pack(side="left")

        # ── Exportaciones ────────────────────────────────────────
        sec_exp = collapsible_section(f, "EXPORTAR", T["mid"], open_default=False)
        checkbox(sec_exp, "Trayectorias CSV",     "exp_tray",  True)
        checkbox(sec_exp, "Análisis Monte Carlo", "exp_mc",    True)
        checkbox(sec_exp, "Mapas de calor",       "exp_mapas", False)
        checkbox(sec_exp, "Visualización 3D",     "exp_3d",    True)

        # ── Simulaciones guardadas ──────────────────────────────
        sec_sims = collapsible_section(f, "SIMULACIONES GUARDADAS", T["green"], open_default=True)

        # Combobox con lista de sims
        fr_sim_sel = tk.Frame(sec_sims, bg=T["bg"])
        fr_sim_sel.pack(fill="x", padx=8, pady=(4, 2))
        fr_sim_sel.bind("<MouseWheel>", _scroll)

        self._sim_selector_var = tk.StringVar(value="— seleccionar —")
        self._sim_selector = ttk.Combobox(
            fr_sim_sel, textvariable=self._sim_selector_var,
            values=[], font=FM, state="readonly", width=22)
        self._sim_selector.pack(side="left", fill="x", expand=True)

        def _refresh_sim_list():
            try:
                from gestor_corridas import BASE_CORRIDAS
                if not BASE_CORRIDAS.exists():
                    return
                sims = sorted(
                    [d.name for d in BASE_CORRIDAS.iterdir()
                     if d.is_dir() and (d / "config.json").exists()],
                    reverse=True)
                self._sim_selector["values"] = sims or ["— sin simulaciones —"]
                if sims:
                    self._sim_selector_var.set(sims[0])
            except Exception:
                pass

        tk.Button(fr_sim_sel, text="↺", command=_refresh_sim_list,
                  bg=T["card"], fg=T["mid"],
                  font=FMB, relief="flat", bd=0,
                  padx=5, pady=1, cursor="hand2").pack(side="left", padx=(4, 0))

        def _load_selected_sim():
            nombre = self._sim_selector_var.get()
            if not nombre or nombre.startswith("—"):
                return
            try:
                from gestor_corridas import BASE_CORRIDAS
                import json
                run_dir = BASE_CORRIDAS / nombre
                cfg_path = run_dir / "config.json"
                if not cfg_path.exists():
                    self._log(f"  [AVISO] No se encontró config.json en {nombre}", "err")
                    return
                with open(cfg_path, encoding="utf-8") as fj:
                    cfg_loaded = json.load(fj)
                self._load_sim_from_disk(run_dir, cfg_loaded)
            except Exception as ex:
                self._log(f"  [AVISO] Error al cargar simulación: {ex}", "err")

        fr_sim_btn = tk.Frame(sec_sims, bg=T["bg"])
        fr_sim_btn.pack(fill="x", padx=8, pady=(2, 6))
        fr_sim_btn.bind("<MouseWheel>", _scroll)

        tk.Button(fr_sim_btn, text="▶  CARGAR SIMULACIÓN",
                  command=_load_selected_sim,
                  bg=T["green"], fg=T["bg"],
                  font=FSB, relief="flat", bd=0,
                  padx=10, pady=4, cursor="hand2").pack(fill="x")

        # Cargar lista inicial (sin bloquear)
        f.after(500, _refresh_sim_list)

        # Espaciado final
        tk.Frame(f, bg=T["bg"], height=10).pack()

    # ══════════════════════════════════════════════════════════════
    #  COLUMNA CENTRAL — Vista 3D + Progreso + Log
    # ══════════════════════════════════════════════════════════════
    def _build_center(self):
        c = self.center

        # ── Banda de estadísticas en tiempo real ────────────────
        band = tk.Frame(c, bg=T["panel"])
        band.pack(fill="x", pady=(0,4))

        stat_row = tk.Frame(band, bg=T["panel"])
        stat_row.pack(fill="x", padx=8, pady=(6,0))

        def badge(key, label, color):
            fr = tk.Frame(stat_row, bg=T["card"], padx=8, pady=4)
            fr.pack(side="left", padx=3)
            tk.Label(fr, text=label, bg=T["card"], fg=T["lo"],
                     font=FS).pack()
            v = tk.StringVar(value="—")
            self._vars[key] = v
            tk.Label(fr, textvariable=v, bg=T["card"], fg=color,
                     font=FMB).pack()
            return v

        badge("stat_total",    "TOTAL",         T["cyan"])
        badge("stat_escaped",  "ESCAPADAS",     T["red"])
        badge("stat_confined", "CONFINADAS",    T["green"])
        badge("stat_hits",     "CHOQUES PARED", T["orange"])
        badge("stat_step",     "PASO",          T["mid"])

        # Botón toggle Partículas / Calor — en la banda de stats
        self._view_mode = tk.StringVar(value="particles")

        def _toggle_view():
            if self._view_mode.get() == "particles":
                self._view_mode.set("heatmap")
                self._btn_view_toggle.config(
                    text="🌡 CALOR", fg=T["orange"], bg=T["muted"])
                self._show_heatmap()
            else:
                self._view_mode.set("particles")
                self._btn_view_toggle.config(
                    text="⚛ PARTÍCULAS", fg=T["cyan"], bg=T["card"])
                self._refresh_3d(G.step)

        self._btn_view_toggle = tk.Button(
            stat_row, text="⚛ PARTÍCULAS",
            command=_toggle_view,
            bg=T["card"], fg=T["cyan"],
            font=("Courier New", 8, "bold"),
            relief="flat", bd=0, padx=10, pady=6, cursor="hand2")
        self._btn_view_toggle.pack(side="right", padx=(4, 3))

        # ── Barra de progreso ────────────────────────────────────
        prog_row = tk.Frame(band, bg=T["panel"])
        prog_row.pack(fill="x", padx=8, pady=(4,6))

        tk.Label(prog_row, text="Progreso  ", bg=T["panel"], fg=T["mid"],
                 font=FS).pack(side="left")
        self._prog_var = tk.DoubleVar(value=0)
        ttk.Progressbar(
            prog_row, variable=self._prog_var,
            maximum=100, style="Plasma.Horizontal.TProgressbar",
            length=100
        ).pack(side="left", fill="x", expand=True, padx=(0,6))
        self._prog_lbl = tk.Label(prog_row, text="0 %",
                                  bg=T["panel"], fg=T["cyan"], font=FS)
        self._prog_lbl.pack(side="left")

        # ── Vista 3D embebida ───────────────────────────────────
        viz = tk.Frame(c, bg=T["panel"])
        viz.pack(fill="both", expand=True, pady=(0,0))

        self.fig3d = Figure(figsize=(6, 5), facecolor=T["plot"])
        self.ax3d  = self.fig3d.add_subplot(111, projection="3d")
        self.ax3d.set_facecolor(T["plot"])
        self._style_3d(self.ax3d)
        self.ax3d.set_title("Sin simulación activa · Configure parámetros",
                            color=T["mid"], fontsize=8)

        self.canvas3d = FigureCanvasTkAgg(self.fig3d, master=viz)
        self.canvas3d.get_tk_widget().pack(fill="both", expand=True)
        self.canvas3d.draw()

        # ── Interacción 3D personalizada ─────────────────────────
        # Rueda = zoom  |  Derecho = rotar  |  Izquierdo = pan
        self._cam = None          # estado de cámara guardado por el usuario
        self._setup_3d_interaction()
        # Vista previa inicial del contenedor según los parámetros por defecto
        self.root.after(50, self._update_preview_contenedor)

        # ── Slider de tiempo (activo sólo cuando hay historia) ──
        slider_f = tk.Frame(c, bg=T["panel"])
        slider_f.pack(fill="x", padx=8, pady=(2, 4))

        tk.Label(slider_f, text="t:", bg=T["panel"], fg=T["mid"],
                 font=FS).pack(side="left")

        self._slider_var = tk.IntVar(value=0)
        self._time_slider = ttk.Scale(
            slider_f, from_=0, to=1,
            orient="horizontal", variable=self._slider_var,
            command=self._on_slider_move)
        self._time_slider.pack(side="left", fill="x", expand=True, padx=(4, 6))
        self._time_slider.state(["disabled"])

        self._slider_lbl = tk.Label(slider_f, text="paso 0",
                                    bg=T["panel"], fg=T["cyan"], font=FS, width=14)
        self._slider_lbl.pack(side="left")

        # ── Controles de reproducción temporal ─────────────────
        pb_f = tk.Frame(c, bg=T["panel"])
        pb_f.pack(fill="x", padx=8, pady=(0, 5))

        _pbkw = dict(relief="flat", font=("Courier New", 9, "bold"),
                     padx=8, pady=3, cursor="hand2", bd=0)

        self.btn_pb_back = tk.Button(
            pb_f, text="◀", command=self._step_back,
            bg=T["card"], fg=T["mid"], state="disabled", **_pbkw)
        self.btn_pb_back.pack(side="left", padx=(0, 2))

        self.btn_pb_play = tk.Button(
            pb_f, text="▶  PLAY", command=self._toggle_playback,
            bg=T["card"], fg=T["green"], state="disabled", **_pbkw)
        self.btn_pb_play.pack(side="left", padx=2)

        self.btn_pb_fwd = tk.Button(
            pb_f, text="▶", command=self._step_forward,
            bg=T["card"], fg=T["mid"], state="disabled", **_pbkw)
        self.btn_pb_fwd.pack(side="left", padx=(2, 12))

        tk.Frame(pb_f, bg=T["border"], width=1).pack(side="left", fill="y", padx=4)

        self.btn_pb_slower = tk.Button(
            pb_f, text="−", command=self._speed_down,
            bg=T["card"], fg=T["mid"], state="disabled", **_pbkw)
        self.btn_pb_slower.pack(side="left", padx=(4, 2))

        self._speed_lbl = tk.Label(
            pb_f, text="x1",
            bg=T["panel"], fg=T["cyan"],
            font=("Courier New", 9, "bold"), width=5, anchor="center")
        self._speed_lbl.pack(side="left", padx=2)

        self.btn_pb_faster = tk.Button(
            pb_f, text="+", command=self._speed_up,
            bg=T["card"], fg=T["mid"], state="disabled", **_pbkw)
        self.btn_pb_faster.pack(side="left", padx=(2, 0))

        # Guardar referencia al frame para repintado de tema
        self._pb_frame = pb_f
        self._pb_btns  = [self.btn_pb_back, self.btn_pb_play,
                          self.btn_pb_fwd, self.btn_pb_slower, self.btn_pb_faster]

        # ── Consola de log ──────────────────────────────────────
        log_f = tk.Frame(c, bg=T["panel"])
        log_f.pack(fill="x", pady=(0,0))

        hdr = tk.Frame(log_f, bg=T["panel"])
        hdr.pack(fill="x", padx=8, pady=(4,0))
        tk.Label(hdr, text="▸ CONSOLA",
                 bg=T["panel"], fg=T["mid"], font=FSB).pack(side="left")
        tk.Button(hdr, text="Limpiar", command=self._log_clear,
                  bg=T["muted"], fg=T["mid"],
                  font=FS, relief="flat", padx=6, pady=1,
                  cursor="hand2").pack(side="right")

        txt_f = tk.Frame(log_f, bg=T["panel"])
        txt_f.pack(fill="both", expand=True, padx=8, pady=4)

        self.log_txt = tk.Text(
            txt_f, height=8,
            bg=T["plot"], fg=T["hi"], font=FS,
            relief="flat", bd=0,
            insertbackground=T["cyan"],
            state="disabled", wrap="word")
        sb_log = ttk.Scrollbar(txt_f, command=self.log_txt.yview)
        self.log_txt.configure(yscrollcommand=sb_log.set)
        self.log_txt.pack(side="left", fill="both", expand=True)
        sb_log.pack(side="right", fill="y")

        for tag, color in LOG_TAGS.items():
            self.log_txt.tag_config(tag, foreground=color)
        self.log_txt.tag_config("dim", foreground=T["lo"])
        self.log_txt.tag_config("ok",  foreground=T["green"])
        self.log_txt.tag_config("err", foreground=T["red"])

    # ══════════════════════════════════════════════════════════════
    #  COLUMNA DERECHA — Stats MC + Gráficas
    # ══════════════════════════════════════════════════════════════
    def _build_right(self):
        r = self.right

        # ── Tabla Monte Carlo ────────────────────────────────────
        mc_f = tk.Frame(r, bg=T["card"])
        mc_f.pack(fill="x", pady=(0,4))

        tk.Label(mc_f, text="  RESULTADOS MONTE CARLO",
                 bg=T["card"], fg=T["cyan"],
                 font=FSB, anchor="w").pack(fill="x", padx=8, pady=(8,2))
        tk.Frame(mc_f, bg=T["border"], height=1).pack(fill="x", padx=8)

        mc_rows = [
            ("n_total",    "Partículas totales",  T["hi"]),
            ("n_escaped",  "Escaparon",           T["red"]),
            ("n_confined", "Confinadas al final", T["green"]),
            ("tau_medio",  "τ medio",             T["cyan"]),
            ("tau_med",    "τ mediana",           T["hi"]),
            ("tau_std",    "Desv. estándar",      T["hi"]),
            ("pct25",      "Percentil 25 %",      T["blue"]),
            ("pct75",      "Percentil 75 %",      T["blue"]),
            ("pct95",      "Percentil 95 %",      T["purple"]),
            ("tau_bohm",   "τ Bohm (teórico)",    T["yellow"]),
            ("ratio_bohm", "τ_sim / τ_Bohm",      T["orange"]),
        ]
        mc_data = tk.Frame(mc_f, bg=T["card"])
        mc_data.pack(fill="x", padx=8, pady=(4,8))

        for key, label, color in mc_rows:
            fr = tk.Frame(mc_data, bg=T["card"])
            fr.pack(fill="x", pady=1)
            tk.Label(fr, text=f"{label}:",
                     bg=T["card"], fg=T["mid"],
                     font=FS, width=20, anchor="w").pack(side="left")
            v = tk.StringVar(value="—")
            self._mc_vars[key] = v
            tk.Label(fr, textvariable=v,
                     bg=T["card"], fg=color,
                     font=FM, anchor="w").pack(side="left")

        # ── Gráficas MC ──────────────────────────────────────────
        chart_f = tk.Frame(r, bg=T["panel"])
        chart_f.pack(fill="both", expand=True, pady=(0,0))

        tk.Label(chart_f, text="  ANÁLISIS ESTADÍSTICO",
                 bg=T["panel"], fg=T["green"],
                 font=FSB, anchor="w").pack(fill="x", padx=8, pady=(6,2))

        self.fig_mc = Figure(figsize=(4, 3.8), facecolor=T["plot"])
        gs = self.fig_mc.add_gridspec(2, 1, hspace=0.55)
        self.ax_dec = self.fig_mc.add_subplot(gs[0])
        self.ax_egy = self.fig_mc.add_subplot(gs[1])
        self._style_mc_axes()

        self.canvas_mc = FigureCanvasTkAgg(self.fig_mc, master=chart_f)
        self.canvas_mc.get_tk_widget().pack(fill="both", expand=True,
                                            padx=4, pady=4)
        self.canvas_mc.draw()

    # ─── Footer ───────────────────────────────────────────────────
    def _build_footer(self):
        tk.Frame(self.root, bg=T["border"], height=1).pack(fill="x")
        ftr = tk.Frame(self.root, bg=T["panel"], height=24)
        ftr.pack(fill="x")
        ftr.pack_propagate(False)
        self._status_var = tk.StringVar(
            value="Listo · Configure los parámetros y presione INICIAR")
        tk.Label(ftr, textvariable=self._status_var,
                 bg=T["panel"], fg=T["mid"], font=FS).pack(side="left", padx=10)

    # ─── Interacción 3D ────────────────────────────
    def _sync_3d_box_aspect(self):
        """
        Proporción x:y:z según límites actuales.
        Si z es muy pequeño frente a xy (tokamak, cilindro bajo), el sólido 3D
        puede verse aplastado o desaparecer; se impone un mínimo visual en z.
        """
        ax = self.ax3d
        try:
            x0, x1 = ax.get_xlim3d()
            y0, y1 = ax.get_ylim3d()
            z0, z1 = ax.get_zlim3d()
            rx = max(x1 - x0, 1e-15)
            ry = max(y1 - y0, 1e-15)
            rz = max(z1 - z0, 1e-15)
            r_xy = max(rx, ry)
            # Mínimo ~20 % del diámetro horizontal para que las mallas se vean
            rz_vis = max(rz, 0.20 * r_xy)
            ax.set_box_aspect((rx, ry, rz_vis))
        except Exception:
            pass

    def _setup_3d_interaction(self):
        """
        Interacción manual (sin mouse_init por defecto):
          • Rueda                    → zoom al centro
          • Clic izquierdo + arrastre → rotar (elev / azim), desde arriba y abajo
          • Clic derecho + arrastre  → pan
          • Clic central + arrastre  → rotar (alternativa)
        `elev` se limita a (-89°, 89°) para evitar singularidades; `azim` ∈ [0,360).
        """
        ax     = self.ax3d
        canvas = self.canvas3d

        # Desconectar eventos por defecto de Axes3D
        for attr in ("_id_drag", "_id_press", "_id_release"):
            cid = getattr(ax, attr, None)
            if cid is not None:
                try:
                    canvas.mpl_disconnect(cid)
                except Exception:
                    pass
        try:
            ax.mouse_init(rotate_btn=[], zoom_btn=[])
        except Exception:
            pass

        self._drag_info = None   # estado interno del arrastre
        sens_e, sens_a = 0.55, 0.55  # grados por píxel (más cómodo ver polo z)

        def _on_press(event):
            if event.inaxes != ax:
                return
            if event.button == 1 or event.button == 2:   # izq / central → rotar
                self._drag_info = {
                    "mode": "rotate",
                    "x0": event.x, "y0": event.y,
                    "elev0": float(ax.elev),
                    "azim0": float(ax.azim),
                }
            elif event.button == 3:        # derecha → pan
                self._drag_info = {
                    "mode": "pan",
                    "x0": event.x, "y0": event.y,
                    "xlim": ax.get_xlim3d(),
                    "ylim": ax.get_ylim3d(),
                    "zlim": ax.get_zlim3d(),
                }

        def _on_motion(event):
            if self._drag_info is None:
                return
            dx = event.x - self._drag_info["x0"]
            dy = event.y - self._drag_info["y0"]

            if self._drag_info["mode"] == "rotate":
                elev = self._drag_info["elev0"] - dy * sens_e
                elev = float(np.clip(elev, -89.0, 89.0))
                azim = (self._drag_info["azim0"] - dx * sens_a) % 360.0
                ax.view_init(elev=elev, azim=azim)
                canvas.draw_idle()

            elif self._drag_info["mode"] == "pan":
                w, h = canvas.get_width_height()
                xl = self._drag_info["xlim"]
                yl = self._drag_info["ylim"]
                zl = self._drag_info["zlim"]
                sx = (xl[1] - xl[0]) / max(w, 1)
                sy = (yl[1] - yl[0]) / max(h, 1)
                sz = (zl[1] - zl[0]) / max(h, 1)
                ax.set_xlim3d(xl[0] - dx * sx,  xl[1] - dx * sx)
                ax.set_ylim3d(yl[0] - dy * sy,  yl[1] - dy * sy)
                ax.set_zlim3d(zl[0] + dy * sz,  zl[1] + dy * sz)
                canvas.draw_idle()

        def _on_release(event):
            self._drag_info = None
            self._sync_3d_box_aspect()
            self._save_cam()

        def _on_scroll(event):
            if event.inaxes != ax:
                return
            factor = 0.88 if event.button == "up" else 1.14
            for get_lim, set_lim in [
                (ax.get_xlim3d, ax.set_xlim3d),
                (ax.get_ylim3d, ax.set_ylim3d),
                (ax.get_zlim3d, ax.set_zlim3d),
            ]:
                lo, hi = get_lim()
                mid  = (lo + hi) / 2.0
                half = (hi - lo) / 2.0 * factor
                set_lim(mid - half, mid + half)
            self._sync_3d_box_aspect()
            self._save_cam()
            canvas.draw_idle()

        canvas.mpl_connect("button_press_event",   _on_press)
        canvas.mpl_connect("motion_notify_event",  _on_motion)
        canvas.mpl_connect("button_release_event", _on_release)
        canvas.mpl_connect("scroll_event",         _on_scroll)

    def _save_cam(self):
        """Snapshot del estado de cámara actual (elev, azim, límites)."""
        try:
            self._cam = {
                "elev": self.ax3d.elev,
                "azim": self.ax3d.azim,
                "xlim": self.ax3d.get_xlim3d(),
                "ylim": self.ax3d.get_ylim3d(),
                "zlim": self.ax3d.get_zlim3d(),
            }
        except Exception:
            pass

    def _restore_cam(self):
        """Restaura el snapshot de cámara (si existe).  No-op en primer render."""
        if self._cam is None:
            return
        try:
            self.ax3d.view_init(elev=self._cam["elev"], azim=self._cam["azim"])
            # Solo reaplicar zoom/pan guardado si el origen sigue dentro del encuadre.
            # La geometría (caja, toro, etc.) está centrada en (0,0,0); límites viejos
            # tras un pan exagerado dejan el volumen invisible (p. ej. ejes 0.2–0.8 m).
            def _encuadra_origen(tupla_lim):
                lo, hi = float(tupla_lim[0]), float(tupla_lim[1])
                a, b = (lo, hi) if lo <= hi else (hi, lo)
                return a <= 0.0 <= b

            if all(
                _encuadra_origen(self._cam[k])
                for k in ("xlim", "ylim", "zlim")
            ):
                self.ax3d.set_xlim3d(self._cam["xlim"])
                self.ax3d.set_ylim3d(self._cam["ylim"])
                self.ax3d.set_zlim3d(self._cam["zlim"])
        except Exception:
            pass

    # ─── Estilos matplotlib ───────────────────────────────────────
    def _style_3d(self, ax):
        ax.set_facecolor(T["plot"])
        self.fig3d.patch.set_facecolor(T["plot"])
        ax.tick_params(colors=T["mid"], labelsize=6)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.label.set_color(T["mid"])
            axis.pane.fill = True
            # Paneles adaptados al tema: oscuro=azul sutil, claro=gris suave
            pane_clr = "#2a3a5022" if T["plot"] == "#0d1020" else "#c0c8d833"
            axis.pane.set_facecolor(pane_clr)
            axis.pane.set_edgecolor(T["grid"] + "55")
        ax.set_xlabel("X [m]", fontsize=7, color=T["mid"])
        ax.set_ylabel("Y [m]", fontsize=7, color=T["mid"])
        ax.set_zlabel("Z [m]", fontsize=7, color=T["mid"])
        # Grid muy sutil — casi invisible para no tapar trayectorias
        ax.grid(True, color=T["grid"], linewidth=0.25, alpha=0.2)

    def _style_mc_axes(self):
        for ax, title, color in [
            (self.ax_dec, "Partículas confinadas vs tiempo", T["green"]),
            (self.ax_egy, "Energía cinética media",          T["cyan"]),
        ]:
            ax.set_facecolor(T["plot"])
            ax.tick_params(colors=T["lo"], labelsize=6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("bottom", "left"):
                ax.spines[sp].set_color(T["grid"])
            ax.set_title(title, color=color, fontsize=7, pad=2)
        self.fig_mc.patch.set_facecolor(T["plot"])
        self._layout_mc_figure()

    def _layout_mc_figure(self):
        """Evita tight_layout (incompatible con algunos GridSpec / versiones)."""
        self.fig_mc.subplots_adjust(
            left=0.16, right=0.97, top=0.94, bottom=0.09, hspace=0.42,
        )

    # ──────────────────────────────────────────────────────────────
    #  SLIDER DE TIEMPO — navegar por la historia de la simulación
    # ──────────────────────────────────────────────────────────────
    def _on_slider_move(self, val):
        """Dibuja el estado de las partículas en el paso del slider."""
        try:
            step = int(float(val))
            if not G.particulas or G.contenedor is None:
                return
            # Número máximo de pasos registrados
            max_steps = max((len(p.historia_x) for p in G.particulas), default=0)
            if max_steps == 0:
                return
            step = min(step, max_steps - 1)

            # En modo mapa de calor: actualizar heatmap acumulado hasta este paso
            if self._view_mode.get() == "heatmap":
                self._slider_lbl.config(text=f"paso {step}")
                self._show_heatmap(up_to_step=step)
                return
            dt_val = float(self._vars["dt"].get())
            escala = self._vars["escala"].get()
            from visualizacion import ESCALAS, _dibujar_contenedor, _limites_vista
            _, factor = ESCALAS.get(escala, ("s", 1.0))
            t_show = step * dt_val / factor

            self._slider_lbl.config(text=f"paso {step}  {t_show:.3f} {escala}")

            self.ax3d.cla()
            self._style_3d(self.ax3d)
            _dibujar_contenedor(self.ax3d, G.contenedor)
            self._repaint_contenedor(self.ax3d)
            lim_xy, lim_z = _limites_vista(G.contenedor)
            self.ax3d.set_xlim(-lim_xy, lim_xy)
            self.ax3d.set_ylim(-lim_xy, lim_xy)
            self.ax3d.set_zlim(-lim_z,  lim_z)
            # Restaurar cámara del usuario (zoom/rot/pan) si ya interactuó
            self._restore_cam()
            self._sync_3d_box_aspect()

            # Líneas de campo
            self._dibujar_flechas_campo(self.ax3d, lim_xy, lim_z)

            for i, (p, c) in enumerate(zip(G.particulas, G.colores)):
                hist = p.historia_x
                if not hist or step >= len(hist):
                    continue
                h = np.array(hist[:step+1])
                pos = h[-1]
                # FIX: antes usaba `i in G.escaped_ids and step >= len(hist)-1`
                # → solo ponía negro en el ÚLTIMO frame; nunca durante playback.
                # Ahora compara el tiempo de escape (por p.id) con el del slider.
                t_escape = G.tiempos_escape.get(p.id, None)
                escaped  = (t_escape is not None) and (t_escape <= step * dt_val + 1e-30)
                tc = self._particle_color(c)
                dot_color = T["escaped"] if escaped else tc
                trail_start = max(0, len(h) - 80)
                self.ax3d.plot(h[trail_start:,0], h[trail_start:,1], h[trail_start:,2],
                               "-", color=tc, alpha=0.55 if not escaped else 0.2,
                               linewidth=0.9 if not escaped else 0.5)
                self.ax3d.plot([pos[0]], [pos[1]], [pos[2]],
                               "o", color=dot_color, markersize=5,
                               markeredgewidth=0.4,
                               markeredgecolor="#ffffff44" if not escaped else "#00000033",
                               zorder=5)

            n_esc_at_step = sum(
                1 for pid, t in G.tiempos_escape.items()
                if t <= step * dt_val
            )
            self.ax3d.set_title(
                f"t = {t_show:.3f} {escala}  paso {step}/{max_steps-1}  |  "                f"escapadas: {n_esc_at_step}/{len(G.particulas)}",
                color=T["hi"], fontsize=8, pad=4)
            self.canvas3d.draw_idle()
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════
    #  LEER CONFIGURACIÓN DESDE EL FORMULARIO
    # ══════════════════════════════════════════════════════════════
    def _get_config(self) -> dict:
        v = self._vars

        def _fget(key, default=0.0):
            """Lee un float de self._vars[key] con fallback."""
            var = v.get(key)
            if var is None:
                return default
            try:
                return float(var.get())
            except Exception:
                return default

        try:
            conteos = {k: self._esp_vars[k].get()
                       for k in self._esp_vars
                       if self._esp_vars[k].get() > 0}
            if not conteos:
                conteos = {"electron": 20}

            geo = v["geometria"].get()

            # ── Parámetros geométricos según forma ───────────────
            geo_params = {
                "radio":      _fget("radio",      0.01),
                "altura":     _fget("altura",     0.02),
                "Lx":         _fget("Lx",         0.02),
                "Ly":         _fget("Ly",         0.02),
                "Lz":         _fget("Lz",         0.02),
                "separacion": _fget("separacion", 0.01),
                "L_placas":   _fget("L_placas",   0.05),
                "R_mayor":    _fget("R_mayor",    0.05),
                "a_menor":    _fget("a_menor",    0.01),
            }

            # ── Parámetros de campo B ────────────────────────────
            b_params = {
                "B0":          _fget("B0",          1.0),
                "B_radio_sol": _fget("B_radio_sol", 0.5),
                "B_R_tok":     _fget("B_R_tok",     1.0),
                "B_pol_tok":   _fget("B_pol_tok",   0.1),
                "Bm_espejo":   _fget("Bm_espejo",   3.0),
                "L_espejo":    _fget("L_espejo",    1.0),
                "G_cuad_B":    _fget("G_cuad_B",    1.0),
            }

            # ── Parámetros de campo E ────────────────────────────
            e_params = {
                "E0x":      _fget("E0x",      0.0),
                "E0y":      _fget("E0y",      0.0),
                "E0z":      _fget("E0z",      0.0),
                "E0_mag":   _fget("E0_mag",   1.0),
                "E_omega":  _fget("E_omega",  1e6),
                "E_k_quad": _fget("E_k_quad", 1.0),
            }

            return {
                "etiqueta":    v["etiqueta"].get().strip() or "sim",
                "motor":       v["motor"].get(),
                "dt":          float(v["dt"].get()),
                "pasos":       int(float(v["pasos"].get())),
                "escala":      v["escala"].get(),
                "geometria":   geo,
                # geometría (retrocompatibilidad: radio/altura siempre presentes)
                **geo_params,
                # B
                "tipo_B":      v["tipo_B"].get(),
                **b_params,
                # E
                "tipo_E":      v["tipo_E"].get(),
                "E0":          [e_params["E0x"], e_params["E0y"], e_params["E0z"]],
                **e_params,
                # Plasma
                "T_plasma":    float(v["T_plasma"].get()),
                "nu_colision": float(v["nu"].get()),
                "conteos":     conteos,
                "n_total":     sum(conteos.values()),
                "exportar": {
                    "trayectorias":     v["exp_tray"].get(),
                    "montecarlo":       v["exp_mc"].get(),
                    "mapas_calor":      v["exp_mapas"].get(),
                    "visualizacion_3d": False,
                },
            }
        except Exception as e:
            raise ValueError(f"Error en parámetros: {e}")

    # ══════════════════════════════════════════════════════════════
    #  CONTROL DE SIMULACIÓN
    # ══════════════════════════════════════════════════════════════
    def _start_sim(self):
        if G.running.is_set():
            return
        try:
            cfg = self._get_config()
        except ValueError as e:
            messagebox.showerror("Parámetros inválidos", str(e))
            return

        G.stop_req.clear()
        G.paused.clear()
        G.running.set()
        G.reset()
        G.total_steps = cfg["pasos"]
        self._cam = None          # resetear cámara al comenzar nueva simulación

        # Inicializar badges
        self._vars["stat_total"].set(str(cfg["n_total"]))
        self._vars["stat_escaped"].set("0")
        self._vars["stat_confined"].set(str(cfg["n_total"]))
        self._vars["stat_hits"].set("0")
        self._vars["stat_step"].set(f"0/{cfg['pasos']}")
        self._prog_var.set(0)
        self._prog_lbl.config(text="0 %")

        self.btn_start.config(state="disabled", bg=T["lo"])
        self.btn_pause.config(state="normal",   text="⏸  PAUSAR")
        self.btn_stop.config( state="normal")
        self._status_var.set(
            f"▶  Simulación en progreso  [{cfg['motor'].upper()}]  "
            f"— {cfg['n_total']} partículas")

        self._log_clear()
        # Resetear slider al inicio
        try:
            self._time_slider.configure(to=1)
            self._slider_var.set(0)
            self._time_slider.state(["disabled"])
            self._slider_lbl.config(text="paso 0")
            # Detener y deshabilitar reproducción temporal
            self._playback_active = False
            if self._playback_after_id is not None:
                self.root.after_cancel(self._playback_after_id)
                self._playback_after_id = None
            self.btn_pb_play.config(text="▶  PLAY", fg=T["green"])
            self._set_playback_btns_state("disabled")
        except Exception:
            pass
        self._log(
            f"▶ INICIO  etiqueta={cfg['etiqueta']}  "
            f"motor={cfg['motor'].upper()}  "
            f"N={cfg['n_total']}  pasos={cfg['pasos']}", "ok")

        sys.stdout = _Tee(sys.__stdout__)

        threading.Thread(
            target=self._sim_thread, args=(cfg,), daemon=True).start()

    def _pause_sim(self):
        if not G.running.is_set():
            return
        if G.paused.is_set():
            G.paused.clear()
            self.btn_pause.config(text="⏸  PAUSAR")
            self._status_var.set("▶  Simulación reanudada")
        else:
            G.paused.set()
            self.btn_pause.config(text="▶  REANUDAR")
            self._status_var.set("⏸  Simulación pausada — presione REANUDAR")

    def _stop_sim(self):
        if G.running.is_set():
            G.stop_req.set()
            G.paused.clear()
            self._status_var.set("⏹  Deteniendo… (espere el paso actual)")

    # ══════════════════════════════════════════════════════════════
    #  HILO DE SIMULACIÓN
    # ══════════════════════════════════════════════════════════════
    def _sim_thread(self, cfg: dict):
        try:
            from main import guardar_salidas, _tau_bohm_ref
            from gestor_corridas import GestorCorrida
            from Aplicaciones import construir_particulas
            import campos as campos_mod
            from contenedor import (ContenedorCilindrico, ContenedorEsferico,
                                    ContenedorCaja, ContenedorPlacasParalelas,
                                    ContenedorTokamak)

            # ── Crear contenedor ────────────────────────────────
            def _make_contenedor():
                geo = cfg["geometria"]
                if geo == "cilindro":
                    return ContenedorCilindrico(
                        radio=cfg["radio"],
                        altura=cfg["altura"])
                if geo == "esfera":
                    return ContenedorEsferico(radio=cfg["radio"])
                if geo == "caja":
                    return ContenedorCaja(
                        Lx=cfg["Lx"],
                        Ly=cfg["Ly"],
                        Lz=cfg["Lz"])
                if geo == "placas":
                    return ContenedorPlacasParalelas(
                        d=cfg["separacion"],
                        L=cfg["L_placas"])
                if geo == "tokamak":
                    return ContenedorTokamak(
                        R=cfg["R_mayor"],
                        a=cfg["a_menor"])
                raise ValueError(f"Geometría desconocida: {geo}")

            contenedor = _make_contenedor()

            # ── Crear partículas ANTES del motor ─────────────────
            # Así el hilo GUI puede leer posiciones en tiempo real
            particulas, motores, colores = construir_particulas(
                cfg["conteos"], contenedor, cfg["dt"],
                T_plasma=cfg["T_plasma"],
                nu=cfg["nu_colision"],
            )
            # Exponer en estado global
            with G._lock:
                G.particulas  = particulas
                G.colores     = colores
                G.contenedor  = contenedor

            # ── Campos — construir funciones según tipo seleccionado ──
            B0    = cfg["B0"]
            radio = cfg["radio"]
            E0_arr = np.array(cfg["E0"])
            tipo_B = cfg.get("tipo_B", "Solenoide (constante axial)")
            tipo_E = cfg.get("tipo_E", "Cero (sin campo E)")

            # Campo B
            if tipo_B == "Cero (sin campo B)":
                fn_B = lambda pos: np.zeros(3)
            elif tipo_B == "Tokamak (toroidal + poloidal)":
                fn_B = lambda pos: campos_mod.campo_magnetico_tokamak(
                    pos, B0=B0, R=radio, Bpol=0.1*B0)
            elif tipo_B == "Dipolo magnético":
                def fn_B(pos, _B0=B0):
                    r = np.linalg.norm(pos)
                    if r < 1e-12:
                        return np.array([0., 0., _B0])
                    r5 = r**5
                    m = np.array([0., 0., _B0 * radio**3])
                    return (3*np.dot(m, pos)*pos/r5**1 - m/r**3) * 1e-7 * 4*np.pi
            elif tipo_B == "Cuadrupolo magnético":
                fn_B = lambda pos, _B0=B0: np.array([_B0*pos[1], _B0*pos[0], 0.])
            elif tipo_B == "Espejo magnético":
                def fn_B(pos, _B0=B0, _L=max(cfg.get("altura", 0.02), 1e-6)):
                    z = pos[2]
                    Bz = _B0 * (1.0 + (z/_L)**2)
                    return np.array([0., 0., Bz])
            else:  # Solenoide por defecto
                fn_B = lambda pos: campos_mod.campo_magnetico_solenoide(
                    pos, B0=B0, radio=radio)

            # Campo E
            if tipo_E == "Cero (sin campo E)":
                fn_E = lambda pos: np.zeros(3)
            elif tipo_E == "Constante uniforme":
                fn_E = lambda pos: campos_mod.campo_electrico_constante(pos, E0=E0_arr)
            elif tipo_E == "Radial centrífugo":
                def fn_E(pos, _mag=np.linalg.norm(E0_arr) or 1.0):
                    r = np.array([pos[0], pos[1], 0.])
                    rn = np.linalg.norm(r)
                    return _mag * r / max(rn, 1e-12)
            elif tipo_E == "Radial centrípeto (confinante)":
                def fn_E(pos, _mag=np.linalg.norm(E0_arr) or 1.0):
                    r = np.array([pos[0], pos[1], 0.])
                    rn = np.linalg.norm(r)
                    return -_mag * r / max(rn, 1e-12)
            elif tipo_E == "Oscilante (onda plana)":
                _t_ref = [0.0]
                _omega = 2*np.pi * 1e6
                def fn_E(pos, _Ez=E0_arr[2], _w=_omega, _t=_t_ref):
                    _t[0] += 1e-9
                    return np.array([0., 0., _Ez * np.sin(_w * _t[0])])
            elif tipo_E == "Cuadrupolo eléctrico":
                def fn_E(pos, _mag=np.linalg.norm(E0_arr) or 1.0):
                    return np.array([_mag*pos[0], -_mag*pos[1], 0.])
            else:
                fn_E = lambda pos: campos_mod.campo_electrico_constante(pos, E0=E0_arr)

            print(f"  [GUI] Campo B: {tipo_B}  |  Campo E: {tipo_E}")

            # Exponer funciones de campo al estado global para visualización
            with G._lock:
                G.fn_E = fn_E
                G.fn_B = fn_B

            # ── Ejecutar motor ───────────────────────────────────
            if cfg["motor"] == "lite":
                from motor_lite import motor_lite, campo_B_solenoide_vec, campo_E_cero_vec
                fn_B_vec = lambda X: np.array([fn_B(x) for x in X])
                if tipo_E == "Cero (sin campo E)":
                    fn_E_lite = campo_E_cero_vec
                else:
                    fn_E_lite = lambda X: np.array([fn_E(x) for x in X])

                print(f"\n  [Lite] Corriendo {cfg['pasos']} pasos, "
                      f"N={len(particulas)} ...")

                # Intentar pasar estado_global si motor_lite lo soporta
                try:
                    tiempos_escape, E_cin_historia = motor_lite(
                        pasos=cfg["pasos"], particulas=particulas,
                        motores_colision=motores,
                        fn_E=fn_E_lite, fn_B=fn_B_vec,
                        dt=cfg["dt"], contenedor=contenedor,
                        registrar_energia=True, verbose=True,
                        paused_event=G.paused, stop_event=G.stop_req,
                        estado_global=G,
                    )
                except TypeError:
                    # motor_lite antiguo — sin soporte de eventos/estado
                    tiempos_escape, E_cin_historia = motor_lite(
                        pasos=cfg["pasos"], particulas=particulas,
                        motores_colision=motores,
                        fn_E=fn_E_lite, fn_B=fn_B_vec,
                        dt=cfg["dt"], contenedor=contenedor,
                        registrar_energia=True, verbose=True,
                    )
                fn_E_out, fn_B_out = fn_E_lite, fn_B_vec

            else:  # PIC
                from motor import motor_simulacion, _configurar_rejilla
                from Aplicaciones import guardar_cache, cargar_cache, _nombre_cache, CACHE_DIR

                print("\n  [PIC] Preparando rejilla Poisson …")
                rejilla = _configurar_rejilla(contenedor, (30,30,30), fn_E, fn_B)
                nombre_cache = _nombre_cache(
                    cfg["geometria"],
                    cfg.get("radio", cfg.get("R_mayor", 0.01)),
                    cfg.get("altura", cfg.get("separacion", cfg.get("Lz", 0.02))),
                    B0, cfg["E0"])
                os.makedirs(CACHE_DIR, exist_ok=True)
                if not cargar_cache(rejilla, nombre_cache):
                    guardar_cache(rejilla, nombre_cache)

                print(f"\n  [PIC] Corriendo {cfg['pasos']} pasos, "
                      f"N={len(particulas)} ...")
                tiempos_escape, E_cin_historia = motor_simulacion(
                    pasos=cfg["pasos"], particulas=particulas,
                    motores_colision=motores, n=cfg["n_total"],
                    B0=B0, E0=cfg["E0"], dt=cfg["dt"],
                    contenedor=contenedor,
                    resolucion_grilla=(30,30,30),
                    registrar_energia=True,
                    fn_E_ext=fn_E, fn_B_ext=fn_B,
                    paused_event=G.paused,
                    stop_event=G.stop_req,
                    estado_global=G,
                )
                fn_E_out, fn_B_out = fn_E, fn_B

            # Actualizar estado compartido
            with G._lock:
                G.tiempos_escape  = tiempos_escape
                G.E_cin_historia  = E_cin_historia
                G.escaped_ids     = set(tiempos_escape.keys())
                G.step            = cfg["pasos"]

            # ── Guardar salidas ──────────────────────────────────
            gestor  = GestorCorrida()
            # Usar solo la etiqueta del usuario como nombre de carpeta
            import re as _re, shutil as _shutil
            etiqueta_limpia = _re.sub(r"[^\w\-]", "_",
                                      cfg.get("etiqueta","sim").strip() or "sim")
            run_dir_auto = gestor.crear_carpeta(cfg)
            from gestor_corridas import BASE_CORRIDAS as _BC
            run_dir_target = _BC / etiqueta_limpia
            if run_dir_target.exists() and run_dir_target != run_dir_auto:
                _n = 1
                while (_BC / f"{etiqueta_limpia}_{_n}").exists():
                    _n += 1
                run_dir_target = _BC / f"{etiqueta_limpia}_{_n}"
            try:
                if run_dir_auto.exists() and not run_dir_target.exists():
                    run_dir_auto.rename(run_dir_target)
                    run_dir = run_dir_target
                else:
                    run_dir = run_dir_auto
            except Exception:
                run_dir = run_dir_auto
            cfg["run_id"] = run_dir.name

            resultado = {
                "particulas":     particulas,
                "motores":        motores,
                "colores":        colores,
                "contenedor":     contenedor,
                "tiempos_escape": tiempos_escape,
                "E_cin_historia": E_cin_historia,
                "fn_E": fn_E_out,
                "fn_B": fn_B_out,
            }

            if G.stop_req.is_set():
                LOG_Q.put("  [AVISO] Simulación detenida — guardando datos parciales…")
                try:
                    import json as _json
                    from tools import guardar_logs_trayectorias
                    from main import _guardar_tiempos_escape, _guardar_energia

                    os.makedirs(run_dir, exist_ok=True)

                    # ── Trayectorias CSV ─────────────────────────
                    if cfg.get("exportar", {}).get("trayectorias", True):
                        guardar_logs_trayectorias(
                            particulas, cfg["dt"],
                            carpeta_salida=os.path.join(run_dir, "trayectorias"),
                        )

                    # ── Monte Carlo parcial ──────────────────────
                    dir_mc = os.path.join(run_dir, "montecarlo")
                    os.makedirs(dir_mc, exist_ok=True)
                    _guardar_tiempos_escape(tiempos_escape, dir_mc)
                    _guardar_energia(E_cin_historia, dir_mc)

                    # ── config.json marcado como parcial ─────────
                    cfg["run_id"]  = run_dir.name
                    cfg["estado"]  = "detenido_parcial"
                    cfg["paso_detenido"] = G.step
                    cfg_path = os.path.join(run_dir, "config.json")
                    with open(cfg_path, "w", encoding="utf-8") as _f:
                        _json.dump(cfg, _f, indent=2, ensure_ascii=False)

                    LOG_Q.put(f"  [AVISO] Datos parciales guardados → {run_dir}")

                    # ── Estadísticas parciales para la GUI ───────
                    import montecarlo as _mc_mod
                    if tiempos_escape:
                        _stats = _mc_mod.calcular_tau(
                            tiempos_escape, cfg["n_total"],
                            cfg["dt"], cfg["pasos"])
                        _tau_ref = _tau_bohm_ref(cfg)
                        G.stats = _stats
                        STAT_Q.put(("mc_done", _stats, _tau_ref,
                                    cfg["n_total"], len(tiempos_escape)))
                except Exception as _e:
                    LOG_Q.put(f"  [AVISO] Guardado parcial falló: {_e}")
            else:
                stats = guardar_salidas(cfg, run_dir, resultado)
                G.stats = stats
                tau_ref = _tau_bohm_ref(cfg)
                STAT_Q.put(("mc_done", stats, tau_ref,
                            cfg["n_total"], len(tiempos_escape)))

            STAT_Q.put(("done", str(run_dir)))

        except Exception as e:
            err_msg = traceback.format_exc()
            LOG_Q.put(f"ERROR: {e}")
            for line in err_msg.splitlines():
                LOG_Q.put(line)
            STAT_Q.put(("error", str(e)))

        finally:
            sys.stdout = sys.__stdout__
            G.running.clear()

    # ══════════════════════════════════════════════════════════════
    #  POLLING — Actualizar GUI desde colas cada 250 ms
    # ══════════════════════════════════════════════════════════════
    def _poll(self):
        # ── Vaciar cola de logs ──────────────────────────────────
        try:
            while True:
                line = LOG_Q.get_nowait()
                self._log(line)
        except queue.Empty:
            pass

        # ── Vaciar cola de stats ─────────────────────────────────
        try:
            while True:
                item = STAT_Q.get_nowait()
                self._handle_stat(item)
        except queue.Empty:
            pass

        # ── Actualizar progreso en tiempo real ───────────────────
        if G.running.is_set() and G.particulas:
            try:
                steps_done = min(len(p.historia_x) for p in G.particulas)
                G.step = steps_done
                total  = max(1, G.total_steps)
                pct    = min(100.0, steps_done / total * 100)

                self._prog_var.set(pct)
                self._prog_lbl.config(text=f"{pct:.0f} %")
                self._vars["stat_step"].set(f"{steps_done}/{total}")

                n_esc  = len(G.escaped_ids)
                n_tot  = len(G.particulas)
                n_conf = n_tot - n_esc
                self._vars["stat_escaped"].set(str(n_esc))
                self._vars["stat_confined"].set(str(max(0, n_conf)))
                self._vars["stat_hits"].set(str(n_esc))

                # Actualizar 3D / mapa de calor cada ~1.5 s
                now = time.time()
                if now - G.last_3d_ts > 1.5 and G.contenedor is not None:
                    G.last_3d_ts = now
                    if self._view_mode.get() == "heatmap":
                        self._show_heatmap()
                    else:
                        self._update_3d_live()

                # Actualizar MC y gráficas en tiempo real cada ~2 s
                # (siempre, no solo cuando hay escapes — la energía cinética
                #  se muestra desde el primer paso aunque no haya fugados)
                if now - getattr(G, "last_mc_ts", 0.0) > 2.0:
                    G.last_mc_ts = now
                    self._update_mc_realtime()

            except Exception:
                pass

        self.root.after(250, self._poll)

    def _handle_stat(self, item):
        kind = item[0]

        if kind == "mc_done":
            _, stats, tau_ref, n_total, n_esc = item
            self._update_mc_table(stats, tau_ref, n_total, n_esc)
            self._update_mc_charts()

        elif kind == "done":
            run_dir = item[1]
            self._prog_var.set(100)
            self._prog_lbl.config(text="100 %")
            self._status_var.set(f"✓  Completado → {run_dir}")
            self._log(f"✓  Corrida finalizada: {run_dir}", "ok")
            self.btn_start.config(state="normal", bg=T["cyan"], fg=T["btn_start_fg"])
            self.btn_pause.config(state="disabled", text="⏸  PAUSAR")
            self.btn_stop.config( state="disabled")
            if G.particulas and G.contenedor is not None:
                self._update_3d_final()

        elif kind == "error":
            self._status_var.set(f"✗  Error: {item[1]}")
            self.btn_start.config(state="normal",   bg=T["cyan"])
            self.btn_pause.config(state="disabled", text="⏸  PAUSAR")
            self.btn_stop.config( state="disabled")

    # ──────────────────────────────────────────────────────────────
    #  LOG
    # ──────────────────────────────────────────────────────────────
    def _log(self, msg: str, force_tag: str = ""):
        self.log_txt.config(state="normal")
        tag = force_tag
        if not tag:
            for key, _ in LOG_TAGS.items():
                if key in msg:
                    tag = key
                    break
        ts = time.strftime("%H:%M:%S")
        self.log_txt.insert("end", f"[{ts}] {msg}\n", tag or "")
        self.log_txt.see("end")
        self.log_txt.config(state="disabled")

    def _log_clear(self):
        self.log_txt.config(state="normal")
        self.log_txt.delete("1.0", "end")
        self.log_txt.config(state="disabled")

    # ──────────────────────────────────────────────────────────────
    #  VISTA 3D — actualización en tiempo real
    # ──────────────────────────────────────────────────────────────
    def _update_3d_live(self):
        try:
            from visualizacion import _dibujar_contenedor, _limites_vista
            self.ax3d.cla()
            self._style_3d(self.ax3d)
            _dibujar_contenedor(self.ax3d, G.contenedor)
            # Repintar contenedor: color cian muy transparente para no tapar partículas
            self._repaint_contenedor(self.ax3d)
            lim_xy, lim_z = _limites_vista(G.contenedor)
            self.ax3d.set_xlim(-lim_xy, lim_xy)
            self.ax3d.set_ylim(-lim_xy, lim_xy)
            self.ax3d.set_zlim(-lim_z,  lim_z)
            # Restaurar cámara del usuario (zoom/rot/pan) si ya interactuó
            self._restore_cam()
            self._sync_3d_box_aspect()

            # ── Líneas de campo E y B ─────────────────────────────
            self._dibujar_flechas_campo(self.ax3d, lim_xy, lim_z)

            for i, (p, c) in enumerate(zip(G.particulas, G.colores)):
                hist = p.historia_x
                if not hist:
                    continue
                hist_arr = np.array(hist)
                pos      = hist_arr[-1]

                # FIX: comparar p.id (no índice i) — funciona tanto en sim en vivo
                # como al cargar desde disco (donde escaped_ids contiene los IDs reales).
                escaped   = p.id in G.escaped_ids
                tc        = self._particle_color(c)
                dot_color = T["escaped"] if escaped else tc
                # Trayectoria: más visible (alpha mayor)
                alpha_tr  = 0.18 if escaped else 0.55
                lw        = 0.5  if escaped else 0.9

                trail = hist_arr[max(0, len(hist_arr)-80):]
                self.ax3d.plot(trail[:,0], trail[:,1], trail[:,2],
                               "-", color=tc, alpha=alpha_tr, linewidth=lw)
                self.ax3d.plot([pos[0]], [pos[1]], [pos[2]],
                               "o", color=dot_color, markersize=5,
                               markeredgewidth=0.4,
                               markeredgecolor="#ffffff44" if not escaped else "#00000033",
                               zorder=5)

            step  = G.step
            total = G.total_steps
            n_esc = len(G.escaped_ids)
            n_tot = len(G.particulas)
            self.ax3d.set_title(
                f"t REAL  paso {step}/{total}  |  "
                f"confinadas {n_tot-n_esc}/{n_tot}",
                color=T["hi"], fontsize=8, pad=4)
            self.canvas3d.draw_idle()
        except Exception:
            pass

    # ── Vista 3D final tras completar simulación ──────────────────
    def _update_3d_final(self):
        try:
            from visualizacion import _dibujar_contenedor, _limites_vista
            self.ax3d.cla()
            self._style_3d(self.ax3d)
            _dibujar_contenedor(self.ax3d, G.contenedor)
            self._repaint_contenedor(self.ax3d)
            lim_xy, lim_z = _limites_vista(G.contenedor)
            self.ax3d.set_xlim(-lim_xy, lim_xy)
            self.ax3d.set_ylim(-lim_xy, lim_xy)
            self.ax3d.set_zlim(-lim_z,  lim_z)
            # Restaurar cámara del usuario (zoom/rot/pan) si ya interactuó
            self._restore_cam()
            self._sync_3d_box_aspect()

            # ── Líneas de campo E y B ─────────────────────────────
            self._dibujar_flechas_campo(self.ax3d, lim_xy, lim_z)

            for i, (p, c) in enumerate(zip(G.particulas, G.colores)):
                hist = p.historia_x
                if not hist:
                    continue
                h = np.array(hist)
                # FIX: usar p.id en vez de i — consistente con escaped_ids al cargar
                escaped   = p.id in G.escaped_ids
                tc        = self._particle_color(c)
                dot_color = T["escaped"] if escaped else tc
                alpha     = 0.3     if escaped else 0.85

                step = max(1, len(h)//200)
                self.ax3d.plot(h[::step,0], h[::step,1], h[::step,2],
                               "-", color=tc, alpha=0.18, linewidth=0.6)
                pos = h[-1]
                self.ax3d.plot([pos[0]], [pos[1]], [pos[2]],
                               "o", color=dot_color,
                               markersize=6, alpha=alpha, zorder=5)

            n_esc = len(G.escaped_ids)
            n_tot = len(G.particulas)
            pct   = 100*n_esc/max(1, n_tot)
            self.ax3d.set_title(
                f"SIMULACIÓN COMPLETA  |  {n_tot} partículas  |  "
                f"{n_esc} escapadas ({pct:.0f} %)  "
                f"|  ● gris = chocó pared",
                color=T["cyan"], fontsize=8, pad=4)
            self.canvas3d.draw_idle()

            # Activar slider de tiempo
            max_steps = max((len(p.historia_x) for p in G.particulas), default=1)
            self._time_slider.configure(to=max_steps - 1)
            self._slider_var.set(max_steps - 1)
            self._time_slider.state(["!disabled"])
            self._slider_lbl.config(text=f"paso {max_steps-1}")

            # Habilitar controles de reproducción temporal
            self._set_playback_btns_state("normal")
            self._speed_lbl.config(text=self._speed_labels[self._speed_idx])
        except Exception:
            pass

    # ── Vista previa estática del contenedor (sin simulación) ─────
    def _update_preview_contenedor(self):
        """
        Dibuja únicamente la geometría del contenedor 3D a partir del formulario,
        sin requerir que haya una simulación en marcha.
        """
        try:
            from visualizacion import _dibujar_contenedor, _limites_vista
            from contenedor import (ContenedorCilindrico, ContenedorEsferico,
                                    ContenedorCaja, ContenedorPlacasParalelas,
                                    ContenedorTokamak)

            cfg = self._get_config()
            geo = cfg["geometria"]

            if geo == "cilindro":
                cont = ContenedorCilindrico(
                    radio=cfg["radio"],
                    altura=cfg["altura"])
            elif geo == "esfera":
                cont = ContenedorEsferico(radio=cfg["radio"])
            elif geo == "caja":
                cont = ContenedorCaja(
                    Lx=cfg["Lx"],
                    Ly=cfg["Ly"],
                    Lz=cfg["Lz"])
            elif geo == "placas":
                cont = ContenedorPlacasParalelas(
                    d=cfg["separacion"],
                    L=cfg["L_placas"])
            elif geo == "tokamak":
                cont = ContenedorTokamak(
                    R=cfg["R_mayor"],
                    a=cfg["a_menor"])
            else:
                return

            # Exponerlo en el estado global para que el resto de vistas lo use
            with G._lock:
                G.contenedor = cont

            self.ax3d.cla()
            self._style_3d(self.ax3d)
            _dibujar_contenedor(self.ax3d, cont)
            self._repaint_contenedor(self.ax3d)
            lim_xy, lim_z = _limites_vista(cont)
            self.ax3d.set_xlim(-lim_xy, lim_xy)
            self.ax3d.set_ylim(-lim_xy, lim_xy)
            self.ax3d.set_zlim(-lim_z,  lim_z)
            self._sync_3d_box_aspect()
            self.ax3d.set_title(
                "Vista previa del contenedor (sin simulación)",
                color=T["mid"], fontsize=8, pad=4)
            self.canvas3d.draw_idle()
        except Exception:
            # En caso de parámetros inválidos o import fallido, no rompemos la GUI
            pass

    # ──────────────────────────────────────────────────────────────
    #  REPINTAR CONTENEDOR — aplicar colores distinguibles
    # ──────────────────────────────────────────────────────────────
    def _repaint_contenedor(self, ax):
        """
        Tras _dibujar_contenedor, ajusta mallas 3D al tema.
        (Evitar hex tipo #RRGGBB08: el byte final es alfa ~3 % y deja el sólido invisible.)
        """
        try:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

            is_dark = (T["plot"] == "#0d1020")
            # Caras translúcidas pero visibles (α ~0.14–0.18)
            if is_dark:
                face_c = (0.0, 0.85, 1.0, 0.16)
                edge_c = (0.4, 0.95, 1.0, 0.7)
                line_c = (0.4, 0.95, 1.0, 0.65)
            else:
                face_c = (0.0, 0.25, 0.55, 0.14)
                edge_c = (0.0, 0.2, 0.45, 0.65)
                line_c = (0.0, 0.2, 0.45, 0.55)

            for col in ax.collections:
                if isinstance(col, Poly3DCollection):
                    col.set_facecolor(face_c)
                    col.set_edgecolor(edge_c)
                    col.set_linewidth(0.6)
                elif isinstance(col, Line3DCollection):
                    col.set_color(line_c)
                    col.set_linewidth(0.75)

            # Algunos contenedores (p. ej. `ContenedorCaja`) se dibujan con `ax.plot`
            # (Line3D), no como *Collection*. Recoloreamos SOLO las líneas existentes
            # en este punto del pipeline (antes de trayectorias/partículas).
            for ln in getattr(ax, "lines", []):
                try:
                    c = ln.get_color()
                    if c in ("#444466", "#444466ff", "#444466FF"):
                        ln.set_color(line_c)
                        ln.set_alpha(line_c[3] if isinstance(line_c, tuple) and len(line_c) == 4 else 0.65)
                        ln.set_linewidth(1.1)
                except Exception:
                    pass
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────
    #  FLECHAS DE CAMPO E y B en la vista 3D
    # ──────────────────────────────────────────────────────────────
    def _dibujar_flechas_campo(self, ax, lim_xy, lim_z, n=3):
        """
        Dibuja una grilla n×n×n de vectores E (naranja) y B (azul)
        en la vista 3D. Solo actúa si G.fn_E / G.fn_B están definidos.
        """
        try:
            if G.fn_E is None and G.fn_B is None:
                return
            coords = np.linspace(-lim_xy * 0.7, lim_xy * 0.7, n)
            coords_z = np.linspace(-lim_z * 0.7, lim_z * 0.7, n)
            xs, ys, zs = np.meshgrid(coords, coords, coords_z, indexing="ij")
            pts = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])

            scale_xy = lim_xy * 0.35 / max(n, 1)
            scale_z  = lim_z  * 0.35 / max(n, 1)

            for fn, color, label in [
                (G.fn_E, T["orange"], "E"),
                (G.fn_B, T["blue"],   "B"),
            ]:
                if fn is None:
                    continue
                for pt in pts:
                    try:
                        vec = np.asarray(fn(pt), dtype=float)
                        mag = np.linalg.norm(vec)
                        if mag < 1e-30:
                            continue
                        unit = vec / mag
                        dx, dy, dz = unit[0]*scale_xy, unit[1]*scale_xy, unit[2]*scale_z
                        ax.quiver(pt[0], pt[1], pt[2],
                                  dx, dy, dz,
                                  color=color, alpha=0.45,
                                  linewidth=0.7, arrow_length_ratio=0.35)
                    except Exception:
                        pass
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────
    #  COLORES DE PARTÍCULAS — adapta CSS names al tema activo
    # ──────────────────────────────────────────────────────────────
    # Mapeo de colores CSS (usados en Aplicaciones.ESPECIES) a claves del tema
    _CSS_TO_THEME = {
        "cyan":    "cyan",
        "red":     "red",
        "orange":  "orange",
        "yellow":  "yellow",
        "green":   "green",
        "magenta": "purple",
        "blue":    "blue",
        "purple":  "purple",
        "white":   "hi",
    }

    def _particle_color(self, css_color: str) -> str:
        """
        Convierte el color CSS de la especie al color correcto del tema activo.
        En modo oscuro los colores brillantes (cyan, yellow) son visibles;
        en modo claro se mapean a versiones oscuras legibles sobre fondo blanco.
        """
        key = self._CSS_TO_THEME.get(css_color, None)
        if key and key in T:
            return T[key]
        # Si ya es un hex o un color no reconocido, devolver tal cual
        return css_color

    # ──────────────────────────────────────────────────────────────
    #  REPRODUCCIÓN TEMPORAL — play/pause/step/velocidad
    # ──────────────────────────────────────────────────────────────
    def _set_playback_btns_state(self, state: str):
        """Habilita o deshabilita todos los botones de reproducción."""
        for b in self._pb_btns:
            try:
                b.config(state=state)
            except Exception:
                pass

    def _toggle_playback(self):
        """Alterna entre PLAY y PAUSE de la reproducción temporal."""
        if self._playback_active:
            self._playback_active = False
            if self._playback_after_id is not None:
                self.root.after_cancel(self._playback_after_id)
                self._playback_after_id = None
            self.btn_pb_play.config(text="▶  PLAY", fg=T["green"])
        else:
            # Solo si el slider está habilitado (simulación terminada)
            try:
                if "disabled" in self._time_slider.state():
                    return
            except Exception:
                return
            # Si está en el último paso, volver al inicio
            max_s = int(self._time_slider.cget("to"))
            if int(self._slider_var.get()) >= max_s:
                self._slider_var.set(0)
                self._on_slider_move(0)
            self._playback_active = True
            self.btn_pb_play.config(text="⏸  PAUSA", fg=T["orange"])
            self._playback_tick()

    def _playback_tick(self):
        """Avanza la animación temporal un tick."""
        if not self._playback_active:
            return
        speed   = self._speeds[self._speed_idx]
        step    = int(self._slider_var.get())
        max_s   = int(self._time_slider.cget("to"))
        # Cuántos pasos avanzar por tick
        steps_per_tick = max(1, int(round(speed)))
        new_step = min(step + steps_per_tick, max_s)
        self._slider_var.set(new_step)
        self._on_slider_move(new_step)
        if new_step >= max_s:
            # Llegó al final — detener
            self._playback_active = False
            self.btn_pb_play.config(text="▶  PLAY", fg=T["green"])
            return
        # Delay: base 80ms / speed (para velocidades bajas, más lento)
        if speed < 1.0:
            delay_ms = int(80 / speed)
        else:
            delay_ms = 80
        self._playback_after_id = self.root.after(delay_ms, self._playback_tick)

    def _step_forward(self):
        """Avanza exactamente un paso en el tiempo."""
        try:
            if "disabled" in self._time_slider.state():
                return
        except Exception:
            return
        # Detener play si estaba activo
        if self._playback_active:
            self._toggle_playback()
        step  = int(self._slider_var.get())
        max_s = int(self._time_slider.cget("to"))
        new_s = min(step + 1, max_s)
        self._slider_var.set(new_s)
        self._on_slider_move(new_s)

    def _step_back(self):
        """Retrocede exactamente un paso en el tiempo."""
        try:
            if "disabled" in self._time_slider.state():
                return
        except Exception:
            return
        if self._playback_active:
            self._toggle_playback()
        step  = int(self._slider_var.get())
        new_s = max(step - 1, 0)
        self._slider_var.set(new_s)
        self._on_slider_move(new_s)

    def _speed_up(self):
        """Aumenta la velocidad de reproducción (máx x5)."""
        if self._speed_idx < len(self._speeds) - 1:
            self._speed_idx += 1
        self._speed_lbl.config(text=self._speed_labels[self._speed_idx])

    def _speed_down(self):
        """Disminuye la velocidad de reproducción (mín x1/5)."""
        if self._speed_idx > 0:
            self._speed_idx -= 1
        self._speed_lbl.config(text=self._speed_labels[self._speed_idx])

    # ──────────────────────────────────────────────────────────────
    #  MAPA DE CALOR
    # ──────────────────────────────────────────────────────────────
    def _show_heatmap(self, up_to_step=None):
        """Muestra el mapa de calor XY. Acepta up_to_step para slider/playback."""
        try:
            from visualizacion import _limites_vista

            if not G.particulas or G.contenedor is None:
                return

            # Recoger posiciones según contexto:
            # - up_to_step dado (slider/playback): historia acumulada hasta ese paso
            # - is_running (tiempo real): historia acumulada con stride (eficiente)
            # - sim terminada sin slider: toda la trayectoria
            xs, ys = [], []
            is_running = G.running.is_set()
            for p in G.particulas:
                hist = p.historia_x
                if not hist:
                    continue
                if up_to_step is not None:
                    # Slider: acumular posiciones hasta el paso indicado
                    end = min(up_to_step + 1, len(hist))
                    h = np.array(hist[:end])
                    xs.extend(h[:, 0].tolist())
                    ys.extend(h[:, 1].tolist())
                elif is_running:
                    # En tiempo real: historia acumulada (stride para rendimiento)
                    h = np.array(hist)
                    stride = max(1, len(h) // 200)
                    xs.extend(h[::stride, 0].tolist())
                    ys.extend(h[::stride, 1].tolist())
                else:
                    # Sim terminada: acumular toda la trayectoria
                    h = np.array(hist)
                    xs.extend(h[:, 0].tolist())
                    ys.extend(h[:, 1].tolist())

            if not xs:
                return

            xs = np.array(xs)
            ys = np.array(ys)
            lim_xy, lim_z = _limites_vista(G.contenedor)

            # Reconstruir figura 2D si aún no está en modo calor
            # (se detecta porque ax3d ya no tiene projection='3d')
            need_rebuild = True
            try:
                need_rebuild = hasattr(self.ax3d, 'get_proj')  # True = sigue siendo 3D
            except Exception:
                need_rebuild = True

            if need_rebuild:
                self.fig3d.clf()
                self._ax_heatmap = self.fig3d.add_subplot(111)
                self.fig3d.patch.set_facecolor(T["plot"])
                self._heatmap_cb = None   # colorbar se crea la primera vez

            ax = self._ax_heatmap
            ax.cla()
            ax.set_facecolor(T["plot"])

            bins = 50
            rng  = [[-lim_xy, lim_xy], [-lim_xy, lim_xy]]
            counts, xedges, yedges = np.histogram2d(xs, ys, bins=bins, range=rng)

            # Usar imshow para rendimiento: es mucho más rápido que hist2d en cada frame
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im = ax.imshow(
                counts.T,
                origin="lower", extent=extent,
                aspect="equal", cmap="plasma",
                interpolation="gaussian",
            )

            # Colorbar: crear solo una vez, actualizar en actualizaciones siguientes
            if getattr(self, "_heatmap_cb", None) is None:
                self._heatmap_cb = self.fig3d.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                self._heatmap_cb.set_label("Densidad de partículas", color=T["mid"], fontsize=7)
            else:
                self._heatmap_cb.update_normal(im)
            self._heatmap_cb.ax.tick_params(colors=T["lo"], labelsize=6)

            # Etiquetas y título
            n_esc = len(G.escaped_ids)
            n_tot = len(G.particulas)
            paso  = up_to_step if up_to_step is not None else G.step
            titulo = (f"Mapa de calor XY — paso {paso}"
                      + (f"  |  {n_esc}/{n_tot} escapadas" if n_tot else ""))
            ax.set_title(titulo, color=T["orange"], fontsize=8, pad=4)
            ax.set_xlabel("X [m]", color=T["mid"], fontsize=8)
            ax.set_ylabel("Y [m]", color=T["mid"], fontsize=8)
            ax.tick_params(colors=T["lo"], labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(T["border"])

            self.canvas3d.draw_idle()
        except Exception as ex:
            self._log(f"  [AVISO] Mapa de calor: {ex}", "err")

    def _refresh_3d(self, step=None):
        """Restaura la vista 3D normal tras salir del mapa de calor."""
        try:
            # Reconstruir subplot 3D
            self.fig3d.clf()
            self.ax3d = self.fig3d.add_subplot(111, projection="3d")
            self.ax3d.set_facecolor(T["plot"])
            self._style_3d(self.ax3d)
            self._setup_3d_interaction()

            if G.particulas and G.contenedor is not None:
                if step is not None:
                    self._on_slider_move(step)
                else:
                    self._update_3d_final()
            else:
                self.ax3d.set_title(
                    "Sin simulación activa · Configure y presione INICIAR",
                    color=T["mid"], fontsize=8)
            self.canvas3d.draw_idle()
        except Exception as ex:
            self._log(f"  [AVISO] _refresh_3d: {ex}", "err")

    # ──────────────────────────────────────────────────────────────
    #  CARGAR SIMULACIÓN DESDE DISCO
    # ──────────────────────────────────────────────────────────────
    def _load_sim_from_disk(self, run_dir, cfg_loaded: dict):
        """Carga trayectorias, tiempos de escape y estadísticas MC de una sim guardada."""
        import json, csv, pathlib
        run_dir = pathlib.Path(run_dir)
        try:
            from contenedor import (ContenedorCilindrico, ContenedorEsferico,
                                    ContenedorCaja, ContenedorPlacasParalelas,
                                    ContenedorTokamak)
            from particulas import Particula

            # ── Reconstruir contenedor ───────────────────────────
            geo = cfg_loaded.get("geometria", "cilindro")
            r   = cfg_loaded.get("radio",    0.01)
            h   = cfg_loaded.get("altura",   0.02)
            Lx  = cfg_loaded.get("Lx",       r*2)
            Ly  = cfg_loaded.get("Ly",       r*2)
            Lz  = cfg_loaded.get("Lz",       h)
            sep = cfg_loaded.get("separacion", h)
            Lpl = cfg_loaded.get("L_placas",   r*2)
            Rm  = cfg_loaded.get("R_mayor",    r)
            am  = cfg_loaded.get("a_menor",    h/4)

            if geo == "cilindro":   cont = ContenedorCilindrico(radio=r, altura=h)
            elif geo == "esfera":   cont = ContenedorEsferico(radio=r)
            elif geo == "caja":     cont = ContenedorCaja(Lx=Lx, Ly=Ly, Lz=Lz)
            elif geo == "placas":   cont = ContenedorPlacasParalelas(d=sep, L=Lpl)
            elif geo == "tokamak":  cont = ContenedorTokamak(R=Rm, a=am)
            else:                   cont = ContenedorCilindrico(radio=r, altura=h)

            # ── Cargar trayectorias CSV ──────────────────────────
            traj_dir = run_dir / "trayectorias"
            particulas, colores = [], []
            if traj_dir.exists():
                import glob
                csvs = sorted(glob.glob(str(traj_dir / "trayectoria_p*.csv")))
                for csv_path in csvs:
                    try:
                        dat = np.loadtxt(csv_path, delimiter=",", skiprows=1)
                        if dat.ndim == 1: dat = dat[np.newaxis, :]
                        # Validar que las columnas sean suficientes (paso, x, y, z, vx, vy, vz)
                        if dat.shape[1] < 4:
                            continue   # CSV malformado — saltar
                        pid = int(csv_path.split("_p")[-1].replace(".csv",""))
                        p = Particula(pid, q=1.0, m=1.0,
                                      x0=dat[0, 1:4], v0=np.zeros(3))
                        # Asegurar que cada entrada de historia sea un array (3,) de float64
                        p.historia_x = [np.asarray(dat[i, 1:4], dtype=float) for i in range(len(dat))]
                        p.historia_v = [np.asarray(dat[i, 4:7], dtype=float) for i in range(len(dat))] if dat.shape[1] > 6 else [np.zeros(3)]*len(dat)
                        particulas.append(p)
                        colores.append("cyan")
                    except Exception:
                        pass

            # ── Cargar tiempos de escape ─────────────────────────
            te = {}
            te_path = run_dir / "montecarlo" / "tiempos_escape.csv"
            if te_path.exists():
                with open(te_path, encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        te[int(row["id"])] = float(row["t_escape_s"])

            # ── Cargar energía cinética ──────────────────────────
            e_hist = []
            e_path = run_dir / "montecarlo" / "energia_cin.csv"
            if e_path.exists():
                d = np.loadtxt(str(e_path), delimiter=",", skiprows=1)
                if d.ndim == 1: d = d[np.newaxis, :]
                e_hist = d[:, 1].tolist() if d.shape[1] > 1 else []

            # ── Actualizar estado global ─────────────────────────
            # IMPORTANTE: escaped_ids debe contener p.id (no índices de lista),
            # igual que durante la simulación en vivo.
            with G._lock:
                G.particulas    = particulas
                G.colores       = colores
                G.contenedor    = cont
                G.tiempos_escape = te
                G.E_cin_historia = e_hist
                G.escaped_ids   = set(te.keys())   # claves son p.id
                G.step          = max((len(p.historia_x) for p in particulas), default=0)
                G.fn_E = None
                G.fn_B = None

            self._log(f"  ✓ Simulación '{run_dir.name}' cargada: "
                      f"{len(particulas)} partículas, {len(te)} escapadas.", "ok")

            # ── Actualizar vista ─────────────────────────────────
            self._view_mode.set("particles")
            self._btn_view_toggle.config(text="⚛  PARTÍCULAS", fg=T["cyan"])
            self._refresh_3d()

            # Habilitar slider y reproducción
            max_s = max((len(p.historia_x) for p in particulas), default=1)
            self._time_slider.configure(to=max_s - 1)
            self._slider_var.set(max_s - 1)
            self._time_slider.state(["!disabled"])
            self._slider_lbl.config(text=f"paso {max_s-1}")
            self._set_playback_btns_state("normal")

            # Actualizar estadísticas MC en panel derecho
            try:
                import montecarlo as mc_mod
                dt   = cfg_loaded.get("dt", 1e-9)
                pasos = cfg_loaded.get("pasos", 1000)
                n_tot = cfg_loaded.get("n_total", len(particulas)) or len(particulas)
                stats = mc_mod.calcular_tau(te, n_tot, dt, pasos)
                from main import _tau_bohm_ref
                tau_ref = _tau_bohm_ref(cfg_loaded)
                self._update_mc_table(stats, tau_ref, n_tot, len(te))
                G.stats = stats
                self._update_mc_charts()
            except Exception:
                pass

        except Exception as ex:
            import traceback
            self._log(f"ERROR al cargar sim: {ex}", "err")
            for line in traceback.format_exc().splitlines():
                self._log(line)

    # ──────────────────────────────────────────────────────────────
    #  MC EN TIEMPO REAL durante la simulación
    # ──────────────────────────────────────────────────────────────
    def _update_mc_realtime(self):
        """Calcula y muestra estadísticas MC con los datos actuales en tiempo real."""
        try:
            import montecarlo as mc_mod

            cfg_pasos = int(float(self._vars["pasos"].get()))
            dt        = float(self._vars["dt"].get())
            n_total   = len(G.particulas) or sum(v.get() for v in self._esp_vars.values()) or 20

            te = dict(G.tiempos_escape)  # copia instantánea

            # Actualizar badges de conteo siempre
            n_esc  = len(te)
            n_conf = n_total - n_esc
            pct_esc = 100 * n_esc / max(1, n_total)
            self._vars["stat_total"].set(str(n_total))
            self._vars["stat_escaped"].set(f"{n_esc}  ({pct_esc:.0f} %)")
            self._vars["stat_confined"].set(str(max(0, n_conf)))
            self._vars["stat_hits"].set(str(n_esc))

            # Calcular tau_ref para la tabla
            tau_ref = 1.0
            try:
                from main import _tau_bohm_ref
                cfg_snap = self._get_config()
                cfg_snap["conteos"] = {k: v.get() for k, v in self._esp_vars.items()
                                       if v.get() > 0} or {"electron": 20}
                tau_ref = _tau_bohm_ref(cfg_snap)
            except Exception:
                pass

            # Actualizar tabla MC (solo si hay escapes para los tau)
            if te:
                stats = mc_mod.calcular_tau(te, n_total, dt, cfg_pasos)
                self._update_mc_table(stats, tau_ref, n_total, n_esc)
            else:
                # Sin escapes: actualizar solo conteos en tabla
                self._mc_vars["n_total"].set(str(n_total))
                self._mc_vars["n_escaped"].set(f"0  (0.0 %)")
                self._mc_vars["n_confined"].set(str(n_total))
                self._mc_vars["tau_bohm"].set(f"{tau_ref:.3e} s")

            # Actualizar gráficas siempre (energía cinética no requiere escapes)
            self._update_mc_charts()

        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────
    #  TABLA MONTE CARLO
    # ──────────────────────────────────────────────────────────────
    def _update_mc_table(self, stats: dict, tau_ref: float,
                         n_total: int, n_esc: int):
        def fmt(x):
            if isinstance(x, float):
                return f"{x:.3e} s"
            return str(x)

        n_conf = n_total - n_esc
        pct_esc = 100*n_esc/max(1, n_total)

        self._mc_vars["n_total"].set(str(n_total))
        self._mc_vars["n_escaped"].set(f"{n_esc}  ({pct_esc:.1f} %)")
        self._mc_vars["n_confined"].set(str(n_conf))
        self._mc_vars["tau_medio"].set(fmt(stats.get("tau_medio",  0.0)))
        self._mc_vars["tau_med"].set(  fmt(stats.get("tau_mediana",0.0)))
        self._mc_vars["tau_std"].set(  fmt(stats.get("tau_std",    0.0)))
        self._mc_vars["pct25"].set(    fmt(stats.get("pct25",      0.0)))
        self._mc_vars["pct75"].set(    fmt(stats.get("pct75",      0.0)))
        self._mc_vars["pct95"].set(    fmt(stats.get("pct95",      0.0)))
        self._mc_vars["tau_bohm"].set( fmt(tau_ref))
        ratio = stats.get("tau_medio", 0.0) / tau_ref if tau_ref > 0 else 0
        self._mc_vars["ratio_bohm"].set(f"{ratio:.4f}")

        # Actualizar badges
        self._vars["stat_escaped"].set(f"{n_esc}  ({pct_esc:.0f} %)")
        self._vars["stat_confined"].set(str(n_conf))
        self._vars["stat_hits"].set(str(n_esc))
        self._vars["stat_total"].set(str(n_total))

    # ──────────────────────────────────────────────────────────────
    #  GRÁFICAS MC
    # ──────────────────────────────────────────────────────────────
    def _update_mc_charts(self):
        try:
            import montecarlo as mc
            from visualizacion import ESCALAS

            cfg_pasos = int(float(self._vars["pasos"].get()))
            dt        = float(self._vars["dt"].get())
            escala    = self._vars["escala"].get()
            n_total   = sum(v.get() for v in self._esp_vars.values()) or 20
            _, factor = ESCALAS.get(escala, ("s", 1.0))

            # ── Decaimiento ──────────────────────────────────────
            self.ax_dec.cla()
            self.ax_dec.set_facecolor(T["plot"])
            if G.tiempos_escape:
                t_arr, N_arr = mc.curva_decaimiento(
                    G.tiempos_escape, n_total, dt, cfg_pasos)
                self.ax_dec.plot(t_arr/factor, N_arr,
                                 color=T["green"], linewidth=1.5)
                self.ax_dec.fill_between(t_arr/factor, N_arr,
                                         alpha=0.15, color=T["green"])
                self.ax_dec.set_xlabel(f"t [{escala}]",
                                       fontsize=6, color=T["mid"])
                self.ax_dec.set_ylabel("N confinadas",
                                       fontsize=6, color=T["mid"])
            self.ax_dec.set_title("Partículas confinadas",
                                  color=T["green"], fontsize=7, pad=2)

            # ── Energía cinética ─────────────────────────────────
            self.ax_egy.cla()
            self.ax_egy.set_facecolor(T["plot"])
            if G.E_cin_historia:
                steps = np.arange(len(G.E_cin_historia))
                self.ax_egy.plot(steps, G.E_cin_historia,
                                 color=T["cyan"], linewidth=1.2)
                self.ax_egy.fill_between(steps, G.E_cin_historia,
                                         alpha=0.15, color=T["cyan"])
                self.ax_egy.set_xlabel("Paso",
                                       fontsize=6, color=T["mid"])
                self.ax_egy.set_ylabel("E_cin [J]",
                                       fontsize=6, color=T["mid"])
            self.ax_egy.set_title("Energía cinética media",
                                  color=T["cyan"], fontsize=7, pad=2)

            for ax in (self.ax_dec, self.ax_egy):
                ax.tick_params(colors=T["lo"], labelsize=5)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                for sp in ("bottom", "left"):
                    ax.spines[sp].set_color(T["grid"])

            self.fig_mc.subplots_adjust(
                left=0.16, right=0.97, top=0.94, bottom=0.09, hspace=0.42,
            )
            self.canvas_mc.draw_idle()

        except Exception as e:
            self._log(f"  [GUI] Error en gráficas: {e}", "dim")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    root = tk.Tk()
    app  = SimuladorGUI(root)

    def _on_close():
        G.stop_req.set()
        G.paused.clear()
        sys.stdout = sys.__stdout__
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
