"""
Aplicaciones.py — Main del simulador PIC 3D
============================================
- Selección de especies (electrón, protón, helio, deuterio...)
- Selección de geometría del contenedor
- Caché de campos externos en disco (.npy) → no se recalculan
- Visualización 3D interactiva (sin RecursionError)
  · Slider de progreso
  · Play / Pausa  (Espacio)
  · Frame a frame  (← →)
  · Velocidad      (+ / -)
"""

import os
import numpy as np

import matplotlib
for _b in ('Qt5Agg', 'TkAgg', 'Agg'):
    try:
        matplotlib.use(_b); break
    except Exception:
        continue

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D

from motor import motor_simulacion, _configurar_rejilla
from tools import guardar_logs_trayectorias
from contenedor import (
    ContenedorCilindrico, ContenedorEsferico,
    ContenedorCaja, ContenedorPlacasParalelas, ContenedorTokamak
)
import campos as campos_mod
from visualizacion import lanzar_visualizacion
import montecarlo as mc


# ══════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════

ESPECIES = {
    "electron":  {"m": 9.109e-31,  "q": -1.602e-19, "color": "cyan",    "label": "e⁻"},
    "proton":    {"m": 1.673e-27,  "q":  1.602e-19,  "color": "red",     "label": "p⁺"},
    "hidrogeno": {"m": 1.674e-27,  "q":  1.602e-19,  "color": "orange",  "label": "H⁺"},
    "helio":     {"m": 6.646e-27,  "q":  3.204e-19,  "color": "yellow",  "label": "He²⁺"},
    "helio3":    {"m": 5.008e-27,  "q":  3.204e-19,  "color": "green",   "label": "He3²⁺"},
    "deuterio":  {"m": 3.344e-27,  "q":  1.602e-19,  "color": "magenta", "label": "D⁺"},
}

ESCALAS = {
    "s":  ("segundos",      1.0),
    "ms": ("milisegundos",  1e-3),
    "us": ("microsegundos", 1e-6),
    "ns": ("nanosegundos",  1e-9),
}

GEOMETRIAS = {
    "1": ("cilindro", "Cilindro"),
    "2": ("esfera",   "Esfera"),
    "3": ("caja",     "Caja cúbica"),
    "4": ("placas",   "Placas paralelas"),
    "5": ("tokamak",  "Tokamak"),
}

CACHE_DIR = "cache"


# ══════════════════════════════════════════════════════════════
#  CACHÉ DE CAMPOS EXTERNOS
# ══════════════════════════════════════════════════════════════

def _nombre_cache(geo, radio, altura, B0, E0):
    E0s = "_".join(f"{v:.2e}" for v in E0)
    return f"{geo}_r{radio:.3f}_h{altura:.3f}_B{B0:.3f}_E{E0s}"

def guardar_cache(rejilla, nombre):
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(f"{CACHE_DIR}/{nombre}_E.npy",    rejilla.E_ext_grilla)
    np.save(f"{CACHE_DIR}/{nombre}_B.npy",    rejilla.B_ext_grilla)
    np.save(f"{CACHE_DIR}/{nombre}_mask.npy", rejilla.mask_libre)
    print(f"  [Caché] Guardado → {CACHE_DIR}/{nombre}_*.npy")

def cargar_cache(rejilla, nombre):
    rE = f"{CACHE_DIR}/{nombre}_E.npy"
    rB = f"{CACHE_DIR}/{nombre}_B.npy"
    rM = f"{CACHE_DIR}/{nombre}_mask.npy"
    if os.path.exists(rE) and os.path.exists(rB):
        rejilla.E_ext_grilla = np.load(rE)
        rejilla.B_ext_grilla = np.load(rB)
        if os.path.exists(rM):
            rejilla.mask_libre = np.load(rM)
            rejilla.phi[~rejilla.mask_libre] = 0.0
        print(f"  [Caché] Cargado ← {CACHE_DIR}/{nombre}_*.npy")
        return True
    return False


# ══════════════════════════════════════════════════════════════
#  DIÁLOGO EN CONSOLA
# ══════════════════════════════════════════════════════════════

def _float(msg, default=None):
    while True:
        try:
            raw = input(msg).strip()
            if raw == "" and default is not None:
                return default
            return float(raw)
        except ValueError:
            print("  → Número inválido")

def _int(msg, minval=0):
    while True:
        try:
            v = int(input(msg))
            if v >= minval: return v
            raise ValueError
        except ValueError:
            print(f"  → Entero ≥ {minval}")

def pedir_parametros():
    print("\n╔══════════════════════════════════════╗")
    print("║   PARÁMETROS DE SIMULACIÓN           ║")
    print("╚══════════════════════════════════════╝")
    dt    = _float("  dt [s] (ej: 1e-10): ")
    pasos = _int("  Pasos  (ej: 1000): ", minval=1)
    return dt, pasos

def pedir_campos():
    print("\n╔══════════════════════════════════════╗")
    print("║   CAMPOS EXTERNOS                    ║")
    print("╚══════════════════════════════════════╝")
    B0 = _float("  B0 [T]  (ej: 1.0): ", default=1.0)
    print("  (B0>0 necesario para giro; E0≠0 solo si quieres deriva eléctrica)")
    print("  E0 = Ex Ey Ez [V/m]:")
    while True:
        try:
            E0 = tuple(float(v) for v in input("  Ex Ey Ez (ej: 0 0 0): ").split())
            if len(E0) == 3: break
        except ValueError:
            pass
        print("  → Tres números separados por espacios")
    return B0, E0

def pedir_geometria():
    print("\n╔══════════════════════════════════════╗")
    print("║   GEOMETRÍA DEL CONTENEDOR           ║")
    print("╚══════════════════════════════════════╝")
    for k, (_, n) in GEOMETRIAS.items():
        print(f"  [{k}] {n}")
    while True:
        op = input("Seleccione [1-5]: ").strip()
        if op in GEOMETRIAS: return GEOMETRIAS[op][0]
        print("  → Opción inválida")

def pedir_escala():
    print("\n╔══════════════════════════════════════╗")
    print("║   ESCALA DE TIEMPO (visualización)   ║")
    print("╚══════════════════════════════════════╝")
    for k, (n, _) in ESCALAS.items():
        print(f"  [{k}] {n}")
    while True:
        op = input("Escala [s/ms/us/ns]: ").strip().lower()
        if op in ESCALAS: return op
        print("  → Opción inválida")

def pedir_especies():
    print("\n╔══════════════════════════════════════╗")
    print("║   CONFIGURACIÓN DE ESPECIES          ║")
    print("╚══════════════════════════════════════╝")
    for k, v in ESPECIES.items():
        print(f"  {k:10s}  m={v['m']:.3e} kg  q={v['q']:+.3e} C  [{v['label']}]")
    conteos = {}; total = 0
    print("\nCantidad de cada especie (0 = omitir):")
    for nombre in ESPECIES:
        n = _int(f"  {nombre}: ")
        if n > 0:
            conteos[nombre] = n; total += n
    if total == 0:
        print("  Al menos una partícula."); return pedir_especies()
    print(f"\n  Total: {total} partículas")
    return conteos, total


# ══════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DE PARTÍCULAS
# ══════════════════════════════════════════════════════════════

def construir_particulas(conteos, contenedor, dt, T_plasma=1e4, nu=500.0):
    from particulas import Particula
    from colisiones import ColisionEstocastica, velocidad_inicial_mb

    rng = np.random.default_rng(42)
    particulas, motores, colores = [], [], []

    for idx, (nombre, n_esp) in enumerate(conteos.items()):
        esp = ESPECIES[nombre]
        for j in range(n_esp):
            gid = sum(list(conteos.values())[:idx]) + j
            x0  = contenedor.posicion_aleatoria(rng)
            v0  = velocidad_inicial_mb(esp["m"], T_plasma, rng=rng)
            p   = Particula(gid, q=esp["q"], m=esp["m"], x0=x0, v0=v0)
            particulas.append(p)
            motores.append(ColisionEstocastica(
                nu=nu, m=esp["m"], T=T_plasma, dt=dt, seed=gid))
            colores.append(esp["color"])

    return particulas, motores, colores


# ══════════════════════════════════════════════════════════════
#  VISUALIZACIÓN 3D  (sin RecursionError)
# ══════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n  Tip: para corridas organizadas en data/simulaciones/ usa:")
    print("       python main.py\n")
    print("╔══════════════════════════════════════╗")
    print("║   SIMULADOR DE PLASMA — PIC 3D       ║")
    print("╚══════════════════════════════════════╝")

    dt, pasos  = pedir_parametros()
    B0, E0     = pedir_campos()
    geo        = pedir_geometria()
    escala     = pedir_escala()

    print("\n  Dimensiones del contenedor:")
    radio  = _float("  Radio  (m) [ej: 0.5]: ", default=0.5)
    altura = _float("  Altura (m) [ej: 1.0]: ", default=1.0)

    contenedor_map = {
        "cilindro": lambda: ContenedorCilindrico(radio=radio, altura=altura),
        "esfera":   lambda: ContenedorEsferico(radio=radio),
        "caja":     lambda: ContenedorCaja(Lx=radio*2, Ly=radio*2, Lz=altura),
        "placas":   lambda: ContenedorPlacasParalelas(d=altura, L=radio*2),
        "tokamak":  lambda: ContenedorTokamak(R=radio, a=altura/4),
    }
    contenedor = contenedor_map[geo]()

    conteos, n = pedir_especies()
    particulas, motores_colision, colores = construir_particulas(
        conteos, contenedor, dt
    )

    # Funciones de campo
    E0_arr = np.array(E0)
    fn_E   = lambda pos: campos_mod.campo_electrico_constante(pos, E0=E0_arr)
    fn_B   = lambda pos: campos_mod.campo_magnetico_solenoide(
                             pos, B0=B0, radio=radio)

    # Rejilla con caché de campos externos
    nombre_cache = _nombre_cache(geo, radio, altura, B0, E0)
    print(f"\n  [Config] Preparando rejilla...")
    rejilla = _configurar_rejilla(contenedor, (30, 30, 30), fn_E, fn_B)

    if not cargar_cache(rejilla, nombre_cache):
        guardar_cache(rejilla, nombre_cache)

    # ── Simulación ──────────────────────────────────────
    print(f"\n▶ Corriendo: {pasos} pasos, dt={dt:.2e}s, N={n} ...")
    resultado = motor_simulacion(
        pasos             = pasos,
        particulas        = particulas,
        motores_colision  = motores_colision,
        n                 = n,
        B0                = B0,
        E0                = E0,
        dt                = dt,
        contenedor        = contenedor,
        resolucion_grilla = (30, 30, 30),
        registrar_energia = True,   # ← activa registro de E_cin
    )

    # motor devuelve (tiempos_escape, E_cin_historia) cuando registrar_energia=True
    tiempos_escape, E_cin_historia = resultado

    guardar_logs_trayectorias(particulas, dt)

    # ── Resumen de colisiones ────────────────────────────
    print("\n── Resumen de colisiones estocásticas ──")
    for i, m in enumerate(motores_colision):
        print(f"  P{i}: {m.resumen()}")

    # ── Análisis Monte Carlo ─────────────────────────────
    stats      = mc.calcular_tau(tiempos_escape, n, dt, pasos)
    t_arr, N_arr = mc.curva_decaimiento(tiempos_escape, n, dt, pasos)

    # Tau de Bohm teórico como referencia
    esp_ref  = list(conteos.keys())[0]
    tau_ref  = mc.tau_bohm(
        radio    = radio,
        B0       = B0,
        T_plasma = 1e4,           # temperatura por defecto
        q        = abs(ESPECIES[esp_ref]["q"]),
    )

    mc.imprimir_resumen(stats, escala_key=escala, tau_ref=tau_ref)
    mc.guardar_resultados(stats, t_arr, N_arr, carpeta="data")

    # ── Visualización de plasma ───────────────────────────
    lanzar_visualizacion(particulas, colores, contenedor, dt, escala,
                         fn_E=fn_E, fn_B=fn_B, n_flechas=4)

    # ── Figura Monte Carlo ──────────────────────────────
    mc.graficar_resultados(
        stats         = stats,
        t_arr         = t_arr,
        N_arr         = N_arr,
        E_cin_historia= E_cin_historia,
        particulas    = particulas,
        contenedor    = contenedor,
        tiempos_escape= tiempos_escape,
        escala_key    = escala,
        tau_ref       = tau_ref,
        guardar_dir   = "data",
    )
