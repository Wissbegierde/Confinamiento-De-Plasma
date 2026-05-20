"""
motor.py
========
Motor PIC — siempre usa Poisson (LU precalculada).
- Matriz A construida vectorizada y factorizada UNA sola vez
- Campos externos precalculados en grilla UNA sola vez
- Loop de partículas paralelizado con joblib (threads)
- Progreso en consola
"""

import numpy as np
from joblib import Parallel, delayed

import campos as campos_mod
from integradores import boris_step
from interacciones import ResolvedorPoisson
from contenedor import (
    ContenedorCilindrico, ContenedorEsferico,
    ContenedorCaja, ContenedorPlacasParalelas, ContenedorTokamak
)


# ══════════════════════════════════════════════════════════════
#  CONFIGURACIÓN DE REJILLA
# ══════════════════════════════════════════════════════════════

def _configurar_rejilla(contenedor, resolucion, fn_E, fn_B):
    """
    Detecta el tipo de contenedor, define la frontera,
    precalcula A (LU) y los campos externos.
    Todo ocurre UNA SOLA VEZ antes del loop.
    """
    if isinstance(contenedor, ContenedorCilindrico):
        L = 2 * contenedor.radio
        H = contenedor.altura
        dims   = [L, L, H]
        offset = np.array([L/2, L/2, H/2])
        def frontera(X, Y, Z):
            return (
                ((X - L/2)**2 + (Y - L/2)**2 > contenedor.radio**2) |
                (Z < 0) | (Z > H)
            )

    elif isinstance(contenedor, ContenedorEsferico):
        L = 2 * contenedor.radio
        dims   = [L, L, L]
        offset = np.array([L/2, L/2, L/2])
        def frontera(X, Y, Z):
            return (X-L/2)**2 + (Y-L/2)**2 + (Z-L/2)**2 > contenedor.radio**2

    elif isinstance(contenedor, ContenedorCaja):
        Lx, Ly, Lz = contenedor.lim * 2
        dims   = [Lx, Ly, Lz]
        offset = contenedor.lim.copy()
        def frontera(X, Y, Z):
            return np.zeros_like(X, dtype=bool)

    elif isinstance(contenedor, ContenedorPlacasParalelas):
        L = contenedor.L
        d = contenedor.d
        dims   = [L, L, d]
        offset = np.array([L/2, L/2, d/2])
        def frontera(X, Y, Z):
            return (Z < 0) | (Z > d)

    elif isinstance(contenedor, ContenedorTokamak):
        R, a = contenedor.R, contenedor.a
        lim  = R + a
        dims   = [2*lim, 2*lim, 2*a]
        offset = np.array([lim, lim, a])
        def frontera(X, Y, Z):
            r_xy = np.sqrt((X - lim)**2 + (Y - lim)**2)
            return (r_xy - R)**2 + (Z - a)**2 > a**2

    else:
        raise ValueError(f"Contenedor {type(contenedor)} no soportado.")

    rejilla = ResolvedorPoisson(dimensiones=dims, resolucion=resolucion)
    rejilla.offset = offset

    rejilla.definir_frontera_vectorizada(frontera)
    rejilla.precalcular_matriz()                          # LU una vez
    rejilla.precalcular_campos_externos(fn_E, fn_B, offset)  # campos una vez

    return rejilla


# ══════════════════════════════════════════════════════════════
#  HELPERS DE COORDENADAS
# ══════════════════════════════════════════════════════════════

def _pg(pos, rejilla):
    """Coordenadas físicas → coordenadas de grilla."""
    return pos + rejilla.offset


def _depositar(rejilla, particulas, escaped_idx=None):
    """Deposita carga aplicando el offset. Omite partículas ya escapadas."""
    class _P:
        def __init__(self, p, off):
            self.x = p.x + off
            self.q = p.q
    activas = [
        p for i, p in enumerate(particulas)
        if escaped_idx is None or i not in escaped_idx
    ]
    rejilla.depositar_carga_cic([_P(p, rejilla.offset) for p in activas])


# ══════════════════════════════════════════════════════════════
#  PASO DE UNA PARTÍCULA (se ejecuta en paralelo)
# ══════════════════════════════════════════════════════════════

def _paso_particula(px, pv, pq, pm,
                    E_ext, B_ext, E_int,
                    motor_col, contenedor, px_old, dt, is_escaped=False):
    # Partícula ya escapada: no se mueve ni interactúa
    if is_escaped:
        return px, pv, False
    E_total             = E_ext + E_int
    x_new, v_new        = boris_step(px, pv, E_total, B_ext, pq, pm, dt)
    v_new, _            = motor_col.aplicar(v_new)
    x_new, v_new, choco = contenedor.manejar_colision(x_new, v_new, px_old)
    return x_new, v_new, choco


# ══════════════════════════════════════════════════════════════
#  MOTOR PRINCIPAL
# ══════════════════════════════════════════════════════════════

def motor_simulacion(
    pasos             = 100,
    particulas        = [],
    motores_colision  = [],
    n                 = 2,
    B0                = 1.0,
    E0                = (0.0, 0.0, 0.0),
    dt                = 1e-10,
    contenedor        = None,
    resolucion_grilla = (30, 30, 30),
    intervalo_log     = 100,
    n_jobs            = -1,
    registrar_energia = False,
    fn_E_ext          = None,   # si se pasa, reemplaza campo_electrico_constante
    fn_B_ext          = None,   # si se pasa, reemplaza campo_magnetico_solenoide
    paused_event      = None,   # threading.Event: set = pausado
    stop_event        = None,   # threading.Event: set = detener
    estado_global     = None,   # objeto _SimState de la GUI (opcional)
):
    """
    Ejecuta el loop PIC principal.

    Parámetros
    ----------
    pasos             : número de pasos de tiempo
    particulas        : lista de objetos Particula
    motores_colision  : lista de ColisionEstocastica (uno por partícula)
    n                 : número total de partículas
    B0                : intensidad del campo magnético [T]
    E0                : campo eléctrico externo (Ex, Ey, Ez) [V/m]
    dt                : paso de tiempo [s]
    contenedor        : objeto ContenedorXxx
    resolucion_grilla : tupla (nx, ny, nz) de la rejilla Poisson
    intervalo_log     : cada cuántos pasos se imprime el progreso
    n_jobs            : hilos para joblib (-1 = todos)
    registrar_energia : si True, devuelve la evolución de E_cin total

    Retorna
    -------
    tiempos_escape : dict {id_particula: t_escape [s]}
                     solo contiene las partículas que chocaron con la pared
    E_cin_historia : list[float] (solo si registrar_energia=True)
                     energía cinética total en cada paso [J]
    """
    if contenedor is None:
        raise ValueError("Debes pasar un contenedor al motor.")

    E0_arr    = np.array(E0, dtype=float)
    radio_ref = getattr(contenedor, 'radio', 0.5)

    # Usar funciones externas si se pasan, sino usar los valores B0/E0
    fn_E = fn_E_ext if fn_E_ext is not None else (
        lambda pos: campos_mod.campo_electrico_constante(pos, E0=E0_arr)
    )
    fn_B = fn_B_ext if fn_B_ext is not None else (
        lambda pos: campos_mod.campo_magnetico_solenoide(pos, B0=B0, radio=radio_ref)
    )

    # ── Estructuras de seguimiento ────────────────────────────
    tiempos_escape  = {}          # {id_particula: t_escape [s]}
    confinadas      = set(range(n))  # índices de partículas aún dentro
    escaped_indices = set()          # índices (no IDs) de partículas escapadas
    E_cin_historia  = []          # energía cinética total por paso

    # ── Precálculo ────────────────────────────────────────────
    print("  [Motor] Configurando rejilla...", flush=True)
    rejilla = _configurar_rejilla(contenedor, resolucion_grilla, fn_E, fn_B)
    print(f"  [Motor] Listo. Iniciando {pasos} pasos × {n} partículas.\n",
          flush=True)

    # ── Loop principal ─────────────────────────────────────────
    for paso in range(pasos):

        # ── Control de pausa y stop ───────────────────────────
        if stop_event is not None and stop_event.is_set():
            print("  [Motor] Detenido por solicitud del usuario.", flush=True)
            break
        if paused_event is not None:
            while paused_event.is_set():
                if stop_event is not None and stop_event.is_set():
                    break
                import time as _time
                _time.sleep(0.05)
        if stop_event is not None and stop_event.is_set():
            print("  [Motor] Detenido por solicitud del usuario.", flush=True)
            break

        if paso % intervalo_log == 0:
            print(f"  Paso {paso:>6}/{pasos}  ({100*paso/pasos:5.1f}%)",
                  flush=True)

        # 1. Campo colectivo (Poisson — solve rápido con LU)
        #    Solo partículas activas depositan carga
        _depositar(rejilla, particulas, escaped_indices)
        rejilla.resolver_con_matriz()

        # 2. Recoger campos para cada partícula
        #    Las ya escapadas reciben ceros — no se evalúan en la rejilla
        #    (FIX: ineficiencia — antes se evaluaban igualmente aunque
        #     _paso_particula luego las ignoraba con is_escaped=True)
        _zeros3 = np.zeros(3)
        E_exts, B_exts, E_ints = [], [], []
        for i in range(n):
            if i in escaped_indices:
                E_exts.append(_zeros3)
                B_exts.append(_zeros3)
                E_ints.append(_zeros3)
                continue
            pg = _pg(particulas[i].x, rejilla)
            E_exts.append(rejilla.obtener_E_externo(pg))
            B_exts.append(rejilla.obtener_B_externo(pg))
            E_ints.append(rejilla.obtener_E_colectivo(pg))

        # 3. Mover partículas en paralelo
        #    Las ya escapadas hacen cortocircuito (is_escaped=True)
        resultados = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_paso_particula)(
                particulas[i].x, particulas[i].v,
                particulas[i].q, particulas[i].m,
                E_exts[i], B_exts[i], E_ints[i],
                motores_colision[i], contenedor,
                particulas[i].x, dt,
                is_escaped=(i in escaped_indices)
            )
            for i in range(n)
        )

        # 4. Actualizar todos juntos + registrar escapes
        t_actual = (paso + 1) * dt
        E_cin_paso = 0.0
        for i, (xn, vn, choco) in enumerate(resultados):
            particulas[i].actualizar_estado(xn, vn)
            # Registrar primera vez que la partícula choca
            if choco and particulas[i].id not in tiempos_escape:
                tiempos_escape[particulas[i].id] = t_actual
                confinadas.discard(i)
                escaped_indices.add(i) 
                particulas[i].v = np.zeros(3)

            
            if registrar_energia and i not in escaped_indices:
                E_cin_paso += 0.5 * particulas[i].m * np.dot(vn, vn)

        if registrar_energia:
            E_cin_historia.append(E_cin_paso)

        # ── Publicar estado en tiempo real hacia la GUI ───────
        if estado_global is not None:
            try:
                with estado_global._lock:
                    estado_global.tiempos_escape = dict(tiempos_escape)
                    estado_global.escaped_ids    = set(tiempos_escape.keys())
                    if registrar_energia:
                        estado_global.E_cin_historia = list(E_cin_historia)
            except Exception:
                pass

    n_escaparon  = len(tiempos_escape)
    n_confinadas = n - n_escaparon
    print(f"  Paso {pasos}/{pasos}  (100.0%) — Completo.", flush=True)
    print(f"  Partículas escapadas: {n_escaparon}/{n}  "
          f"| Confinadas: {n_confinadas}/{n}\n", flush=True)

    if registrar_energia:
        return tiempos_escape, E_cin_historia
    return tiempos_escape
