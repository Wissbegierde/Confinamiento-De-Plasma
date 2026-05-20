"""
test_montecarlo.py
==================
Pruebas unitarias e integración para:
  - motor.py  → registro de tiempos de escape
  - montecarlo.py → todas las funciones de análisis

Las pruebas usan datos sintéticos o una simulación mínima (sin rejilla Poisson)
para que corran en segundos.
"""

import sys
import os
import numpy as np

# ── Asegurar que los módulos del proyecto se encuentren ──────
sys.path.insert(0, os.path.dirname(__file__))

import montecarlo as mc
from particulas import Particula
from colisiones import ColisionEstocastica, velocidad_inicial_mb
from contenedor import ContenedorCilindrico, ContenedorEsferico
from integradores import boris_step
from motor_lite import motor_lite, campo_B_solenoide_vec, campo_E_cero_vec
from test_helpers import (
    NEGRIT,
    RESET,
    afirmar,
    fallo,
    imprimir_resumen_final,
    ok,
    reset_resultados,
    setup_utf8_stdout_win,
)
# ══════════════════════════════════════════════════════════════
#  BLOQUE 1 — calcular_tau()
# ══════════════════════════════════════════════════════════════

def test_calcular_tau():
    print(f"\n{NEGRIT}[1] calcular_tau(){RESET}")

    dt    = 1e-9
    pasos = 1000
    n     = 10

    # 5 escapan en distintos tiempos, 5 confinadas
    tiempos_escape = {0: 100e-9, 1: 200e-9, 2: 300e-9, 3: 400e-9, 4: 500e-9}
    stats = mc.calcular_tau(tiempos_escape, n, dt, pasos)

    afirmar(stats["n_total"]      == 10,   "n_total == 10")
    afirmar(stats["n_escaparon"]  == 5,    "n_escaparon == 5")
    afirmar(stats["n_confinadas"] == 5,    "n_confinadas == 5")
    afirmar(abs(stats["fraccion_esc"] - 0.5) < 1e-9,
            "fraccion_esc == 0.5")

    # Las confinadas reciben t_max = 1000 * 1e-9 = 1e-6
    t_max = pasos * dt
    t_arr = stats["t_escape_arr"]
    afirmar(len(t_arr) == 10, "t_escape_arr tiene 10 elementos")
    afirmar(np.all(t_arr[5:] == t_max),
            "confinadas tienen t == t_max")

    # Sin escapes
    stats0 = mc.calcular_tau({}, n, dt, pasos)
    afirmar(stats0["n_escaparon"]  == 0, "sin escapes: n_escaparon == 0")
    afirmar(stats0["n_confinadas"] == n, "sin escapes: n_confinadas == n")
    afirmar(np.all(stats0["t_escape_arr"] == t_max),
            "sin escapes: todos tienen t_max")


# ══════════════════════════════════════════════════════════════
#  BLOQUE 2 — fraccion_confinadas()
# ══════════════════════════════════════════════════════════════

def test_fraccion_confinadas():
    print(f"\n{NEGRIT}[2] fraccion_confinadas(){RESET}")

    afirmar(mc.fraccion_confinadas({}, 10)            == 1.0, "0 escapes → 1.0")
    afirmar(mc.fraccion_confinadas({0:1e-9}, 10)      == 0.9, "1 escape → 0.9")
    afirmar(mc.fraccion_confinadas({i:1e-9 for i in range(10)}, 10) == 0.0,
            "10 escapes → 0.0")


# ══════════════════════════════════════════════════════════════
#  BLOQUE 3 — curva_decaimiento()
# ══════════════════════════════════════════════════════════════

def test_curva_decaimiento():
    print(f"\n{NEGRIT}[3] curva_decaimiento(){RESET}")

    dt    = 1e-9
    pasos = 10
    n     = 4

    # Escapes en pasos 2, 5, 8
    tiempos_escape = {0: 2*dt, 1: 5*dt, 2: 8*dt}
    t_arr, N_arr = mc.curva_decaimiento(tiempos_escape, n, dt, pasos)

    afirmar(len(t_arr) == pasos + 1, "t_arr tiene pasos+1 elementos")
    afirmar(len(N_arr) == pasos + 1, "N_arr tiene pasos+1 elementos")
    afirmar(N_arr[0]  == 4, "N_arr[0] == n")
    afirmar(N_arr[2]  == 3, "N_arr[2] == 3  (escape en paso 2)")
    afirmar(N_arr[5]  == 2, "N_arr[5] == 2  (escape en paso 5)")
    afirmar(N_arr[8]  == 1, "N_arr[8] == 1  (escape en paso 8)")
    afirmar(N_arr[10] == 1, "N_arr[10] == 1 (partícula 3 nunca escapó)")

    # Monotonía: nunca debe crecer
    afirmar(np.all(np.diff(N_arr) <= 0), "N_arr es no creciente")

    # Sin escapes: siempre n
    _, N0 = mc.curva_decaimiento({}, n, dt, pasos)
    afirmar(np.all(N0 == n), "sin escapes: N constante == n")


# ══════════════════════════════════════════════════════════════
#  BLOQUE 4 — tau_bohm() y radio_larmor()
# ══════════════════════════════════════════════════════════════

def test_modelos_teoricos():
    print(f"\n{NEGRIT}[4] tau_bohm() y radio_larmor(){RESET}")

    # Bohm: D = kT/(16qB), τ = R²/D
    kB   = 1.380649e-23
    q    = 1.602e-19
    B0   = 1.0
    T    = 1e4
    R    = 0.5
    D_esperado = kB * T / (16 * q * B0)
    tau_esperado = R**2 / D_esperado
    tau_calc = mc.tau_bohm(R, B0, T, q)
    error_rel = abs(tau_calc - tau_esperado) / tau_esperado
    afirmar(error_rel < 1e-10, f"tau_bohm correcto (err={error_rel:.2e})")

    # Radio de Larmor: r = mv/(|q|B)
    m     = 9.109e-31
    v     = 1e6
    r_esp = m * v / (q * B0)
    r_cal = mc.radio_larmor(m, v, q, B0)
    afirmar(abs(r_cal - r_esp) / r_esp < 1e-10,
            f"radio_larmor correcto (r={r_cal:.4e} m)")

    # B más grande → τ más pequeño (mejor confinamiento con Bohm da τ ∝ B)
    tau1 = mc.tau_bohm(R, 1.0, T, q)
    tau2 = mc.tau_bohm(R, 2.0, T, q)
    afirmar(tau2 > tau1, "τ_Bohm crece con B (τ ∝ B)")


# ══════════════════════════════════════════════════════════════
#  BLOQUE 5 — motor con escape real (sin Poisson, sin joblib)
# ══════════════════════════════════════════════════════════════

def test_motor_escape_simple():
    """
    Simulación mínima sin rejilla Poisson:
    - Partícula lanzada hacia la pared a velocidad constante
    - Verificar que el tiempo de escape registrado es coherente
    """
    print(f"\n{NEGRIT}[5] Escape real en contenedor esférico{RESET}")

    radio  = 0.5
    dt     = 1e-4          # paso de tiempo grande para converger rápido
    pasos  = 200

    cont = ContenedorEsferico(radio=radio)

    # Partícula en el centro con velocidad radial pura: escapará en t ≈ R/v
    v_mag  = radio / (50 * dt)   # escape teórico ~50 pasos
    x0     = np.array([0.0, 0.0, 0.0])
    v0     = np.array([v_mag, 0.0, 0.0])
    p      = Particula(0, q=1e-6, m=1e-9, x0=x0, v0=v0)

    tiempos_escape = {}

    for paso in range(pasos):
        # Boris sin campo (E=B=0) → movimiento rectilíneo uniforme
        E = np.zeros(3); B = np.zeros(3)
        x_new, v_new = boris_step(p.x, p.v, E, B, p.q, p.m, dt)
        x_new, v_new, choco = cont.manejar_colision(x_new, v_new, p.x)
        p.actualizar_estado(x_new, v_new)

        if choco and p.id not in tiempos_escape:
            tiempos_escape[p.id] = (paso + 1) * dt
            break

    t_escape_sim  = tiempos_escape.get(0, None)
    t_escape_teo  = radio / v_mag          # R / v
    afirmar(t_escape_sim is not None, "escape registrado")
    if t_escape_sim:
        error = abs(t_escape_sim - t_escape_teo) / t_escape_teo
        afirmar(error < 0.05,          # menos del 5% de error (un paso de dt)
                f"t_escape coherente  sim={t_escape_sim:.4e}s "
                f"teo={t_escape_teo:.4e}s  err={100*error:.1f}%")


# ══════════════════════════════════════════════════════════════
#  BLOQUE 5b — motor_lite vectorizado (reemplaza _test_motor_lite.py)
# ══════════════════════════════════════════════════════════════

def test_motor_lite_minimo():
    print(f"\n{NEGRIT}[5b] motor_lite() vectorizado (smoke){RESET}")
    N, PASOS, DT = 4, 80, 1e-10
    RADIO, ALTURA = 0.05, 0.10
    M, Q = 1.673e-27, 1.602e-19
    T, NU, B0 = 1e4, 500.0, 0.5
    rng = np.random.default_rng(0)
    cont = ContenedorCilindrico(radio=RADIO, altura=ALTURA)
    particulas, motores = [], []
    for i in range(N):
        x0 = cont.posicion_aleatoria(rng)
        v0 = velocidad_inicial_mb(M, T, rng=rng)
        p = Particula(i, q=Q, m=M, x0=x0, v0=v0)
        motores.append(ColisionEstocastica(nu=NU, m=M, T=T, dt=DT, seed=i))
        particulas.append(p)

    fn_B = lambda X, b=B0, r=RADIO: campo_B_solenoide_vec(X, B0=b, radio=r)
    te, E_hist = motor_lite(
        pasos=PASOS,
        particulas=particulas,
        motores_colision=motores,
        fn_E=campo_E_cero_vec,
        fn_B=fn_B,
        dt=DT,
        contenedor=cont,
        registrar_energia=True,
        verbose=False,
        intervalo_log=10000,
    )
    afirmar(len(E_hist) > 0, "motor_lite registra energía")
    afirmar(len(E_hist) <= PASOS, "historia energía ≤ pasos")
    mc.calcular_tau(te, N, DT, PASOS)  # no debe lanzar
    ok("calcular_tau con salida de motor_lite")
    t_arr, N_arr = mc.curva_decaimiento(te, N, DT, PASOS)
    afirmar(N_arr[0] == N, "N_arr[0] == N")
    afirmar(np.all(np.diff(N_arr) <= 0), "N_arr no creciente")


# ══════════════════════════════════════════════════════════════
#  BLOQUE 6 — guardar_resultados() crea los CSV
# ══════════════════════════════════════════════════════════════

def test_guardar_resultados():
    print(f"\n{NEGRIT}[6] guardar_resultados() → CSV{RESET}")

    dt    = 1e-9
    pasos = 100
    n     = 5
    tiempos_escape = {0: 20e-9, 1: 50e-9, 2: 80e-9}
    stats          = mc.calcular_tau(tiempos_escape, n, dt, pasos)
    t_arr, N_arr   = mc.curva_decaimiento(tiempos_escape, n, dt, pasos)

    carpeta = os.path.join(os.path.dirname(__file__), "..", "data", "test_mc")
    mc.guardar_resultados(stats, t_arr, N_arr, carpeta=carpeta)

    ruta_stats = os.path.join(carpeta, "montecarlo_stats.csv")
    ruta_decay = os.path.join(carpeta, "montecarlo_decaimiento.csv")

    afirmar(os.path.exists(ruta_stats), "montecarlo_stats.csv creado")
    afirmar(os.path.exists(ruta_decay), "montecarlo_decaimiento.csv creado")

    # Verificar contenido del CSV de estadísticas
    with open(ruta_stats) as f:
        lineas = f.readlines()
    afirmar(len(lineas) == 2, "stats CSV tiene cabecera + 1 fila de datos")

    # Verificar contenido del CSV de decaimiento
    datos = np.loadtxt(ruta_decay, delimiter=",", skiprows=1)
    afirmar(datos.shape == (pasos + 1, 2),
            f"decay CSV shape == ({pasos+1}, 2)")
    afirmar(datos[0, 1] == n, "primer N_confinadas == n")


# ══════════════════════════════════════════════════════════════
#  BLOQUE 7 — graficar_resultados() no lanza excepciones
# ══════════════════════════════════════════════════════════════

def test_graficar_resultados():
    print(f"\n{NEGRIT}[7] graficar_resultados() sin errores{RESET}")

    import matplotlib
    matplotlib.use("Agg")   # backend sin ventana → seguro en test

    dt    = 1e-9
    pasos = 200
    n     = 20
    rng   = np.random.default_rng(0)

    # Tiempos de escape aleatorios para 15 de 20 partículas
    tiempos_escape = {i: rng.uniform(10e-9, 180e-9) for i in range(15)}
    E_cin_historia = list(rng.uniform(0.9, 1.1, pasos))   # energía simulada

    stats        = mc.calcular_tau(tiempos_escape, n, dt, pasos)
    t_arr, N_arr = mc.curva_decaimiento(tiempos_escape, n, dt, pasos)

    carpeta_out = os.path.join(os.path.dirname(__file__), "..", "data", "test_mc")
    try:
        mc.graficar_resultados(
            stats          = stats,
            t_arr          = t_arr,
            N_arr          = N_arr,
            E_cin_historia = E_cin_historia,
            escala_key     = "ns",
            guardar_dir    = carpeta_out,
        )
        ok("graficar_resultados() ejecutado sin excepciones")

        ruta_fig = os.path.join(carpeta_out, "analisis_montecarlo.png")
        afirmar(os.path.exists(ruta_fig), "figura PNG guardada en disco")
        size_kb = os.path.getsize(ruta_fig) / 1024
        afirmar(size_kb > 50, f"figura tiene tamaño razonable ({size_kb:.0f} KB)")
    except Exception as e:
        fallo("graficar_resultados()", str(e))


# ══════════════════════════════════════════════════════════════
#  BLOQUE 8 — imprimir_resumen() no lanza excepciones
# ══════════════════════════════════════════════════════════════

def test_imprimir_resumen():
    print(f"\n{NEGRIT}[8] imprimir_resumen(){RESET}")

    dt    = 1e-9
    pasos = 100
    n     = 10
    tiempos_escape = {i: (i+1)*10e-9 for i in range(6)}
    stats = mc.calcular_tau(tiempos_escape, n, dt, pasos)
    tau_ref = mc.tau_bohm(0.5, 1.0, 1e4, 1.602e-19)

    try:
        mc.imprimir_resumen(stats, escala_key="ns", tau_ref=tau_ref)
        ok("imprimir_resumen() sin excepciones")
    except Exception as e:
        fallo("imprimir_resumen()", str(e))


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    setup_utf8_stdout_win()
    reset_resultados()

    print(f"\n{NEGRIT}==========================================={RESET}")
    print(f"{NEGRIT}  TEST: motor_lite + montecarlo + motor simple {RESET}")
    print(f"{NEGRIT}==========================================={RESET}")

    test_calcular_tau()
    test_fraccion_confinadas()
    test_curva_decaimiento()
    test_modelos_teoricos()
    test_motor_escape_simple()
    test_motor_lite_minimo()
    test_guardar_resultados()
    test_graficar_resultados()
    test_imprimir_resumen()

    imprimir_resumen_final()
