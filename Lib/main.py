"""
main.py — Punto de entrada del simulador de plasma
===================================================

Cada corrida se guarda en su propia carpeta:

    data/simulaciones/<run_id>/
        config.json           - parámetros completos (reproducible)
        trayectorias/         - CSV por partícula
        montecarlo/           - stats, decaimiento, gráficas MC
        figuras/              - copias de figuras principales
        mapas_calor/          - si se pidió

    data/simulaciones/index.csv   ← índice de todas las corridas

Uso
---
    cd Lib
    python main.py              # menú interactivo
    python main.py --list       # listar corridas anteriores
    python main.py --list 30    # últimas 30
"""

import argparse
import os
import shutil
import sys

import numpy as np

# Backend gráfico
import matplotlib
for _b in ("TkAgg", "Qt5Agg", "Agg"):
    try:
        matplotlib.use(_b)
        break
    except Exception:
        continue

from gestor_corridas import GestorCorrida, BASE_CORRIDAS
from ui_consola import recoger_config

from Aplicaciones import (
    ESPECIES,
    construir_particulas,
    guardar_cache,
    cargar_cache,
    _nombre_cache,
    CACHE_DIR,
    pedir_campo_E,
    pedir_campo_B,
)
from contenedor import (
    ContenedorCilindrico,
    ContenedorEsferico,
    ContenedorCaja,
    ContenedorPlacasParalelas,
    ContenedorTokamak,
)
import campos as campos_mod
from motor import motor_simulacion, _configurar_rejilla
from motor_lite import motor_lite, campo_B_solenoide_vec, campo_E_cero_vec
from tools import guardar_logs_trayectorias
from visualizacion import lanzar_visualizacion
import montecarlo as mc


def _crear_contenedor(config: dict):
    geo = config["geometria"]
    r, h = config["radio"], config["altura"]
    if geo == "cilindro":
        return ContenedorCilindrico(radio=r, altura=h)
    if geo == "esfera":
        return ContenedorEsferico(radio=r)
    if geo == "caja":
        return ContenedorCaja(Lx=r * 2, Ly=r * 2, Lz=h)
    if geo == "placas":
        return ContenedorPlacasParalelas(d=h, L=r * 2)
    if geo == "tokamak":
        return ContenedorTokamak(R=r, a=h / 4)
    raise ValueError(f"Geometría desconocida: {geo}")


def _tau_bohm_ref(config: dict) -> float:
    esp = list(config["conteos"].keys())[0]
    return mc.tau_bohm(
        radio=config["radio"],
        B0=config["B0"],
        T_plasma=config["T_plasma"],
        q=abs(ESPECIES[esp]["q"]),
    )


def ejecutar_pic(config: dict, run_dir: os.PathLike, fn_E=None, fn_B=None):
    """Motor PIC con Poisson."""
    run_dir = os.path.abspath(run_dir)
    contenedor = _crear_contenedor(config)
    dt, pasos = config["dt"], config["pasos"]
    B0, E0 = config["B0"], tuple(config["E0"])
    radio = config["radio"]

    particulas, motores, colores = construir_particulas(
        config["conteos"],
        contenedor,
        dt,
        T_plasma=config["T_plasma"],
        nu=config["nu_colision"],
    )
    n = config["n_total"]

    E0_arr = np.array(E0)
    # Usar funciones de campo seleccionadas por el usuario si se pasaron,
    # sino usar los valores por defecto del config (compatibilidad total)
    if fn_E is None:
        fn_E = lambda pos: campos_mod.campo_electrico_constante(pos, E0=E0_arr)
    if fn_B is None:
        if config["geometria"] == "tokamak":
            fn_B = lambda pos: campos_mod.campo_magnetico_tokamak(
                pos, B0=B0, R=radio, Bpol=0.1 * B0,
            )
        else:
            fn_B = lambda pos: campos_mod.campo_magnetico_solenoide(
                pos, B0=B0, radio=radio,
            )

    nombre_cache = _nombre_cache(config["geometria"], radio, config["altura"], B0, E0)
    os.makedirs(CACHE_DIR, exist_ok=True)
    print("\n  [PIC] Preparando rejilla Poisson...")
    rejilla = _configurar_rejilla(contenedor, (30, 30, 30), fn_E, fn_B)
    if not cargar_cache(rejilla, nombre_cache):
        guardar_cache(rejilla, nombre_cache)

    print(f"\n  [PIC] Corriendo {pasos} pasos, N={n} ...")
    resultado = motor_simulacion(
        pasos=pasos,
        particulas=particulas,
        motores_colision=motores,
        n=n,
        B0=B0,
        E0=E0,
        dt=dt,
        contenedor=contenedor,
        resolucion_grilla=(30, 30, 30),
        registrar_energia=True,
        fn_E_ext=fn_E,
        fn_B_ext=fn_B,
    )
    tiempos_escape, E_cin_historia = resultado
    return {
        "particulas": particulas,
        "motores": motores,
        "colores": colores,
        "contenedor": contenedor,
        "tiempos_escape": tiempos_escape,
        "E_cin_historia": E_cin_historia,
        "fn_E": fn_E,
        "fn_B": fn_B,
    }


def ejecutar_lite(config: dict, run_dir: os.PathLike, fn_E=None, fn_B=None):
    """Motor vectorizado sin Poisson."""
    contenedor = _crear_contenedor(config)
    dt, pasos = config["dt"], config["pasos"]
    B0, radio = config["B0"], config["radio"]

    particulas, motores, colores = construir_particulas(
        config["conteos"],
        contenedor,
        dt,
        T_plasma=config["T_plasma"],
        nu=config["nu_colision"],
    )

    if fn_B is None:
        if config["geometria"] == "tokamak":
            fn_B = lambda X, b=B0, r=radio: np.array([
                campos_mod.campo_magnetico_tokamak(x, B0=b, R=r, Bpol=0.1 * b) for x in X
            ])
        else:
            fn_B = lambda X, b=B0, r=radio: campo_B_solenoide_vec(X, B0=b, radio=r)
    else:
        # fn_B viene de pedir_campo_B (pos→B), envolver para motor_lite que espera (X→[B])
        _fn_B_single = fn_B
        fn_B = lambda X: np.array([_fn_B_single(x) for x in X])
    if fn_E is None:
        fn_E_lite = campo_E_cero_vec
    else:
        _fn_E_single = fn_E
        fn_E_lite = lambda X: np.array([_fn_E_single(x) for x in X])
    print(f"\n  [Lite] Corriendo {pasos} pasos, N={len(particulas)} ...")
    tiempos_escape, E_cin_historia = motor_lite(
        pasos=pasos,
        particulas=particulas,
        motores_colision=motores,
        fn_E=fn_E_lite,
        fn_B=fn_B,
        dt=dt,
        contenedor=contenedor,
        registrar_energia=True,
        verbose=True,
    )
    # fn_E_lite y fn_B ya contienen los campos seleccionados por el usuario;
    # se devuelven tal cual para que visualización y guardado los usen correctamente.
    return {
        "particulas": particulas,
        "motores": motores,
        "colores": colores,
        "contenedor": contenedor,
        "tiempos_escape": tiempos_escape,
        "E_cin_historia": E_cin_historia,
        "fn_E": fn_E_lite,
        "fn_B": fn_B,
    }


def _guardar_energia(E_cin_historia: list, carpeta_mc: str):
    if not E_cin_historia:
        return
    ruta = os.path.join(carpeta_mc, "energia_cin.csv")
    datos = np.column_stack((np.arange(len(E_cin_historia)), E_cin_historia))
    np.savetxt(ruta, datos, delimiter=",", header="paso,E_cin_J", comments="")
    print(f"  [MC] Energía cinética → {ruta}")


def _guardar_tiempos_escape(tiempos_escape: dict, carpeta_mc: str):
    """Guarda {id: t_escape} para poder regenerar figuras después."""
    import csv
    os.makedirs(carpeta_mc, exist_ok=True)
    ruta = os.path.join(carpeta_mc, "tiempos_escape.csv")
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "t_escape_s"])
        for pid in sorted(tiempos_escape.keys()):
            w.writerow([pid, tiempos_escape[pid]])
    print(f"  [MC] Tiempos escape → {ruta}")


def guardar_salidas(config: dict, run_dir: os.PathLike, resultado: dict):
    run_dir = os.path.abspath(run_dir)
    exp = config["exportar"]
    dt, pasos = config["dt"], config["pasos"]
    n = config["n_total"]
    te = resultado["tiempos_escape"]
    escala = config["escala"]

    stats = mc.calcular_tau(te, n, dt, pasos)
    t_arr, N_arr = mc.curva_decaimiento(te, n, dt, pasos)
    tau_ref = _tau_bohm_ref(config)

    # Resumen en texto
    resumen_path = os.path.join(run_dir, "resumen.txt")
    with open(resumen_path, "w", encoding="utf-8") as f:
        f.write(f"run_id: {config.get('run_id')}\n")
        f.write(f"motor: {config['motor']}\n")
        f.write(f"tau_medio_s: {stats['tau_medio']}\n")
        f.write(f"frac_escapadas: {stats['fraccion_esc']}\n")
        f.write(f"tau_bohm_ref_s: {tau_ref}\n")
        for i, m in enumerate(resultado["motores"]):
            f.write(f"colisiones_p{i}: {m.resumen()}\n")

    if exp.get("trayectorias", True):
        guardar_logs_trayectorias(
            resultado["particulas"], dt,
            carpeta_salida=os.path.join(run_dir, "trayectorias"),
        )

  # Visualización 3D ANTES de guardar figuras estáticas (evita backend Agg)
    if exp.get("visualizacion_3d", True):
        try:
            print("\n  [3D] Abriendo ventana interactiva (cierra la ventana para continuar)...")
            lanzar_visualizacion(
                resultado["particulas"],
                resultado["colores"],
                resultado["contenedor"],
                dt,
                escala,
                fn_E=resultado["fn_E"],
                fn_B=resultado["fn_B"],
                n_flechas=4,
            )
        except Exception as e:
            print(f"  [AVISO] Visualización 3D: {e}")
            import traceback
            traceback.print_exc()

    if exp.get("montecarlo", True):
        dir_mc = os.path.join(run_dir, "montecarlo")
        mc.guardar_resultados(stats, t_arr, N_arr, carpeta=dir_mc)
        _guardar_tiempos_escape(te, dir_mc)
        _guardar_energia(resultado["E_cin_historia"], dir_mc)
        mc.imprimir_resumen(stats, escala_key=escala, tau_ref=tau_ref)
        dir_fig = os.path.join(run_dir, "figuras")
        try:
            mc.graficar_resultados(
                stats=stats,
                t_arr=t_arr,
                N_arr=N_arr,
                E_cin_historia=resultado["E_cin_historia"],
                particulas=resultado["particulas"],
                contenedor=resultado["contenedor"],
                tiempos_escape=te,
                escala_key=escala,
                tau_ref=tau_ref,
                guardar_dir=dir_fig,
                mostrar=False,
            )
        except Exception as e:
            print(f"  [AVISO] Gráfica MC: {e}")
            import traceback
            traceback.print_exc()

    if exp.get("mapas_calor", False):
        from mapas_calor import generar_mapas_desde_corrida
        generar_mapas_desde_corrida(
            config,
            resultado=resultado,
            carpeta_salida=os.path.join(run_dir, "mapas_calor"),
        )
        src = os.path.join(run_dir, "mapas_calor", "mapas_calor.png")
        dst = os.path.join(run_dir, "figuras", "mapas_calor.png")
        if os.path.exists(src):
            shutil.copy2(src, dst)

    return stats


def correr_simulacion(config: dict = None) -> str:
    if config is None:
        config = recoger_config()

    gestor = GestorCorrida()
    run_dir = gestor.crear_carpeta(config)
    config["run_id"] = run_dir.name

    print(f"\n  Carpeta de esta corrida:\n  {run_dir}\n")

    # Selección de campos (nuestro aporte — sin tocar lo del compañero)
    fn_E, tag_E     = pedir_campo_E()
    fn_B, B0_real, tag_B = pedir_campo_B(radio=config["radio"])
    # Actualizar B0 en config si el usuario eligió un campo con B0 explícito
    if B0_real != 0.0:
        config["B0"] = B0_real

    if config["motor"] == "lite":
        resultado = ejecutar_lite(config, run_dir, fn_E=fn_E, fn_B=fn_B)
    else:
        resultado = ejecutar_pic(config, run_dir, fn_E=fn_E, fn_B=fn_B)

    stats = guardar_salidas(config, run_dir, resultado)

    # Actualizar config.json con resultados
    import json
    config["tau_medio_s"] = stats["tau_medio"]
    config["frac_escapadas"] = stats["fraccion_esc"]
    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    gestor.registrar_en_indice(config, run_dir, stats)

    print("\n" + "=" * 60)
    print("  CORRIDA FINALIZADA")
    run_name = os.path.basename(run_dir)
    print(f"  ID:      {run_name}")
    print(f"  Datos:   {run_dir}")
    print(f"  Índice:  {BASE_CORRIDAS / 'index.csv'}")
    print("=" * 60 + "\n")
    return str(run_dir)


def _cargar_particulas_impacto(run_dir: str, tiempos_escape: dict):
    """Carga posición final de partículas escapadas desde trayectorias/*.csv."""
    import glob
    from particulas import Particula

    if not tiempos_escape:
        return []
    dir_traj = os.path.join(run_dir, "trayectorias")
    particulas = []
    for pid in tiempos_escape:
        ruta = os.path.join(dir_traj, f"trayectoria_p{pid}.csv")
        if not os.path.exists(ruta):
            continue
        d = np.loadtxt(ruta, delimiter=",", skiprows=1)
        x0 = d[-1, 1:4]
        p = Particula(pid, q=1.0, m=1.0, x0=x0, v0=np.zeros(3))
        particulas.append(p)
    return particulas


def regenerar_figuras(run_dir: str):
    """Regenera figuras/montecarlo PNG desde CSV guardados en una corrida."""
    import csv
    import json
    import matplotlib
    matplotlib.use("Agg")

    run_dir = os.path.abspath(run_dir)
    dir_mc = os.path.join(run_dir, "montecarlo")
    dir_fig = os.path.join(run_dir, "figuras")
    os.makedirs(dir_fig, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    dec = np.loadtxt(
        os.path.join(dir_mc, "montecarlo_decaimiento.csv"),
        delimiter=",", skiprows=1,
    )
    t_arr, N_arr = dec[:, 0], dec[:, 1]

    ruta_te = os.path.join(dir_mc, "tiempos_escape.csv")
    if os.path.exists(ruta_te):
        te = {}
        with open(ruta_te, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                te[int(row["id"])] = float(row["t_escape_s"])
    else:
        te = {}
        n0 = int(N_arr[0])
        for i in range(1, len(N_arr)):
            if N_arr[i] < N_arr[i - 1]:
                for _ in range(int(N_arr[i - 1] - N_arr[i])):
                    te[len(te)] = t_arr[i]

    n = int(cfg["n_total"])
    dt, pasos = cfg["dt"], cfg["pasos"]
    stats = mc.calcular_tau(te, n, dt, pasos)

    tau_ref = _tau_bohm_ref(cfg)
    E_path = os.path.join(dir_mc, "energia_cin.csv")
    E_hist = None
    if os.path.exists(E_path):
        E_hist = np.loadtxt(E_path, delimiter=",", skiprows=1)[:, 1].tolist()

    contenedor = _crear_contenedor(cfg)
    particulas = _cargar_particulas_impacto(run_dir, te)
    mc.graficar_resultados(
        stats, t_arr, N_arr,
        E_cin_historia=E_hist,
        particulas=particulas,
        contenedor=contenedor,
        tiempos_escape=te,
        escala_key=cfg.get("escala", "us"),
        tau_ref=tau_ref,
        guardar_dir=dir_fig,
        mostrar=False,
    )
    print(f"  Figuras regeneradas en {dir_fig}")
    return dir_fig


def main():
    parser = argparse.ArgumentParser(description="Simulador de plasma — corridas organizadas")
    parser.add_argument("--list", nargs="?", const=20, type=int,
                        help="Listar últimas N corridas (default 20)")
    parser.add_argument("--figuras", metavar="RUN_DIR",
                        help="Regenerar figuras desde una carpeta de corrida")
    args = parser.parse_args()

    if args.figuras:
        regenerar_figuras(args.figuras)
        return

    if args.list is not None:
        GestorCorrida.listar_corridas(args.list)
        return

    correr_simulacion()


if __name__ == "__main__":
    main()
