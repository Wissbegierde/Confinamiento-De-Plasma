"""
mapas_calor.py
==============
Semana 13: mapas de calor de densidad y flujo en paredes.

Por defecto (desde main.py con motor PIC) usa la **misma corrida** que el
Monte Carlo: trayectorias en historia_x e impactos en tiempos_escape.
No relanza motor_lite en cilindro.

Corrida lite separada solo si se llama ejecutar_semana13() o mapas sin resultado PIC.
"""

import os
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(__file__))

from particulas import Particula
from colisiones import ColisionEstocastica, velocidad_inicial_mb
from contenedor import (
    ContenedorCilindrico,
    ContenedorTokamak,
    ContenedorEsferico,
    ContenedorCaja,
    ContenedorPlacasParalelas,
    clasificar_impacto_pared,
)
from motor_lite import motor_lite, campo_B_solenoide_vec, campo_E_cero_vec
import campos as campos_mod
import montecarlo as mc

# ── Configuración por defecto ─────────────────────────────────
N_PARTICULAS = 80
PASOS = 15_000
DT = 1e-8
M = 1.673e-27
Q = 1.602e-19
T_PLASMA = 1e4
NU_COLISION = 500.0
B0 = 0.05
RADIO = 0.01
ALTURA = 0.02
INTERVALO_MUESTREO = 25
NR, NZ = 40, 30
SEED = 42

GUARDAR_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "mapas_calor"
)


def _construir_particulas(contenedor, rng, seed_base):
    return _construir_particulas_custom(
        contenedor, rng, seed_base, N_PARTICULAS, T_PLASMA, NU_COLISION, DT
    )


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


def _limites_rz(contenedor):
    """Límites del histograma (r_xy, z) según geometría."""
    if isinstance(contenedor, ContenedorCilindrico):
        return contenedor.radio, contenedor.z_min, contenedor.z_max
    if isinstance(contenedor, ContenedorTokamak):
        return contenedor.R + contenedor.a, -contenedor.a, contenedor.a
    if isinstance(contenedor, ContenedorEsferico):
        return contenedor.radio, -contenedor.radio, contenedor.radio
    if isinstance(contenedor, ContenedorPlacasParalelas):
        L = getattr(contenedor, "L", 0.02)
        return L / 2, contenedor.z_min, contenedor.z_max
    if isinstance(contenedor, ContenedorCaja):
        lim = contenedor.lim
        return max(lim[0], lim[1]), -lim[2], lim[2]
    raise TypeError(f"Geometría no soportada para mapas: {type(contenedor)}")


def _extraer_X_activas(snap):
    """De un snapshot (dict con escapo o array legacy) devuelve posiciones vivas."""
    if isinstance(snap, dict):
        X = snap["X"]
        esc = snap["escapo"]
        return X[~esc]
    return snap


def densidad_rz_desde_muestreos(historial, contenedor, nr=NR, nz=NZ):
    """
    Acumula densidad en el plano (r_xy, z) solo de partículas NO escapadas.
    historial: lista de dict {'X', 'escapo'} o arrays (N,3) legacy.
    """
    r_max, z_min, z_max = _limites_rz(contenedor)
    r_edges = np.linspace(0, r_max, nr + 1)
    z_edges = np.linspace(z_min, z_max, nz + 1)
    H = np.zeros((nr, nz), dtype=float)
    n_contribuciones = 0

    for snap in historial:
        X = _extraer_X_activas(snap)
        if len(X) == 0:
            continue
        r = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
        z = X[:, 2]
        valid = np.array([contenedor.esta_dentro(x) for x in X])
        if not valid.any():
            continue
        H += np.histogram2d(
            r[valid], z[valid],
            bins=[r_edges, z_edges],
        )[0]
        n_contribuciones += int(valid.sum())

    if n_contribuciones > 0:
        H /= n_contribuciones
    return H, r_edges, z_edges


def flujo_pared_desde_impactos(impactos, contenedor, nz=NZ, nr_tok=25):
    """
    Panel de flujo de impactos.
    Cilindro: [tipo_pared × z]. Tokamak: histograma 2D (R_xy, z) en la pared.
    """
    if isinstance(contenedor, ContenedorTokamak):
        r_min = max(0.0, contenedor.R - contenedor.a)
        r_max = contenedor.R + contenedor.a
        r_edges = np.linspace(r_min, r_max, nr_tok + 1)
        z_edges = np.linspace(-contenedor.a, contenedor.a, nz + 1)
        Hf = np.zeros((nr_tok, nz), dtype=float)
        for imp in impactos:
            x = np.asarray(imp["x"], dtype=float)
            r_xy = np.linalg.norm(x[:2])
            z = x[2]
            ir = np.clip(np.searchsorted(r_edges, r_xy) - 1, 0, nr_tok - 1)
            iz = np.clip(np.searchsorted(z_edges, z) - 1, 0, nz - 1)
            Hf[ir, iz] += 1.0
        return {
            "modo": "rz_impactos",
            "H": Hf,
            "r_edges": r_edges,
            "z_edges": z_edges,
            "tipos": ["impactos_pared"],
        }

    z_min, z_max = _limites_rz(contenedor)[1:]
    z_edges = np.linspace(z_min, z_max, nz + 1)
    tipos = sorted({imp.get("tipo_pared", "lateral") for imp in impactos})
    if not tipos:
        tipos = ["(sin impactos)"]
    flujo = {t: np.zeros(nz, dtype=float) for t in tipos}

    for imp in impactos:
        t = imp.get("tipo_pared", "lateral")
        if t not in flujo:
            flujo[t] = np.zeros(nz, dtype=float)
            tipos.append(t)
        z = imp["x"][2]
        iz = np.clip(np.searchsorted(z_edges, z) - 1, 0, nz - 1)
        flujo[t][iz] += 1.0

    return {"modo": "tipos_z", "flujo": flujo, "z_edges": z_edges, "tipos": tipos}


def _historial_desde_pic(particulas, tiempos_escape, dt, intervalo=10):
    if not particulas or not particulas[0].historia_x:
        return []
    n_frames = min(len(p.historia_x) for p in particulas)
    esc_paso = {pid: int(round(t / dt)) for pid, t in tiempos_escape.items()}
    historial = []
    for k in range(0, n_frames, max(1, intervalo)):
        X_all = np.array([p.historia_x[k] for p in particulas])
        esc = np.array([
            p.id in esc_paso and k >= esc_paso[p.id] for p in particulas
        ])
        historial.append({"X": X_all, "escapo": esc})
    return historial


def _impactos_desde_pic(particulas, tiempos_escape, contenedor):
    impactos = []
    for p in particulas:
        if p.id not in tiempos_escape:
            continue
        x = np.asarray(p.x, dtype=float)
        impactos.append({
            "id": p.id,
            "x": x,
            "t": tiempos_escape[p.id],
            "tipo_pared": clasificar_impacto_pared(x, contenedor),
        })
    return impactos


def _fn_B_lite(config):
    B0, radio = config["B0"], config["radio"]
    if config["geometria"] == "tokamak":
        def fn_B(X):
            return np.array([
                campos_mod.campo_magnetico_tokamak(x, B0=B0, R=radio, Bpol=0.1)
                for x in X
            ])
        return fn_B
    return lambda X: campo_B_solenoide_vec(X, B0=B0, radio=radio)


def correr_simulacion_con_diagnostico(
    seed=SEED,
    n_particulas=None,
    pasos=None,
    dt=None,
    B0_local=None,
    radio=None,
    altura=None,
    T_plasma=None,
    nu=None,
    config=None,
    verbose=True,
):
    """Ejecuta motor_lite registrando posiciones e impactos (solo referencia / semana 13)."""
    if config is not None:
        cont = _crear_contenedor(config)
        n_p = config["n_total"]
        ps = config["pasos"]
        dt_u = config["dt"]
        T = config["T_plasma"]
        nu_u = config["nu_colision"]
        from Aplicaciones import construir_particulas
        particulas, motores, _ = construir_particulas(
            config["conteos"], cont, dt_u,
            T_plasma=T, nu=nu_u,
        )
        fn_B = _fn_B_lite(config)
    else:
        n_p = n_particulas if n_particulas is not None else N_PARTICULAS
        ps = pasos if pasos is not None else PASOS
        dt_u = dt if dt is not None else DT
        r = radio if radio is not None else RADIO
        h = altura if altura is not None else ALTURA
        T = T_plasma if T_plasma is not None else T_PLASMA
        nu_u = nu if nu is not None else NU_COLISION
        b0 = B0_local if B0_local is not None else B0
        rng = np.random.default_rng(seed)
        cont = ContenedorCilindrico(radio=r, altura=h)
        particulas, motores = _construir_particulas_custom(
            cont, rng, seed * 1000, n_p, T, nu_u, dt_u
        )
        fn_B = lambda X: campo_B_solenoide_vec(X, B0=b0, radio=r)

    historial = []
    impactos = []

    tiempos_escape, E_cin = motor_lite(
        pasos=ps,
        particulas=particulas,
        motores_colision=motores,
        fn_E=campo_E_cero_vec,
        fn_B=fn_B,
        dt=dt_u,
        contenedor=cont,
        registrar_energia=True,
        registrar_muestreo=True,
        intervalo_muestreo=INTERVALO_MUESTREO,
        historial_posiciones=historial,
        registrar_impactos=True,
        impactos_pared=impactos,
        verbose=verbose,
    )
    stats = mc.calcular_tau(tiempos_escape, n_p, dt_u, ps)
    return cont, historial, impactos, tiempos_escape, stats, E_cin


def _construir_particulas_custom(contenedor, rng, seed_base, n, T, nu, dt):
    particulas, motores = [], []
    for i in range(n):
        x0 = contenedor.posicion_aleatoria(rng)
        v0 = velocidad_inicial_mb(M, T, rng=rng)
        p = Particula(i, q=Q, m=M, x0=x0, v0=v0)
        col = ColisionEstocastica(nu=nu, m=M, T=T, dt=dt, seed=seed_base + i)
        particulas.append(p)
        motores.append(col)
    return particulas, motores


def guardar_datos(cont, H, r_edges, z_edges, flujo_info,
                  stats, impactos, historial, carpeta=GUARDAR_DIR,
                  config=None, motor_mapas="pic"):
    os.makedirs(carpeta, exist_ok=True)

    r_cent = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_cent = 0.5 * (z_edges[:-1] + z_edges[1:])
    Rg, Zg = np.meshgrid(r_cent, z_cent, indexing="ij")

    densidad_csv = np.column_stack([
        Rg.ravel(), Zg.ravel(), H.ravel(),
    ])
    np.savetxt(
        os.path.join(carpeta, "densidad_rz.csv"),
        densidad_csv,
        delimiter=",",
        header="r_m,z_m,densidad_norm",
        comments="",
    )

    ruta_flujo = os.path.join(carpeta, "flujo_pared.csv")
    if flujo_info.get("modo") == "rz_impactos":
        r_edges_f = flujo_info["r_edges"]
        z_edges_f = flujo_info["z_edges"]
        Hf = flujo_info["H"]
        r_cent_f = 0.5 * (r_edges_f[:-1] + r_edges_f[1:])
        z_cent_f = 0.5 * (z_edges_f[:-1] + z_edges_f[1:])
        filas = []
        for ir, rc in enumerate(r_cent_f):
            for iz, zc in enumerate(z_cent_f):
                filas.append([rc, zc, Hf[ir, iz]])
        np.savetxt(
            ruta_flujo,
            np.array(filas),
            delimiter=",",
            header="R_xy_m,z_m,conteo_impactos",
            comments="",
        )
    else:
        flujo = flujo_info["flujo"]
        z_edges_f = flujo_info["z_edges"]
        tipos = flujo_info["tipos"]
        z_cent_f = 0.5 * (z_edges_f[:-1] + z_edges_f[1:])
        filas = []
        for tipo in tipos:
            for iz, zc in enumerate(z_cent_f):
                filas.append([tipo, zc, flujo.get(tipo, np.zeros(len(z_cent_f)))[iz]])
        np.savetxt(
            ruta_flujo,
            np.array(filas, dtype=object),
            delimiter=",",
            fmt="%s",
            header="tipo_pared,z_m,conteo_impactos",
            comments="",
        )

    ruta_imp = os.path.join(carpeta, "impactos_detalle.csv")
    with open(ruta_imp, "w", encoding="utf-8") as f:
        f.write("id,t_s,x,y,z,tipo_pared\n")
        for imp in impactos:
            x = imp["x"]
            f.write(
                f"{imp['id']},{imp['t']:.6e},"
                f"{x[0]:.6e},{x[1]:.6e},{x[2]:.6e},{imp['tipo_pared']}\n"
            )

    cfg = config or {}
    meta = {
        "N_particulas": cfg.get("n_total", N_PARTICULAS),
        "pasos": cfg.get("pasos", PASOS),
        "dt_s": cfg.get("dt", DT),
        "B0_T": cfg.get("B0", B0),
        "radio_m": cfg.get("radio", RADIO),
        "altura_m": cfg.get("altura", ALTURA),
        "geometria": cfg.get("geometria", "cilindro"),
        "motor_mapas": motor_mapas,
        "fuente": (
            "Misma corrida PIC (historia_x + tiempos_escape)"
            if motor_mapas == "pic"
            else "Corrida motor_lite separada (referencia)"
        ),
        "n_muestras": len(historial),
        "n_impactos": len(impactos),
        "tau_medio_s": float(stats["tau_medio"]),
        "frac_confinadas": float(stats["n_confinadas"] / stats["n_total"]),
    }
    with open(os.path.join(carpeta, "config_corrida.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Datos guardados en {carpeta}/")


def graficar_mapas(H, r_edges, z_edges, flujo_info,
                   carpeta=GUARDAR_DIR, B0_plot=None, radio_plot=None, n_plot=None,
                   motor_mapas="pic", geometria="cilindro"):
    B0_plot = B0_plot if B0_plot is not None else B0
    radio_plot = radio_plot if radio_plot is not None else RADIO
    n_plot = n_plot if n_plot is not None else N_PARTICULAS
    os.makedirs(carpeta, exist_ok=True)

    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    r_cent = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_cent = 0.5 * (z_edges[:-1] + z_edges[1:])
    im0 = ax0.pcolormesh(
        z_cent * 1e3, r_cent * 1e3, H,
        shading="auto", cmap="inferno",
    )
    ax0.set_xlabel("z [mm]")
    ax0.set_ylabel("R_xy [mm]")
    tit_dens = "Densidad (misma corrida PIC)" if motor_mapas == "pic" else "Densidad (motor lite)"
    ax0.set_title(tit_dens)
    plt.colorbar(im0, ax=ax0, label="densidad norm.")

    ax1 = fig.add_subplot(gs[0, 1])
    if flujo_info.get("modo") == "rz_impactos":
        r_edges_f = flujo_info["r_edges"]
        z_edges_f = flujo_info["z_edges"]
        r_cent_f = 0.5 * (r_edges_f[:-1] + r_edges_f[1:])
        z_cent_f = 0.5 * (z_edges_f[:-1] + z_edges_f[1:])
        im1 = ax1.pcolormesh(
            z_cent_f * 1e3, r_cent_f * 1e3, flujo_info["H"],
            shading="auto", cmap="hot",
        )
        ax1.set_xlabel("z [mm]")
        ax1.set_ylabel("R_xy [mm]")
        ax1.set_title("Impactos en pared (misma corrida PIC)")
        plt.colorbar(im1, ax=ax1, label="N impactos")
    else:
        flujo = flujo_info["flujo"]
        tipos = flujo_info["tipos"]
        z_edges_f = flujo_info["z_edges"]
        z_cent_f = 0.5 * (z_edges_f[:-1] + z_edges_f[1:])
        mat = np.vstack([flujo[t] for t in tipos])
        im1 = ax1.imshow(
            mat,
            aspect="auto",
            origin="lower",
            extent=[z_cent_f[0] * 1e3, z_cent_f[-1] * 1e3, -0.5, len(tipos) - 0.5],
            cmap="hot",
        )
        ax1.set_yticks(range(len(tipos)))
        ax1.set_yticklabels(tipos)
        ax1.set_xlabel("z [mm]")
        ax1.set_title("Flujo de impactos en pared")
        plt.colorbar(im1, ax=ax1, label="N impactos")

    fuente = "PIC" if motor_mapas == "pic" else "lite"
    fig.suptitle(
        f"Mapas ({fuente}) — {geometria}, B₀={B0_plot} T, "
        f"R={radio_plot*1e3:.1f} mm, N={n_plot}",
        fontsize=11,
    )
    ruta = os.path.join(carpeta, "mapas_calor.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura guardada: {ruta}")


def _mapas_desde_pic(resultado: dict, config: dict, carpeta: str):
    cont = resultado["contenedor"]
    particulas = resultado["particulas"]
    te = resultado["tiempos_escape"]
    dt, pasos, n = config["dt"], config["pasos"], config["n_total"]
    stats = mc.calcular_tau(te, n, dt, pasos)

    historial = _historial_desde_pic(particulas, te, dt, intervalo=10)
    impactos = _impactos_desde_pic(particulas, te, cont)
    print(
        f"  [Mapas PIC] {len(historial)} muestras, "
        f"{len(impactos)} impactos (= escapes MC)"
    )

    H, r_edges, z_edges = densidad_rz_desde_muestreos(historial, cont)
    flujo_info = flujo_pared_desde_impactos(impactos, cont)
    guardar_datos(
        cont, H, r_edges, z_edges, flujo_info,
        stats, impactos, historial, carpeta=carpeta, config=config,
        motor_mapas="pic",
    )
    graficar_mapas(
        H, r_edges, z_edges, flujo_info,
        carpeta=carpeta,
        B0_plot=config["B0"],
        radio_plot=config["radio"],
        n_plot=config["n_total"],
        motor_mapas="pic",
        geometria=config.get("geometria", "?"),
    )
    return H, flujo_info, stats


def _mapas_desde_lite(config: dict, carpeta: str, seed=42):
    print("\n  [Mapas] Corrida motor_lite (referencia, no PIC)...")
    cont, historial, impactos, te, stats, _ = correr_simulacion_con_diagnostico(
        seed=seed, config=config, verbose=False,
    )
    H, r_edges, z_edges = densidad_rz_desde_muestreos(historial, cont)
    flujo_info = flujo_pared_desde_impactos(impactos, cont)
    guardar_datos(
        cont, H, r_edges, z_edges, flujo_info,
        stats, impactos, historial, carpeta=carpeta, config=config,
        motor_mapas="lite",
    )
    graficar_mapas(
        H, r_edges, z_edges, flujo_info,
        carpeta=carpeta,
        B0_plot=config["B0"],
        radio_plot=config["radio"],
        n_plot=config["n_total"],
        motor_mapas="lite",
        geometria=config.get("geometria", "?"),
    )
    return H, flujo_info, stats


def generar_mapas_desde_corrida(
    config: dict,
    resultado: dict = None,
    carpeta_salida: str = None,
    seed=42,
):
    """
    Mapas alineados con la corrida principal.
    motor=pic + resultado → historia_x y tiempos_escape (mismo panel MC).
    Sin resultado o motor=lite → motor_lite con la geometría de config.
    """
    carpeta = carpeta_salida or GUARDAR_DIR
    os.makedirs(carpeta, exist_ok=True)
    motor = config.get("motor", "pic")

    if resultado is not None and motor == "pic":
        print("\n  [Mapas] Desde corrida PIC (no se relanza motor_lite)...")
        return _mapas_desde_pic(resultado, config, carpeta)
    if resultado is not None and motor == "lite":
        print("\n  [Mapas] motor=lite: re-simulación con muestreo...")
        return _mapas_desde_lite(config, carpeta, seed=seed)
    print("\n  [AVISO] Sin resultado PIC: corrida lite separada.")
    return _mapas_desde_lite(config, carpeta, seed=seed)


def ejecutar_mapas_en_carpeta(config: dict, carpeta_salida: str, resultado=None, seed=42):
    """Compatibilidad: preferir generar_mapas_desde_corrida(..., resultado=...)."""
    return generar_mapas_desde_corrida(
        config, resultado=resultado, carpeta_salida=carpeta_salida, seed=seed,
    )


def ejecutar_semana13(seed=SEED):
    print("=== Semana 13: mapas de calor (motor_lite cilindro demo) ===")
    cont, historial, impactos, te, stats, E_cin = correr_simulacion_con_diagnostico(
        seed
    )
    H, r_edges, z_edges = densidad_rz_desde_muestreos(historial, cont)
    flujo_info = flujo_pared_desde_impactos(impactos, cont)
    guardar_datos(
        cont, H, r_edges, z_edges, flujo_info,
        stats, impactos, historial, motor_mapas="lite",
    )
    graficar_mapas(H, r_edges, z_edges, flujo_info, motor_mapas="lite")
    print(f"  tau medio = {stats['tau_medio']*1e6:.2f} us")
    frac = stats["n_confinadas"] / stats["n_total"]
    print(f"  Fraccion confinadas = {frac*100:.1f}%")
    print(f"  Impactos registrados = {len(impactos)}")
    return H, flujo_info, stats


if __name__ == "__main__":
    ejecutar_semana13()
