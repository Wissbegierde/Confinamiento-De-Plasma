"""
ui_consola.py — Menús interactivos para configurar una simulación.
"""

from Aplicaciones import (
    ESPECIES, ESCALAS, GEOMETRIAS,
    pedir_parametros, pedir_campos, pedir_geometria,
    pedir_escala,
    _float, _int,
)


def _mostrar_especies_disponibles():
    nombres = ", ".join(ESPECIES.keys())
    print(f"\n  Nombres válidos: {nombres}")
    for nombre, esp in ESPECIES.items():
        print(f"    {nombre:10s}  [{esp['label']}]")


def pedir_especies():
    """
    Pregunta solo si usar todas las especies o algunas por nombre.
    La cantidad por especie se pide una sola vez (mismo N para cada una).
    """
    print("\n╔══════════════════════════════════════╗")
    print("║   ESPECIES                           ║")
    print("╚══════════════════════════════════════╝")
    _mostrar_especies_disponibles()
    print("\n  [1] Usar TODAS las especies")
    print("  [2] Elegir algunas (escribir nombres)")
    while True:
        op = input("  Opción [1/2, default=1]: ").strip() or "1"
        if op in ("1", "2"):
            break
        print("  → Opción inválida")

    if op == "1":
        nombres = list(ESPECIES.keys())
    else:
        print("\n  Escribe los nombres separados por espacio o coma.")
        print("  Ej: proton electron   |   proton, hidrogeno, helio")
        while True:
            raw = input("  Especies: ").strip().lower()
            tokens = [
                t.strip()
                for t in raw.replace(",", " ").split()
                if t.strip()
            ]
            invalid = [t for t in tokens if t not in ESPECIES]
            if tokens and not invalid:
                nombres = list(dict.fromkeys(tokens))
                break
            if invalid:
                print(f"  → No válidos: {', '.join(invalid)}")
            else:
                print("  → Escribe al menos un nombre.")

    n_por = int(_float("  Partículas por especie (ej: 20): ", default=20))
    conteos = {nombre: n_por for nombre in nombres}
    n_total = sum(conteos.values())
    print(f"\n  Especies activas: {', '.join(nombres)}")
    print(f"  {n_por} por especie → {n_total} partículas en total")
    return conteos, n_total


def pedir_etiqueta() -> str:
    print("\n╔══════════════════════════════════════╗")
    print("║   ETIQUETA DE LA CORRIDA             ║")
    print("╚══════════════════════════════════════╝")
    print("  Nombre corto para distinguir esta simulación")
    print("  (ej: entrega1, barrido_alto_B, prueba_tapas)")
    return input("  Etiqueta [opcional]: ").strip()


def pedir_motor() -> str:
    print("\n╔══════════════════════════════════════╗")
    print("║   MOTOR DE SIMULACIÓN                ║")
    print("╚══════════════════════════════════════╝")
    print("  [1] PIC completo (Poisson + colisiones) — más lento, más físico")
    print("  [2] Lite vectorizado — más rápido, ideal para MC y mapas de calor")
    while True:
        op = input("  Motor [1/2, default=1]: ").strip() or "1"
        if op == "1":
            return "pic"
        if op == "2":
            return "lite"
        print("  → Opción inválida")


def pedir_exportaciones() -> dict:
    print("\n╔══════════════════════════════════════╗")
    print("║   SALIDAS A GENERAR                  ║")
    print("╚══════════════════════════════════════╝")
    def _si(msg, default=True):
        hint = "S/n" if default else "s/N"
        raw = input(f"  {msg} [{hint}]: ").strip().lower()
        if raw == "":
            return default
        return raw in ("s", "si", "sí", "y", "yes", "1")

    return {
        "trayectorias": _si("Guardar trayectorias CSV", True),
        "montecarlo": _si("Estadísticas y gráficas Monte Carlo", True),
        "mapas_calor": _si("Mapas de calor (densidad + flujo pared)", False),
        "visualizacion_3d": _si("Abrir animación 3D al final", True),
    }


def pedir_plasma() -> tuple:
    print("\n╔══════════════════════════════════════╗")
    print("║   PLASMA (colisiones)                ║")
    print("╚══════════════════════════════════════╝")
    T = _float("  Temperatura T [K] (ej: 1e4): ", default=1e4)
    nu = _float("  Frecuencia colisión nu [Hz] (ej: 500): ", default=500.0)
    return T, nu


def recoger_config() -> dict:
    """Pregunta todos los parámetros y devuelve un dict serializable."""
    print("\n╔══════════════════════════════════════╗")
    print("║   NUEVA SIMULACIÓN                   ║")
    print("╚══════════════════════════════════════╝")

    etiqueta = pedir_etiqueta()
    motor = pedir_motor()
    dt, pasos = pedir_parametros()
    B0, E0 = pedir_campos()
    geo = pedir_geometria()
    escala = pedir_escala()
    T_plasma, nu = pedir_plasma()
    exportar = pedir_exportaciones()

    print("\n  Dimensiones del contenedor:")
    radio = _float("  Radio  (m) [ej: 0.01]: ", default=0.01)
    altura = _float("  Altura (m) [ej: 0.02]: ", default=0.02)

    conteos, n_total = pedir_especies()

    return {
        "etiqueta": etiqueta,
        "motor": motor,
        "dt": dt,
        "pasos": pasos,
        "B0": B0,
        "E0": list(E0),
        "geometria": geo,
        "escala": escala,
        "radio": radio,
        "altura": altura,
        "conteos": conteos,
        "n_total": n_total,
        "T_plasma": T_plasma,
        "nu_colision": nu,
        "exportar": exportar,
    }
