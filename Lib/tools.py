
"""
Funciones auxiliares: guardado de trayectorias y pruebas helicoidales.
"""
import os
import numpy as np
from particulas import Particula
from integradores import boris_step

def guardar_logs_trayectorias(particulas, dt, carpeta_salida="data"):
    """
    Guarda las trayectorias de N partículas en archivos CSV individuales.
    """
    # Creamos la carpeta si no existe
    os.makedirs(carpeta_salida, exist_ok=True)

    for i, p in enumerate(particulas):
        # Convertimos la historia de posiciones a un array de NumPy
        trayectoria = np.array(p.historia_x)
        
        # Si la partícula no tiene historia (ej. pasos=0), saltamos para evitar errores
        if trayectoria.size == 0:
            continue
            
        pasos_totales = trayectoria.shape[0]
        
        # Creamos el vector de tiempo
        t = np.arange(pasos_totales) * dt

        # Unimos el tiempo con las coordenadas (x, y, z)
        datos = np.column_stack((t, trayectoria))

        # Generamos un nombre dinámico: trayectoria_p1.csv, trayectoria_p2.csv...
        nombre_archivo = f"trayectoria_p{i+1}.csv"
        ruta_completa = os.path.join(carpeta_salida, nombre_archivo)

        # Guardamos el archivo
        np.savetxt(
            ruta_completa,
            datos,
            delimiter=",",
            header="t,x,y,z",
            comments="",
        )
    
    print(f"Se han guardado las trayectorias de {len(particulas)} partículas en '{carpeta_salida}'.")


def prueba_trayectoria_helicoidal(
    q=1e-6,
    m=1e-9,
    B0=1.0,
    v_perp_mod=1e3,
    v_par_mod=1e3,
    dt=5e-5,
    pasos=800,
):
    """
    Prueba de una trayectoria helicoidal en un campo B uniforme y validación
    del radio de Larmor simulando una sola partícula sin interacciones.
    """
    v_perp = np.array([0.0, -v_perp_mod, 0.0])
    v_par = np.array([0.0, 0.0, v_par_mod])
    v0 = v_perp + v_par

    r_larmor_teorico = m * np.linalg.norm(v_perp) / (abs(q) * B0)
    x0 = np.array([r_larmor_teorico, 0.0, 0.0])

    p = Particula(id_particula=99, q=q, m=m, x0=x0, v0=v0)

    for _ in range(pasos):
        B = np.array([0.0, 0.0, B0])
        E = np.zeros(3)
        x_nueva, v_nueva = boris_step(p.x, p.v, E, B, p.q, p.m, dt)
        p.actualizar_estado(x_nueva, v_nueva)

    trayectoria = np.array(p.historia_x)
    r_xy = np.sqrt(trayectoria[:, 0] ** 2 + trayectoria[:, 1] ** 2)
    r_larmor_numerico = r_xy.mean()

    os.makedirs("data", exist_ok=True)
    datos = np.column_stack((np.arange(trayectoria.shape[0]) * dt, trayectoria))
    np.savetxt(
        os.path.join("data", "trayectoria_helicoidal.csv"),
        datos,
        delimiter=",",
        header="t,x,y,z",
        comments="",
    )

    print("=== Prueba de trayectoria helicoidal ===")
    print(f"Radio de Larmor teórico:  {r_larmor_teorico:.6e} m")
    print(f"Radio de Larmor numérico: {r_larmor_numerico:.6e} m")


