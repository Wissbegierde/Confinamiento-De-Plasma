import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib

from integradores import boris_step
from particulas import Particula
from interacciones import calcular_E_interaccion
from campos import campo_electrico_constante, campo_magnetico_solenoide
from contenedor import ContenedorCilindrico
from colisiones import ColisionEstocastica, velocidad_inicial_mb  

# Forzar backend interactivo para que abra la ventana
matplotlib.use('TkAgg')


def guardar_logs_trayectorias(p1, p2, dt, carpeta_salida="data"):
    os.makedirs(carpeta_salida, exist_ok=True)

    trayectoria1 = np.array(p1.historia_x)
    trayectoria2 = np.array(p2.historia_x)
    pasos_totales = trayectoria1.shape[0]
    t = np.arange(pasos_totales) * dt

    datos1 = np.column_stack((t, trayectoria1))
    datos2 = np.column_stack((t, trayectoria2))

    np.savetxt(
        os.path.join(carpeta_salida, "trayectoria_p1.csv"),
        datos1,
        delimiter=",",
        header="t,x,y,z",
        comments="",
    )
    np.savetxt(
        os.path.join(carpeta_salida, "trayectoria_p2.csv"),
        datos2,
        delimiter=",",
        header="t,x,y,z",
        comments="",
    )


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
    v_perp = np.array([0.0, v_perp_mod, 0.0])
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


# --- 2. CONFIGURACIÓN ---
dt = 5e-5
pasos = 800
pausado = True
frame_actual = 0

# Parámetros físicos
p_radio = 0.5
B0 = 1.0
E0 = (0.0, 0.0, 0.0)
m_particula = 1e-9          # masa [kg]
q_particula = 1e-6          # carga [C]
T_plasma = 1e4              # temperatura del plasma [K]   
nu_colision = 500.0         # frecuencia de colisión [Hz]  

contenedor = ContenedorCilindrico(radio=p_radio)

# Velocidades iniciales distribuidas según Maxwell-Boltzmann
rng_global = np.random.default_rng(42)
v0_p1 = velocidad_inicial_mb(m_particula, T_plasma, rng=rng_global)
v0_p2 = velocidad_inicial_mb(m_particula, T_plasma, rng=rng_global)

p1 = Particula(1, q=q_particula, m=m_particula, x0=[-0.05, 0, 0], v0=v0_p1)
p2 = Particula(2, q=q_particula, m=m_particula, x0=[ 0.05, 0, 0], v0=v0_p2)

# Modelos estocásticos de colisión, uno por partícula
col1 = ColisionEstocastica(nu=nu_colision, m=m_particula, T=T_plasma, dt=dt, seed=1)
col2 = ColisionEstocastica(nu=nu_colision, m=m_particula, T=T_plasma, dt=dt, seed=2)

# Pre-cálculo
for _ in range(pasos):
    # Campos externos
    B1 = campo_magnetico_solenoide(p1.x, B0=B0, radio=p_radio)
    B2 = campo_magnetico_solenoide(p2.x, B0=B0, radio=p_radio)
    E_ext_1 = campo_electrico_constante(p1.x, E0=E0)
    E_ext_2 = campo_electrico_constante(p2.x, E0=E0)

    # Interacción entre partículas (Coulomb simple)
    E1_int = calcular_E_interaccion(p1, p2)
    E2_int = calcular_E_interaccion(p2, p1)

    E1 = E_ext_1 + E1_int
    E2 = E_ext_2 + E2_int

    x1, v1 = boris_step(p1.x, p1.v, E1, B1, p1.q, p1.m, dt)
    x2, v2 = boris_step(p2.x, p2.v, E2, B2, p2.q, p2.m, dt)

    # Colisión estocástica: redistribución de velocidad (Semana 5)
    v1, _ = col1.aplicar(v1)
    v2, _ = col2.aplicar(v2)

    # Detección de colisión con la pared del contenedor
    if not contenedor.esta_dentro(x1):
        x1 = contenedor.proyectar_a_frontera(x1)
        v1 = np.zeros(3)
    if not contenedor.esta_dentro(x2):
        x2 = contenedor.proyectar_a_frontera(x2)
        v2 = np.zeros(3)

    p1.actualizar_estado(x1, v1)
    p2.actualizar_estado(x2, v2)

print(col1.resumen())
print(col2.resumen())

guardar_logs_trayectorias(p1, p2, dt)
prueba_trayectoria_helicoidal()

# --- 3. INTERFAZ ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.grid(True)
ax.set_title("ESPACIO: Pausa | FLECHAS: <- / ->")

punto1, = ax.plot([], [], 'ro', markersize=10, label="P1")
punto2, = ax.plot([], [], 'bo', markersize=10, label="P2")
ax.legend()

def presionar_tecla(event):
    global pausado, frame_actual
    if event.key == ' ':
        pausado = not pausado
    elif event.key == 'right':
        pausado = True
        frame_actual = (frame_actual + 1) % pasos
    elif event.key == 'left':
        pausado = True
        frame_actual = (frame_actual - 1) % pasos
    # Actualización inmediata al presionar flechas
    pos1 = p1.historia_x[frame_actual]
    pos2 = p2.historia_x[frame_actual]
    punto1.set_data([pos1[0]], [pos1[1]])
    punto2.set_data([pos2[0]], [pos2[1]])
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('key_press_event', presionar_tecla)

def update(frame):
    global frame_actual
    if not pausado:
        frame_actual = (frame_actual + 1) % pasos
        pos1 = p1.historia_x[frame_actual]
        pos2 = p2.historia_x[frame_actual]
        punto1.set_data([pos1[0]], [pos1[1]])
        punto2.set_data([pos2[0]], [pos2[1]])
    return punto1, punto2

ani = FuncAnimation(fig, update, interval=10, blit=True, save_count=pasos)
plt.show()