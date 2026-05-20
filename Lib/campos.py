import numpy as np

# ══════════════════════════════════════════════════════════════
#  CAMPOS ELÉCTRICOS
# ══════════════════════════════════════════════════════════════

def campo_electrico_constante(pos, E0=(0.0, 0.0, 0.0)):
    """
    Campo uniforme en cualquier dirección.
    E0 = (Ex, Ey, Ez) en V/m
    Útil para: aceleración de haces, placas paralelas.
    """
    return np.array(E0, dtype=float)


def campo_electrico_radial(pos, E0=1.0, centro=(0.0, 0.0, 0.0)):
    """
    Campo radial que apunta desde/hacia un centro.
    E0 > 0 -> repulsivo (como carga positiva en el centro)
    E0 < 0 -> atractivo
    Útil para: confinamiento radial, trampa de Penning.
    """
    r = pos - np.array(centro)
    dist = np.linalg.norm(r)
    if dist < 1e-10:
        return np.zeros(3)
    return E0 * r / dist**2


def campo_electrico_cuadrupolar(pos, k=1.0):
    """
    Campo cuadrupolar: E = k*(x, -y, 0)  [trampa de cuadrupolo 2D]
    o en 3D:          E = k*(x, y, -2z)  [trampa de Paul]
    Útil para: confinamiento de iones, trampas de Paul.
    """
    x, y, z = pos
    return k * np.array([x, y, -2*z])


def campo_electrico_oscilante(pos, E0=(0.0, 0.0, 1.0), omega=1e6, t=0.0):
    """
    Campo oscilante en el tiempo: E = E0 * cos(omega * t)
    Útil para: trampas de RF, aceleración resonante.
    """
    return np.array(E0) * np.cos(omega * t)


def campo_electrico_dipolar(pos, p_dipolo=(0.0, 0.0, 1.0), r0=(0.0, 0.0, 0.0)):
    """
    Campo de un dipolo eléctrico en r0 con momento p_dipolo.
    Útil para: modelar electrodos, antenas cercanas.
    """
    ke = 8.987e9
    p = np.array(p_dipolo)
    r = pos - np.array(r0)
    dist = np.linalg.norm(r)
    if dist < 1e-10:
        return np.zeros(3)
    r_hat = r / dist
    return ke / dist**3 * (3 * np.dot(p, r_hat) * r_hat - p)


def campo_electrico_lineal_z(pos, dEdz=1.0, z0=0.0):
    """
    Campo que crece linealmente en Z: E = dEdz * (z - z0) * z_hat
    Útil para: gradientes de campo, lentes electrostáticas.
    """
    return np.array([0.0, 0.0, dEdz * (pos[2] - z0)])


# ══════════════════════════════════════════════════════════════
#  CAMPOS MAGNÉTICOS
# ══════════════════════════════════════════════════════════════

def campo_magnetico_constante(pos, B0=1.0, direccion=(0.0, 0.0, 1.0)):
    """
    Campo uniforme en cualquier dirección.
    Útil para: solenoide ideal, campo de fondo de tokamak.
    """
    d = np.array(direccion, dtype=float)
    return B0 * d / np.linalg.norm(d)


def campo_magnetico_solenoide(pos, B0=1.0, radio=0.5, eje='z'):
    """
    Aproximación de solenoide finito: B uniforme dentro, 0 fuera.
    Útil para: confinamiento magnético básico, experimentos de laboratorio.
    """
    ejes = {'x': 0, 'y': 1, 'z': 2}
    idx_radial = [i for i in range(3) if i != ejes[eje]]
    r = np.linalg.norm([pos[i] for i in idx_radial])
    
    B = np.zeros(3)
    if r <= radio:
        B[ejes[eje]] = B0
    return B


def campo_magnetico_dipolar(pos, m_dipolo=(0.0, 0.0, 1.0), r0=(0.0, 0.0, 0.0)):
    """
    Campo de dipolo magnético (como una barra magnética o bobina pequeña).
    Útil para: campo terrestre simplificado, espejo magnético.
    """
    mu0 = 4 * np.pi * 1e-7
    m = np.array(m_dipolo)
    r = pos - np.array(r0)
    dist = np.linalg.norm(r)
    if dist < 1e-10:
        return np.zeros(3)
    r_hat = r / dist
    return (mu0 / (4 * np.pi)) / dist**3 * (3 * np.dot(m, r_hat) * r_hat - m)


def campo_magnetico_espejo(pos, B0=1.0, L=1.0, Bm=3.0):
    """
    Espejo magnético: B(z) = B0 * (1 + (Bm-1)*(z/L)²)
    Más fuerte en los extremos → atrapa partículas en el centro.
    Útil para: confinamiento en reactores de espejo, Van Allen.
    """
    z = pos[2]
    Bz = B0 * (1 + (Bm - 1) * (z / L)**2)
    # Componente radial por conservación de flujo: Br = -r/2 * dBz/dz
    r = np.linalg.norm(pos[:2])
    dBz_dz = B0 * 2 * (Bm - 1) * z / L**2
    Br = -r / 2 * dBz_dz
    # Dirección radial
    if r < 1e-10:
        return np.array([0.0, 0.0, Bz])
    r_hat = pos[:2] / r
    return np.array([Br * r_hat[0], Br * r_hat[1], Bz])


def campo_magnetico_tokamak(pos, B0=1.0, R=1.0, Bpol=0.1):
    """
    Campo de tokamak: toroidal (dominante) + poloidal (confinamiento).
    B_tor = B0 * R / r_xy  (decae con la distancia al eje)
    B_pol = Bpol en dirección azimutal en el plano del tubo
    Útil para: simulación de plasma en reactores de fusión.
    """
    x, y, z = pos
    r_xy = np.linalg.norm([x, y])
    if r_xy < 1e-10:
        return np.zeros(3)

    # Campo toroidal: circunda el eje Z, decae como 1/r
    phi_hat = np.array([-y, x, 0.0]) / r_xy
    B_tor = B0 * R / r_xy * phi_hat

    # Campo poloidal: circunda el tubo del toro
    # Vector desde el círculo central hacia la partícula
    centro_tubo = np.array([x, y, 0.0]) * (R / r_xy)
    vec_pol = pos - centro_tubo
    dist_pol = np.linalg.norm(vec_pol)
    if dist_pol < 1e-10:
        return B_tor
    # Dirección poloidal: perpendicular a vec_pol en el plano del tubo
    # Rotación 90° en el plano (vec_pol × phi_hat)
    pol_hat = np.cross(phi_hat, vec_pol / dist_pol)
    B_pol = Bpol * pol_hat

    return B_tor + B_pol


def campo_magnetico_cuadrupolar(pos, G=1.0):
    """
    Cuadrupolo magnético: B = G * (y, x, 0)
    Útil para: enfocamiento de haces (lentes magnéticas), aceleradores.
    """
    x, y, z = pos
    return G * np.array([y, x, 0.0])


def campo_magnetico_helicoidal(pos, B0=1.0, Bz=0.5, k=2*np.pi):
    """
    Campo helicoidal: B gira en XY mientras avanza en Z.
    B = B0*(cos(kz), sin(kz), 0) + Bz*z_hat
    Útil para: stellarators simplificados, ondas de plasma.
    """
    z = pos[2]
    return np.array([B0 * np.cos(k*z), B0 * np.sin(k*z), Bz])