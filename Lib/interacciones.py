import numpy as np


def calcular_E_interaccion(p_target, p_source):
    ke = 8.987e9
    r_vec = p_target.x - p_source.x
    dist = np.linalg.norm(r_vec)
    if dist < 1e-3:
        return np.zeros(3)
    return ke * p_source.q * r_vec / dist**3

