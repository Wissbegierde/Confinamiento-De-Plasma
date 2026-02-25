import numpy as np


def boris_step(x, v, E, B, q, m, dt):
    v_menos = v + (q / m) * (dt / 2) * E
    t = (q / m) * (dt / 2) * B
    s = 2 * t / (1 + np.dot(t, t))
    v_prima = v_menos + np.cross(v_menos, t)
    v_mas = v_menos + np.cross(v_prima, s)
    v_nueva = v_mas + (q / m) * (dt / 2) * E
    x_nueva = x + v_nueva * dt
    return x_nueva, v_nueva

