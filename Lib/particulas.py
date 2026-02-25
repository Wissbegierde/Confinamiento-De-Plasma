import numpy as np


class Particula:
    def __init__(self, id_particula, q, m, x0, v0):
        self.id = id_particula
        self.q = q
        self.m = m
        self.x = np.array(x0, dtype=float)
        self.v = np.array(v0, dtype=float)
        self.historia_x = [self.x.copy()]

    def actualizar_estado(self, x_nueva, v_nueva):
        self.x = x_nueva
        self.v = v_nueva
        self.historia_x.append(self.x.copy())

