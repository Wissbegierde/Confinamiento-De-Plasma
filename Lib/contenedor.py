import numpy as np


class ContenedorCilindrico:
    """
    Contenedor cilíndrico infinito en la dirección z.
    La frontera está dada por un radio fijo en el plano (x, y).
    """

    def __init__(self, radio):
        self.radio = float(radio)

    def esta_dentro(self, x):
        r = np.linalg.norm(x[:2])
        return r <= self.radio

    def proyectar_a_frontera(self, x):
        """
        Si la partícula está fuera, la "pega" a la pared manteniendo z.
        """
        r = np.linalg.norm(x[:2])
        if r == 0:
            return x
        factor = self.radio / r
        return np.array([x[0] * factor, x[1] * factor, x[2]])

