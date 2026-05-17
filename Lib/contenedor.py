import numpy as np

class ContenedorBase:
    """Clase base: define la interfaz común para todos los contenedores."""
    
    def esta_dentro(self, x):
        raise NotImplementedError
    
    def posicion_aleatoria(self, rng):
        raise NotImplementedError
    
    def proyectar_a_frontera(self, x):
        raise NotImplementedError
    
    def manejar_colision(self, x_new, v_new, x_old):
        """
        Lógica general de colisión con la pared.
        Si la partícula salió, la pega a la frontera y anula su velocidad.
        Retorna (x_final, v_final, choco:bool)
        """
        if not self.esta_dentro(x_new):
            x_pared = self.proyectar_a_frontera(x_new)
            return x_pared, np.zeros(3), True
        return x_new, v_new, False


# ─────────────────────────────────────────────
class ContenedorCilindrico(ContenedorBase):
    """
    Cilindro recto con eje en Z.
    """
    def __init__(self, radio, altura):
        self.radio = radio
        self.altura = altura
        self.z_min = -altura / 2
        self.z_max =  altura / 2

    def esta_dentro(self, x):
        r_xy = np.linalg.norm(x[:2])
        return r_xy <= self.radio and self.z_min <= x[2] <= self.z_max

    def posicion_aleatoria(self, rng):
        while True:
            px = rng.uniform(-self.radio, self.radio)
            py = rng.uniform(-self.radio, self.radio)
            if px**2 + py**2 <= self.radio**2:
                pz = rng.uniform(self.z_min, self.z_max)
                return np.array([px, py, pz])

    def proyectar_a_frontera(self, x):
        r = np.linalg.norm(x[:2])
        # Pared lateral
        if r > self.radio:
            if r == 0:
                return np.array([self.radio, 0, x[2]])
            factor = self.radio / r
            px = x[0] * factor
            py = x[1] * factor
        else:
            px, py = x[0], x[1]
        # Tapas superior/inferior
        pz = np.clip(x[2], self.z_min, self.z_max)
        return np.array([px, py, pz])


# ─────────────────────────────────────────────
class ContenedorEsferico(ContenedorBase):
    """
    Esfera centrada en el origen.
    """
    def __init__(self, radio):
        self.radio = radio

    def esta_dentro(self, x):
        return np.linalg.norm(x) <= self.radio

    def posicion_aleatoria(self, rng):
        while True:
            p = rng.uniform(-self.radio, self.radio, size=3)
            if np.linalg.norm(p) <= self.radio:
                return p

    def proyectar_a_frontera(self, x):
        r = np.linalg.norm(x)
        if r == 0:
            return np.array([self.radio, 0.0, 0.0])
        return x * (self.radio / r)


# ─────────────────────────────────────────────
class ContenedorCaja(ContenedorBase):
    """
    Caja rectangular (paralelepípedo).
    """
    def __init__(self, Lx, Ly, Lz):
        self.lim = np.array([Lx, Ly, Lz]) / 2  # semilados

    def esta_dentro(self, x):
        return np.all(np.abs(x) <= self.lim)

    def posicion_aleatoria(self, rng):
        return rng.uniform(-self.lim, self.lim)

    def proyectar_a_frontera(self, x):
        # Proyecta al punto más cercano en la superficie de la caja
        return np.clip(x, -self.lim, self.lim)


# ─────────────────────────────────────────────
class ContenedorPlacasParalelas(ContenedorBase):
    """
    Dos placas infinitas separadas una distancia d en el eje Z.
    En X e Y se limita con L para la distribución inicial.
    """
    def __init__(self, d, L=1.0):
        self.d = d          # separación entre placas
        self.L = L          # tamaño lateral (para distribución)
        self.z_min = -d / 2
        self.z_max =  d / 2

    def esta_dentro(self, x):
        return self.z_min <= x[2] <= self.z_max

    def posicion_aleatoria(self, rng):
        return np.array([
            rng.uniform(-self.L/2, self.L/2),
            rng.uniform(-self.L/2, self.L/2),
            rng.uniform(self.z_min, self.z_max)
        ])

    def proyectar_a_frontera(self, x):
        pz = np.clip(x[2], self.z_min, self.z_max)
        return np.array([x[0], x[1], pz])


# ─────────────────────────────────────────────
class ContenedorTokamak(ContenedorBase):
    """
    Toro (donut): radio mayor R (centro del tubo al centro del toro),
                  radio menor a (radio del tubo).
    Eje de simetría en Z.
    """
    def __init__(self, R, a):
        self.R = R  # radio mayor
        self.a = a  # radio menor (debe ser a < R)

    def _distancia_al_tubo(self, x):
        """Distancia desde x hasta el círculo central del toro."""
        r_xy = np.linalg.norm(x[:2])          # distancia al eje Z
        return np.sqrt((r_xy - self.R)**2 + x[2]**2)

    def esta_dentro(self, x):
        return self._distancia_al_tubo(x) <= self.a

    def posicion_aleatoria(self, rng):
        """Muestreo por rechazo dentro del toro."""
        lim = self.R + self.a
        while True:
            p = rng.uniform([-lim, -lim, -self.a], [lim, lim, self.a])
            if self._distancia_al_tubo(p) <= self.a:
                return p

    def proyectar_a_frontera(self, x):
        """
        Proyecta al punto más cercano en la superficie del toro.
        1. Encuentra el punto más cercano en el círculo central.
        2. Desde ahí proyecta radialmente al radio menor.
        """
        r_xy = np.linalg.norm(x[:2])
        if r_xy == 0:
            # Caso degenerado: sobre el eje Z
            centro = np.array([self.R, 0.0, 0.0])
        else:
            # Punto más cercano en el círculo central (radio R, en plano XY)
            centro = np.array([x[0], x[1], 0.0]) * (self.R / r_xy)
        
        # Vector desde ese centro al punto x
        vec = x - centro
        norm_vec = np.linalg.norm(vec)
        if norm_vec == 0:
            vec = np.array([0.0, 0.0, self.a])
        else:
            vec = vec * (self.a / norm_vec)
        
        return centro + vec


def clasificar_impacto_pared(x, contenedor):
    """
    Clasifica el punto de impacto en la pared del contenedor.
    Retorna: 'lateral', 'z_min', 'z_max', 'esfera' o 'caja' según geometría.
    """
    x = np.asarray(x, dtype=float)

    if isinstance(contenedor, ContenedorCilindrico):
        r = np.linalg.norm(x[:2])
        dr = abs(r - contenedor.radio)
        dz_min = abs(x[2] - contenedor.z_min)
        dz_max = abs(x[2] - contenedor.z_max)
        if dr <= min(dz_min, dz_max):
            return "lateral"
        return "z_min" if dz_min < dz_max else "z_max"

    if isinstance(contenedor, ContenedorEsferico):
        return "esfera"

    if isinstance(contenedor, ContenedorCaja):
        lim = contenedor.lim
        dists = {
            "x_min": x[0] + lim[0],
            "x_max": lim[0] - x[0],
            "y_min": x[1] + lim[1],
            "y_max": lim[1] - x[1],
            "z_min": x[2] + lim[2],
            "z_max": lim[2] - x[2],
        }
        return min(dists, key=dists.get)

    if isinstance(contenedor, ContenedorPlacasParalelas):
        dz_min = abs(x[2] - contenedor.z_min)
        dz_max = abs(x[2] - contenedor.z_max)
        return "z_min" if dz_min < dz_max else "z_max"

    if isinstance(contenedor, ContenedorTokamak):
        return "tokamak"

    return "desconocido"