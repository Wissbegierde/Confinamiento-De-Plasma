"""
interacciones.py
================
ResolvedorPoisson con factorización LU precalculada.

Flujo correcto:
  1. definir_frontera_vectorizada()  — una vez
  2. precalcular_matriz()            — una vez  (~12s para grilla 30³)
  3. precalcular_campos_externos()   — una vez
  4. depositar_carga_cic()  +  resolver_con_matriz()  — cada paso (~60ms)
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import factorized   # ← LU una sola vez
from joblib import Parallel, delayed


# ══════════════════════════════════════════════════════════════
#  COULOMB DIRECTO (debug / N muy pequeño)
# ══════════════════════════════════════════════════════════════

def calcular_E_interaccion(p_target, p_source):
    ke    = 8.987e9
    r_vec = p_target.x - p_source.x
    dist  = np.linalg.norm(r_vec)
    if dist < 1e-10:
        return np.zeros(3)
    return ke * p_source.q * r_vec / dist**3


# ══════════════════════════════════════════════════════════════
#  RESOLVEDOR DE POISSON
# ══════════════════════════════════════════════════════════════

class ResolvedorPoisson:
    def __init__(self, dimensiones, resolucion, epsilon_0=8.854e-12):
        self.nx, self.ny, self.nz = resolucion
        self.Lx, self.Ly, self.Lz = dimensiones
        self.dx = self.Lx / (self.nx - 1)
        self.dy = self.Ly / (self.ny - 1)
        self.dz = self.Lz / (self.nz - 1)
        self.eps0 = epsilon_0
        self.N    = self.nx * self.ny * self.nz

        self.phi        = np.zeros((self.nx, self.ny, self.nz))
        self.rho        = np.zeros((self.nx, self.ny, self.nz))
        self.mask_libre = np.ones((self.nx, self.ny, self.nz), dtype=bool)

        # Se llenan en precalcular_matriz()
        self._solver      = None   # callable: b → phi  (sustitución LU)
        self._libre_flat  = None   # mask aplanada (cacheada para resolver rápido)
        self._phi_flat0   = None   # valores Dirichlet aplanados

        # Campos externos precalculados
        self.E_ext_grilla = None
        self.B_ext_grilla = None

    # ─────────────────────────────────────────────────────────
    def definir_frontera_vectorizada(self, funcion_frontera):
        xs = np.arange(self.nx) * self.dx
        ys = np.arange(self.ny) * self.dy
        zs = np.arange(self.nz) * self.dz
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        self.mask_libre = ~funcion_frontera(X, Y, Z)
        self.phi[~self.mask_libre] = 0.0
        self._X, self._Y, self._Z = X, Y, Z

    # ─────────────────────────────────────────────────────────
    def precalcular_matriz(self):
        """
        Construye A vectorizado y la factoriza con LU (SuperLU).
        La factorización tarda ~10-15s para grilla 30³,
        pero cada solve posterior cuesta solo ~60ms.
        """
        print(f"  [Poisson] Construyendo A ({self.N} nodos, vectorizado)...",
              flush=True)

        nx, ny, nz = self.nx, self.ny, self.nz
        N  = self.N
        cx = 1.0 / self.dx**2
        cy = 1.0 / self.dy**2
        cz = 1.0 / self.dz**2
        cdiag = 2.0 * (cx + cy + cz)

        # Índices planos de todos los nodos
        I, J, K = np.meshgrid(np.arange(nx), np.arange(ny),
                               np.arange(nz), indexing='ij')
        I = I.ravel(); J = J.ravel(); K = K.ravel()
        n_idx = I * ny * nz + J * nz + K
        libre = self.mask_libre.ravel()

        rows = [n_idx]
        cols = [n_idx]
        data = [np.where(libre, -cdiag, 1.0)]

        for di, dj, dk, c in [
            ( 1,0,0,cx),(-1,0,0,cx),
            (0, 1,0,cy),(0,-1,0,cy),
            (0,0, 1,cz),(0,0,-1,cz),
        ]:
            ni = I+di; nj = J+dj; nk = K+dk
            ok = libre & (ni>=0)&(ni<nx) & (nj>=0)&(nj<ny) & (nk>=0)&(nk<nz)
            rows.append(n_idx[ok])
            cols.append(ni[ok]*ny*nz + nj[ok]*nz + nk[ok])
            data.append(np.full(ok.sum(), c))

        A = csr_matrix(
            (np.concatenate(data),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(N, N)
        )
        print("  [Poisson] A lista. Factorizando LU (una sola vez)...", flush=True)

        # ── FACTORIZACIÓN LU ──────────────────────────────────
        # Devuelve un callable solve(b) → x, sin rehacer LU
        self._solver = factorized(A)

        # Precachear máscara y valores Dirichlet aplanados
        self._libre_flat = libre
        self._phi_flat0  = self.phi.ravel().copy()   # potencial en paredes

        print("  [Poisson] LU lista. Cada solve costará ~60ms.", flush=True)

    # ─────────────────────────────────────────────────────────
    def precalcular_campos_externos(self, fn_E, fn_B, offset):
        """Evalúa fn_E y fn_B en toda la grilla en paralelo (joblib)."""
        print("  [Poisson] Precalculando campos externos...", flush=True)
        nx, ny, nz = self.nx, self.ny, self.nz

        X_f = self._X - offset[0]
        Y_f = self._Y - offset[1]
        Z_f = self._Z - offset[2]

        def _fila(fn, i):
            out = np.zeros((ny, nz, 3))
            for j in range(ny):
                for k in range(nz):
                    out[j,k] = fn(np.array([X_f[i,j,k], Y_f[i,j,k], Z_f[i,j,k]]))
            return i, out

        self.E_ext_grilla = np.zeros((nx, ny, nz, 3))
        self.B_ext_grilla = np.zeros((nx, ny, nz, 3))

        for i, fila in Parallel(n_jobs=-1)(delayed(_fila)(fn_E, i) for i in range(nx)):
            self.E_ext_grilla[i] = fila
        for i, fila in Parallel(n_jobs=-1)(delayed(_fila)(fn_B, i) for i in range(nx)):
            self.B_ext_grilla[i] = fila

        print("  [Poisson] Campos externos listos.", flush=True)

    def obtener_E_externo(self, pos_grilla):
        i = int(np.clip(pos_grilla[0]/self.dx, 0, self.nx-2))
        j = int(np.clip(pos_grilla[1]/self.dy, 0, self.ny-2))
        k = int(np.clip(pos_grilla[2]/self.dz, 0, self.nz-2))
        return self.E_ext_grilla[i, j, k]

    def obtener_B_externo(self, pos_grilla):
        i = int(np.clip(pos_grilla[0]/self.dx, 0, self.nx-2))
        j = int(np.clip(pos_grilla[1]/self.dy, 0, self.ny-2))
        k = int(np.clip(pos_grilla[2]/self.dz, 0, self.nz-2))
        return self.B_ext_grilla[i, j, k]

    # ─────────────────────────────────────────────────────────
    def depositar_carga_cic(self, particulas):
        """Cloud-In-Cell trilineal."""
        self.rho.fill(0)
        vol = self.dx * self.dy * self.dz
        for p in particulas:
            fi = p.x[0]/self.dx; fj = p.x[1]/self.dy; fk = p.x[2]/self.dz
            i0 = int(fi); i1 = min(i0+1, self.nx-1)
            j0 = int(fj); j1 = min(j0+1, self.ny-1)
            k0 = int(fk); k1 = min(k0+1, self.nz-1)
            wx = fi-i0; wy = fj-j0; wz = fk-k0
            self.rho[i0,j0,k0] += p.q*(1-wx)*(1-wy)*(1-wz)/vol
            self.rho[i1,j0,k0] += p.q*   wx *(1-wy)*(1-wz)/vol
            self.rho[i0,j1,k0] += p.q*(1-wx)*   wy *(1-wz)/vol
            self.rho[i0,j0,k1] += p.q*(1-wx)*(1-wy)*   wz /vol
            self.rho[i1,j1,k0] += p.q*   wx *   wy *(1-wz)/vol
            self.rho[i1,j0,k1] += p.q*   wx *(1-wy)*   wz /vol
            self.rho[i0,j1,k1] += p.q*(1-wx)*   wy *   wz /vol
            self.rho[i1,j1,k1] += p.q*   wx *   wy *   wz /vol

    # ─────────────────────────────────────────────────────────
    def resolver_con_matriz(self):
        """
        Arma b (vectorizado) y resuelve usando la LU precalculada.
        Costo real: solo sustitución triangular ~ 60ms para grilla 30³.
        """
        if self._solver is None:
            raise RuntimeError("Llama a precalcular_matriz() primero.")

        # b vectorizado — sin ningún loop Python
        b = np.where(
            self._libre_flat,
            -self.rho.ravel() / self.eps0,
            self._phi_flat0          # Dirichlet: phi en paredes
        )

        self.phi = self._solver(b).reshape((self.nx, self.ny, self.nz))

    # ─────────────────────────────────────────────────────────
    def obtener_E_colectivo(self, pos):
        """Gradiente centrado O(h²)."""
        i = int(np.clip(pos[0]/self.dx, 1, self.nx-2))
        j = int(np.clip(pos[1]/self.dy, 1, self.ny-2))
        k = int(np.clip(pos[2]/self.dz, 1, self.nz-2))
        Ex = -(self.phi[i+1,j,k] - self.phi[i-1,j,k]) / (2*self.dx)
        Ey = -(self.phi[i,j+1,k] - self.phi[i,j-1,k]) / (2*self.dy)
        Ez = -(self.phi[i,j,k+1] - self.phi[i,j,k-1]) / (2*self.dz)
        return np.array([Ex, Ey, Ez])

    # ─────────────────────────────────────────────────────────
    def construir_y_resolver(self):
        """Compatibilidad con código anterior."""
        self.precalcular_matriz()
        self.resolver_con_matriz()
