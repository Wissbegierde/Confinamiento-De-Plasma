"""
test_colisiones.py
==================
Pruebas de verificación para el módulo colisiones.py.

Ejecutar desde la carpeta Lib:
    python test_colisiones.py
"""
import numpy as np
from colisiones import velocidad_inicial_mb, ColisionEstocastica

kB = 1.380649e-23
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

# ─────────────────────────────────────────────────────────────
# TEST 1: velocidad_inicial_mb genera vectores 3D
# ─────────────────────────────────────────────────────────────
v = velocidad_inicial_mb(m=1e-9, T=1e4)
ok = v.shape == (3,)
print(f"[{'PASS' if ok else 'FAIL'}] velocidad_inicial_mb devuelve vector 3D: shape={v.shape}")

# ─────────────────────────────────────────────────────────────
# TEST 2: La media de |v| coincide con el valor teórico de MB
#         <v> = sqrt(8 kB T / pi m)
# ─────────────────────────────────────────────────────────────
m, T = 1e-9, 1e4
N = 100_000
rng = np.random.default_rng(0)
muestras = np.array([np.linalg.norm(velocidad_inicial_mb(m, T, rng=rng)) for _ in range(N)])

v_mean_teorico  = np.sqrt(8 * kB * T / (np.pi * m))
v_mean_numerico = muestras.mean()
error_rel = abs(v_mean_numerico - v_mean_teorico) / v_mean_teorico

ok = error_rel < 0.01   # menos del 1%
print(f"[{'PASS' if ok else 'FAIL'}] <|v|> MB: teórico={v_mean_teorico:.4e}  numérico={v_mean_numerico:.4e}  error={error_rel*100:.3f}%")

# ─────────────────────────────────────────────────────────────
# TEST 3: Dirección isotrópica (media de cada componente ≈ 0)
# ─────────────────────────────────────────────────────────────
vecs = np.array([velocidad_inicial_mb(m, T, rng=rng) for _ in range(N)])
medias = vecs.mean(axis=0)
ok = np.all(np.abs(medias) < 0.01 * v_mean_teorico)
print(f"[{'PASS' if ok else 'FAIL'}] Componentes isotrópicas: <vx>={medias[0]:.2e}  <vy>={medias[1]:.2e}  <vz>={medias[2]:.2e}")

# ─────────────────────────────────────────────────────────────
# TEST 4: ColisionEstocastica — probabilidad empírica vs teórica
#         P = 1 - exp(-ν·dt)
# ─────────────────────────────────────────────────────────────
nu, dt = 500.0, 5e-5
p_teorica = 1.0 - np.exp(-nu * dt)
col = ColisionEstocastica(nu=nu, m=m, T=T, dt=dt, seed=42)
v_test = np.zeros(3)
for _ in range(200_000):
    v_test, _ = col.aplicar(v_test)

error_p = abs(col.tasa_colision_real - p_teorica)
ok = error_p < 0.002
print(f"[{'PASS' if ok else 'FAIL'}] P_colision: teórica={p_teorica:.5f}  empírica={col.tasa_colision_real:.5f}  Δ={error_p:.5f}")

# ─────────────────────────────────────────────────────────────
# TEST 5: Después de colisión, la velocidad cambia
# ─────────────────────────────────────────────────────────────
col2 = ColisionEstocastica(nu=1e9, m=m, T=T, dt=1.0, seed=0)  # P≈1, siempre colisiona
v_antes = np.array([0.0, 0.0, 0.0])
v_despues, colisiono = col2.aplicar(v_antes)
ok = colisiono and np.linalg.norm(v_despues) > 0
print(f"[{'PASS' if ok else 'FAIL'}] Tras colisión la velocidad cambia: |v_nueva|={np.linalg.norm(v_despues):.4e} m/s")

# ─────────────────────────────────────────────────────────────
# TEST 6: Sin colisión, la velocidad no cambia
# ─────────────────────────────────────────────────────────────
col3 = ColisionEstocastica(nu=0.0, m=m, T=T, dt=dt, seed=0)   # P=0, nunca colisiona
v_orig = np.array([1.0, 2.0, 3.0])
v_out, colisiono = col3.aplicar(v_orig)
ok = not colisiono and np.allclose(v_orig, v_out)
print(f"[{'PASS' if ok else 'FAIL'}] Sin colisión la velocidad se mantiene: colisionó={colisiono}")

print("\n" + col.resumen())
