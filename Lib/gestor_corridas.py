"""
gestor_corridas.py
==================
Organiza cada simulación en su propia carpeta bajo data/simulaciones/
y mantiene un índice global (index.csv).
"""

import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path


# Raíz del proyecto = padre de Lib/
_PROYECTO = Path(__file__).resolve().parent.parent
BASE_CORRIDAS = _PROYECTO / "data" / "simulaciones"
INDEX_CSV = BASE_CORRIDAS / "index.csv"

INDEX_COLS = [
    "run_id", "fecha", "etiqueta", "motor", "geometria",
    "B0_T", "N_particulas", "pasos", "dt_s", "tau_medio_s",
    "frac_escapadas", "carpeta",
]


def _slug(texto: str, max_len: int = 24) -> str:
    s = re.sub(r"[^\w\-]+", "_", texto.strip().lower())
    return s[:max_len].strip("_") or "run"


def generar_run_id(config: dict) -> str:
    """ID único: fecha_hora + geometría + B0 + N + etiqueta opcional."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    geo = config.get("geometria", "geo")
    b0 = config.get("B0", 0.0)
    n = config.get("n_total", 0)
    etq = _slug(config.get("etiqueta", ""), 16)
    base = f"{ts}_{geo}_B{b0:.3f}_N{n}"
    return f"{base}_{etq}" if etq else base


class GestorCorrida:
    def __init__(self, base: Path = None):
        self.base = Path(base) if base else BASE_CORRIDAS
        self.base.mkdir(parents=True, exist_ok=True)

    def crear_carpeta(self, config: dict) -> Path:
        run_id = generar_run_id(config)
        run_dir = self.base / run_id
        if run_dir.exists():
            run_id = f"{run_id}_dup"
            run_dir = self.base / run_id

        for sub in ("trayectorias", "montecarlo", "figuras", "mapas_calor"):
            (run_dir / sub).mkdir(parents=True)

        fecha = datetime.now().isoformat(timespec="seconds")
        config_guardado = {
            **config,
            "run_id": run_id,
            "fecha": fecha,
            "carpeta": str(run_dir.relative_to(_PROYECTO)).replace("\\", "/"),
        }
        config["fecha"] = fecha
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_guardado, f, indent=2, ensure_ascii=False)

        return run_dir

    def registrar_en_indice(self, config: dict, run_dir: Path, stats: dict = None):
        stats = stats or {}
        row = {
            "run_id": config.get("run_id", run_dir.name),
            "fecha": config.get("fecha", ""),
            "etiqueta": config.get("etiqueta", ""),
            "motor": config.get("motor", "pic"),
            "geometria": config.get("geometria", ""),
            "B0_T": config.get("B0", ""),
            "N_particulas": config.get("n_total", ""),
            "pasos": config.get("pasos", ""),
            "dt_s": config.get("dt", ""),
            "tau_medio_s": stats.get("tau_medio", ""),
            "frac_escapadas": stats.get("fraccion_esc", ""),
            "carpeta": str(run_dir.relative_to(_PROYECTO)).replace("\\", "/"),
        }
        nuevo = not INDEX_CSV.exists()
        with open(INDEX_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=INDEX_COLS)
            if nuevo:
                w.writeheader()
            w.writerow(row)

    @staticmethod
    def listar_corridas(n: int = 20):
        if not INDEX_CSV.exists():
            print("  No hay corridas registradas aún.")
            print(f"  Índice esperado en: {INDEX_CSV}")
            return
        with open(INDEX_CSV, encoding="utf-8") as f:
            filas = list(csv.DictReader(f))
        if not filas:
            print("  Índice vacío.")
            return
        print(f"\n  Últimas {min(n, len(filas))} simulaciones:\n")
        print(f"  {'run_id':<42} {'etiqueta':<12} {'B0':>6} {'N':>4} {'tau_us':>8}")
        print("  " + "-" * 78)
        for row in filas[-n:]:
            tau = row.get("tau_medio_s", "")
            try:
                tau_us = f"{float(tau)*1e6:.2f}" if tau else "—"
            except ValueError:
                tau_us = "—"
            print(
                f"  {row['run_id']:<42} "
                f"{row.get('etiqueta',''):<12} "
                f"{row.get('B0_T',''):>6} "
                f"{row.get('N_particulas',''):>4} "
                f"{tau_us:>8}"
            )
        print(f"\n  Índice completo: {INDEX_CSV}\n")
