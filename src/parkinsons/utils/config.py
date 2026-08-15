from pathlib import Path

import scienceplots  # noqa: F401
from matplotlib import pyplot as plt

PROJ_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJ_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJ_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = TABLES_DIR
STATIC_DIR = Path(__file__).resolve().parent.parent / "api"

for directory in [FIGURES_DIR, TABLES_DIR, MODELS_DIR, STATIC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

plt.style.use("science")
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.edgecolor"] = "black"
plt.rcParams["legend.loc"] = "best"
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.5
plt.rcParams["grid.linestyle"] = "-"
plt.rcParams["grid.linewidth"] = 0.5
