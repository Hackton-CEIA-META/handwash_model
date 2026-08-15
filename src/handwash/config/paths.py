"""Caminhos centrais do projeto - unica fonte da verdade sobre onde tudo mora."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Dataset bruto do Kaggle - vive fora do repo (nao duplicamos 1.3GB de video em git)
RAW_DATASET_DIR = Path(r"D:\Hackton Meta\HandWashDataset")

# Bronze: metadados estruturados do dado bruto (nao os bytes do video)
BRONZE_DIR = REPO_ROOT / "data" / "bronze"
BRONZE_MANIFEST = BRONZE_DIR / "manifest.csv"
BRONZE_PROBE_REPORT = BRONZE_DIR / "probe_report.csv"

# Silver: landmarks extraidos por video, limpos/estruturados
SILVER_DIR = REPO_ROOT / "data" / "silver"
SILVER_LANDMARKS_DIR = SILVER_DIR / "landmarks"
SILVER_MANIFEST = SILVER_DIR / "manifest.csv"

# Gold: janelas normalizadas, prontas pra treino (Dia 2)
GOLD_DIR = REPO_ROOT / "data" / "gold"
GOLD_CLASSES_JSON = GOLD_DIR / "classes.json"

# Modelos
MODELS_DIR = REPO_ROOT / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
MEDIAPIPE_MODEL_DIR = MODELS_DIR / "mediapipe"
HAND_LANDMARKER_TASK = MEDIAPIPE_MODEL_DIR / "hand_landmarker.task"
