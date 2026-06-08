from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

AUDIO_RAW_DIR = REPO_ROOT / "audio_raw"
AUDIO_PROCESSED_DIR = REPO_ROOT / "audio_processed"
MODELS_DIR = REPO_ROOT / "models"
VECTOR_INDEX_DIR = REPO_ROOT / "vector_index"
FEATURES_DB = REPO_ROOT / "features.db"
