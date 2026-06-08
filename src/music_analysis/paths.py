"""Common local paths."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

AUDIO_RAW_DIR = REPO_ROOT / "audio_raw"
AUDIO_PROCESSED_DIR = REPO_ROOT / "audio_processed"
MODELS_DIR = REPO_ROOT / "models"
VECTOR_INDEX_DIR = REPO_ROOT / "vector_index"
METADATA_DIR = REPO_ROOT / "metadata"
LOG_DIR = REPO_ROOT / "logs"
FEATURES_DB = REPO_ROOT / "features.db"

for _d in (
    AUDIO_RAW_DIR,
    AUDIO_PROCESSED_DIR,
    MODELS_DIR,
    VECTOR_INDEX_DIR,
    METADATA_DIR,
    LOG_DIR,
):
    _d.mkdir(exist_ok=True)
