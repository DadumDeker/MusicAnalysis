import subprocess
import sys
from datetime import datetime
from pathlib import Path

from music_analysis.paths import (
    AUDIO_PROCESSED_DIR,
    AUDIO_RAW_DIR,
    LOG_DIR,
    REPO_ROOT,
)

# ====================== CONFIG ======================
RAW_DIR = AUDIO_RAW_DIR
PROCESSED_DIR = AUDIO_PROCESSED_DIR

PREPROCESS_SCRIPT = REPO_ROOT / "preprocess.py"
ESSENTIA_SCRIPT = REPO_ROOT / "EssentiaProcess.py"
CLAP_SCRIPT = REPO_ROOT / "CLAPprocess.py"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
ERROR_LOG = LOG_DIR / f"pipeline_error_{TIMESTAMP}.log"
SUMMARY_LOG = LOG_DIR / f"pipeline_summary_{TIMESTAMP}.log"
# ===================================================


def log(message: str, error: bool = False):
    prefix = "❌ ERROR" if error else "✅ INFO"
    print(f"{prefix}: {message}")
    with open(SUMMARY_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
    if error:
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")


def run_script(script_path: Path, stage_name: str) -> bool:
    if not script_path.exists():
        log(f"Script not found: {script_path.name}", error=True)
        return False

    log(f"Starting {stage_name}...")
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            timeout=7200,  # 2 hours max
        )

        if result.stdout.strip():
            print(result.stdout.strip())

        if result.returncode == 0:
            log(f"{stage_name} completed successfully.")
            return True
        else:
            log(f"{stage_name} failed (code {result.returncode})", error=True)
            with open(ERROR_LOG, "a", encoding="utf-8") as f:
                f.write(f"\n--- {stage_name} ERROR ---\n{result.stderr}\n")
            print(result.stderr.strip()[-800:])
            return False
    except Exception as e:
        log(f"Failed to run {stage_name}: {e}", error=True)
        return False


def main():
    print("=" * 80)
    print("🎬 FULL AUDIO ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"Working directory: {REPO_ROOT}\n")

    # 1. Preprocessing (FFmpeg)
    if RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        log("Checking if preprocessing is needed...")
        # Simple check: if any raw file has no matching wav
        needs_pre = any(
            not (PROCESSED_DIR / f"{f.stem}.wav").exists()
            for f in RAW_DIR.iterdir()
            if f.is_file()
        )
        if needs_pre:
            run_script(PREPROCESS_SCRIPT, "FFmpeg Preprocessing")
        else:
            log("⏭️ All files already preprocessed. Skipping FFmpeg.")
    else:
        log("No files in audio_raw/ — skipping preprocessing.")

    print("\n" + "=" * 80 + "\n")

    # 2. Essentia + VGGish
    run_script(ESSENTIA_SCRIPT, "Essentia + VGGish + Deep Models")

    print("\n" + "=" * 80 + "\n")

    # 3. CLAP (Natural Language)
    run_script(CLAP_SCRIPT, "CLAP (Text-Audio Matching)")

    print("\n" + "=" * 80)
    print("🎉 FULL PIPELINE FINISHED!")
    print(f"Logs saved in: {LOG_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
