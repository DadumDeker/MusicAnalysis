import subprocess
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from music_analysis.paths import AUDIO_PROCESSED_DIR, AUDIO_RAW_DIR

# ====================== CONFIG ======================
RAW_DIR = AUDIO_RAW_DIR
PROCESSED_DIR = AUDIO_PROCESSED_DIR

SUPPORTED_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
}

TARGET_SR = 48000
FORCE_MONO = True
APPLY_LOUDNORM = False  # ← Set to True only when you need it (slower)
OVERWRITE = False  # Don't re-process files that already exist
# ===================================================


def preprocess_audio(input_path: Path, target_sr: int = TARGET_SR) -> Optional[Path]:
    output_path = PROCESSED_DIR / f"{input_path.stem}.wav"

    if output_path.exists() and not OVERWRITE:
        return output_path

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1" if FORCE_MONO else "0",
        "-ar",
        str(target_sr),
    ]

    if APPLY_LOUDNORM:
        command.extend(["-af", "loudnorm=I=-16:TP=-1:LRA=11"])

    command.append(str(output_path))

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )

    if result.returncode == 0:
        return output_path
    else:
        print(f"❌ Failed: {input_path.name}")
        print(result.stderr.strip()[-600:])
        return None


def main():
    audio_files = [
        f
        for f in RAW_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not audio_files:
        print("No audio files found in audio_raw/")
        return

    print(f"Found {len(audio_files)} files to process\n")

    # Progress bar with ETA
    processed = 0
    start_time = time.time()

    for audio_file in tqdm(audio_files, desc="Converting to WAV", unit="file"):
        result = preprocess_audio(audio_file)
        if result:
            processed += 1

    # Final summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("✅ Preprocessing finished!")
    print(f"   Successfully processed: {processed}/{len(audio_files)} files")
    print(f"   Total time: {elapsed / 60:.1f} minutes")
    print(f"   Output folder: {PROCESSED_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
