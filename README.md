# MusicAnalysis

## Setup

See [docs/setup.md](docs/setup.md).

# Music Intelligence

Semantic Discovery - Smart Playlists - Audio Intelligence - CLAP + Essentia Pipeline

A full-stack music analysis and discovery system. Extract rich audio features, generate semantic tags with CLAP, build a searchable vector index with FAISS, and explore everything through a shitty, broken, dark-themed Streamlit dashboard.

---

## Features

- **Audio Preprocessing** - FFmpeg normalization to 48kHz mono WAV
- **Rich Feature Extraction** - BPM, Key, Scale, Danceability, Loudness + deep embeddings (EffNet-Discogs, VGGish) via Essentia
- **Semantic Understanding** - CLAP model matches audio to natural language descriptions (techno, dark, energetic, etc.)
- **Vector Search** - FAISS index built from CLAP scores (expandable to full embeddings)
- **Interactive Dashboard** - Filter by BPM/danceability, semantic text search, smart playlist generator, audio playback, and analytics
- **End-to-End Pipeline** - One-command orchestration via run_pipeline.py

---

## Project Structure

```
.
├── audio_raw/              # Drop your original tracks here (.mp3, .wav, .flac, etc.)
├── audio_processed/        # 48kHz mono WAV files (auto-generated)
├── metadata/               # Per-track JSON with features + CLAP tags
├── models/                 # Essentia .pb models (EffNet, VGGish)
├── vector_index/           # FAISS index + metadata pickle
├── logs/                   # Pipeline run logs
├── dashboard.py            # Streamlit UI
├── run_pipeline.py         # Master orchestrator
├── preprocess.py           # FFmpeg conversion
├── EssentiaProcess.py      # Feature + embedding extraction
├── CLAPprocess.py          # CLAP semantic tagging (referenced as add_clap.py in pipeline)
├── build_vector_db.py      # Build FAISS index from CLAP data
├── search_tracks.py        # CLI search tool (currently placeholder)
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install streamlit pandas numpy plotly pillow tqdm

# Heavy dependencies
pip install essentia-tensorflow torch transformers faiss-cpu
```

System requirement: ffmpeg must be installed and available in PATH.

Optional (recommended for speed):
```bash
pip install rapidfuzz
```

### 2. Prepare Models

Place the following Essentia models in the models/ folder:
- discogs-effnet-bs64-1.pb
- audioset-vggish-3.pb

CLAP model (laion/clap-htsat-unfused) will be downloaded automatically on first run.

### 3. Run the Full Pipeline

```bash
python run_pipeline.py
```

This will:
1. Preprocess audio (if needed)
2. Extract Essentia features + embeddings
3. Run CLAP semantic analysis

Then build the vector index:

```bash
python build_vector_db.py
```

### 4. Launch the Dashboard

```bash
streamlit run dashboard.py
```

Open the URL shown in your terminal. Use the sidebar for semantic search, BPM/danceability filters, and generate smart playlists.

---

## Configuration

Most scripts have a clear CONFIG section at the top with paths and toggles:
- preprocess.py: TARGET_SR, FORCE_MONO, APPLY_LOUDNORM, OVERWRITE
- run_pipeline.py: timeout and logging paths
- CLAPprocess.py: candidate description list for tagging

---

## Current Limitations & Known Issues

- Script name mismatch: run_pipeline.py calls add_clap.py but the file is named CLAPprocess.py. Pipeline will fail until fixed.
- Audio path inconsistency: Dashboard looks for audio in audio_raw/, but processed files live in audio_processed/.
- Semantic search is not yet vector-based: dashboard.py and search_tracks.py use fuzzy text matching + random scoring. The FAISS index exists but is not integrated.
- No requirements.txt yet.
- Embeddings functionality note: I was unable to get the deep embeddings (EffNet, VGGish, and full CLAP audio embeddings) to function reliably in the current setup. The pipeline currently relies primarily on CLAP text-match scores and basic audio features (BPM, danceability, key).
- Broad exception handling in several places makes debugging harder.
- CLAP embedding extraction contains fragile hasattr logic due to transformers output variations.

---

## Roadmap & Future Plans

We are actively improving the system. Next major milestone:

Custom K-pop Weights for Inference
We will begin creating and fine-tuning our own model weights using a curated dataset of K-pop songs. This will significantly improve semantic understanding, mood detection, and recommendation quality for K-pop and related genres.

Other planned improvements:
- Integrate real FAISS / embedding-based semantic search into the dashboard
- Add support for user-uploaded tracks and on-the-fly analysis
- Export smart playlists to Spotify/JSON/M3U
- Fine-grained genre/mood taxonomy trained on K-pop
- Web UI improvements and mobile responsiveness
