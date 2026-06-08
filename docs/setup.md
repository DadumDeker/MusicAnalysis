# Setup

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) - Python package manager
- `wget` - used by `pull_models.sh`

## Install Deps

```bash
uv sync
```

## scripts/pull_models.sh

Downloads two Essentia TensorFlow models into `models/`:

```bash
bash scripts/pull_models.sh
```

## scripts/download_samples.py

Downloads public-domain audio from the Internet Archive into `audio_raw/`.

```bash
uv run scripts/download_samples.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--songs`, `-x` | `50` | Total tracks |
| `--genres`, `-y` | `5` | Genres to spread across |
| `--output`, `-o` | `audio_raw` | Output directory |
| `--genre-list` | jazz, classical, blues, folk, electronic, country, ragtime, gospel | Comma-separated override |
| `--workers`, `-w` | half physical cores | Download threads |
| `--min-length` | `30.0` | Min duration (seconds) |
| `--max-length` | `300.0` | Max duration (seconds) |
| `--max-size` | `104857600` | Max file size (bytes) |

## (Optional) scripts/gen_essentia_stubs.py

Generates type stubs for `essentia.standard` into `stubs/`.

```bash
uv run python scripts/gen_essentia_stubs.py
```
