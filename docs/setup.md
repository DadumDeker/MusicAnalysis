# Setup

```bash
uv sync
wget  # required for pull_models.sh
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
