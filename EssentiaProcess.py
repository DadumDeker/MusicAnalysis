import json
from pathlib import Path

import essentia.standard as es
from essentia import Pool

# ====================== CONFIG ======================
BASE_DIR = Path(__file__).parent.resolve()

AUDIO_DIR = BASE_DIR / "audio_processed"
METADATA_DIR = BASE_DIR / "metadata"
MODELS_DIR = BASE_DIR / "models"

METADATA_DIR.mkdir(exist_ok=True)

print(f"Models folder: {MODELS_DIR.resolve()}\n")
# ===================================================


# TODO (joshua-dean1_ecolab): return typed dict or dataclass
def extract_features(audio_path: Path) -> dict:
    print(f"\nAnalyzing: {audio_path.name}")

    audio_48k = es.MonoLoader(filename=str(audio_path), sampleRate=48000)()
    audio_16k = es.MonoLoader(filename=str(audio_path), sampleRate=16000)()

    features = {"track_name": audio_path.stem}

    # === Basic Features ===
    try:
        bpm, _, _, _, _ = es.RhythmExtractor2013(method="multifeature")(audio_48k)
        key, scale, key_strength = es.KeyExtractor()(audio_48k)

        features.update(
            {
                "bpm": round(float(bpm), 2),
                "key": key,
                "scale": scale,
                "key_strength": round(float(key_strength), 4),
                "loudness": round(float(es.Loudness()(audio_48k)), 4),
                "danceability": round(float(es.Danceability()(audio_48k)[0]), 4),
            }
        )
    except Exception as e:
        print(f"   Basic features error: {e}")

    # === Deep Embeddings ===
    pool = Pool()

    # 1. EffNet-Discogs (Best one)
    effnet_path = MODELS_DIR / "discogs-effnet-bs64-1.pb"
    if effnet_path.exists():
        try:
            effnet = es.TensorflowPredictEffnetDiscogs(graphFilename=str(effnet_path))
            effnet_emb = effnet(audio_16k)
            features["effnet_embedding"] = effnet_emb.mean(axis=0).tolist()
            pool.set("embeddings", effnet_emb)
            print(
                f"   ✅ EffNet embedding saved ({len(features['effnet_embedding'])} dims)"
            )
        except Exception as e:
            print(f"   ❌ EffNet failed: {e}")
    else:
        print(f"   ⚠️ EffNet model not found: {effnet_path.name}")

    # 2. VGGish Embeddings
    vggish_path = MODELS_DIR / "audioset-vggish-3.pb"
    if vggish_path.exists():
        try:
            vggish = es.TensorflowPredictVGGish(
                graphFilename=str(vggish_path), output="model/vggish/embeddings"
            )
            vggish_emb = vggish(audio_16k)
            features["vggish_embedding"] = vggish_emb.mean(axis=0).tolist()
            print("   ✅ VGGish embedding saved")
        except Exception as e:
            print(f"   ❌ VGGish failed: {e}")

    return features


def save_features(features: dict):
    output_path = METADATA_DIR / f"{features['track_name']}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)
    print(f"✅ Saved: {output_path.name}")


def main():
    audio_files = sorted(AUDIO_DIR.glob("*.wav"))
    print(f"Found {len(audio_files)} .wav files\n")

    for audio_file in audio_files:
        try:
            features = extract_features(audio_file)
            save_features(features)
        except Exception as e:
            print(f"❌ ERROR on {audio_file.name}: {e}")

    print(f"\n🎉 Finished processing {len(audio_files)} tracks!")


if __name__ == "__main__":
    main()
