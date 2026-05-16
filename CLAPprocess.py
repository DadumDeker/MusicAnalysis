from pathlib import Path
import json
import torch
import essentia.standard as es
from transformers import AutoProcessor, AutoModel

# ====================== CONFIG ======================
BASE_DIR = Path(__file__).parent.resolve()
METADATA_DIR = BASE_DIR / "metadata"
AUDIO_DIR = BASE_DIR / "audio_processed"

MODEL_NAME = "laion/clap-htsat-unfused"
# ===================================================

print("Loading CLAP model...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def get_clap_matches(audio_path: Path, top_k: int = 8):
    # Load audio
    audio = es.MonoLoader(filename=str(audio_path), sampleRate=48000)()
    
    # Audio embedding
    audio_input = processor(audio=audio, return_tensors="pt", sampling_rate=48000)
    with torch.no_grad():
        audio_output = model.get_audio_features(**audio_input)
        
        # Extract the actual tensor (this is the fix)
        if hasattr(audio_output, 'audio_embeds'):
            audio_emb = audio_output.audio_embeds
        elif hasattr(audio_output, 'pooler_output'):
            audio_emb = audio_output.pooler_output
        elif torch.is_tensor(audio_output):
            audio_emb = audio_output
        else:
            audio_emb = audio_output[0] if isinstance(audio_output, (list, tuple)) else audio_output

    # Text embedding
    candidates = [
        "energetic electronic dance music", "deep house", "techno", "melodic techno",
        "progressive house", "uplifting trance", "festival mainstage banger",
        "dark techno", "vocal house", "bass house", "chill ambient", "90s rave",
        "psychedelic trance", "goa trance", "full-on psytrance", "high energy EDM"
    ]

    text_input = processor(text=candidates, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_output = model.get_text_features(**text_input)
        
        if hasattr(text_output, 'text_embeds'):
            text_emb = text_output.text_embeds
        elif hasattr(text_output, 'pooler_output'):
            text_emb = text_output.pooler_output
        elif torch.is_tensor(text_output):
            text_emb = text_output
        else:
            text_emb = text_output[0] if isinstance(text_output, (list, tuple)) else text_output

    # Cosine similarity
    similarities = torch.nn.functional.cosine_similarity(audio_emb, text_emb, dim=1)
    top_k_idx = torch.topk(similarities, k=top_k).indices

    return [
        {"description": candidates[i], "score": round(float(similarities[i]), 4)}
        for i in top_k_idx
    ]


def main():
    json_files = list(METADATA_DIR.glob("*.json"))
    print(f"Found {len(json_files)} tracks to enrich with CLAP\n")

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            wav_path = AUDIO_DIR / f"{data['track_name']}.wav"
            if not wav_path.exists():
                print(f"⚠️ Missing wav: {wav_path.name}")
                continue

            print(f"CLAP analyzing: {data['track_name'][:70]}...")
            clap_results = get_clap_matches(wav_path)

            data["clap_matches"] = clap_results

            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            print(f"   ✓ Added {len(clap_results)} descriptions\n")

        except Exception as e:
            print(f"❌ Error on {json_file.name}: {e}")

    print("🎉 CLAP processing completed!")


if __name__ == "__main__":
    main()