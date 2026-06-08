import pickle
from pathlib import Path

import faiss
import numpy as np

from features import Features

# ====================== CONFIG ======================
BASE_DIR = Path(__file__).parent.resolve()
INDEX_DIR = BASE_DIR / "vector_index"

# Load index and metadata
index = faiss.read_index(str(INDEX_DIR / "tracks_index.faiss"))
with open(INDEX_DIR / "metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print(f"✅ Loaded {len(metadata)} tracks\n")
# ===================================================


def search(query: str = None, track_name: str = None, top_k: int = 10):
    print(f"\n🔍 Searching for: {query or track_name}\n")

    # For now, we do a simple random / basic search (since we don't have query vector)
    # We'll improve this once the index is stable
    results: list[tuple[float, Features]] = []
    for track in metadata:
        score = np.random.random()  # placeholder

        if query:
            # Very basic text match on CLAP descriptions
            clap_text = " ".join([m["description"] for m in track.clap_matches]).lower()
            if query.lower() in clap_text:
                score += 0.5

        results.append((score, track))

    # Sort by score
    results.sort(key=lambda x: x[0], reverse=True)

    print(f"Top {top_k} results:\n")
    for rank, (score, track) in enumerate(results[:top_k], 1):
        print(f"{rank:2d}. {track.track_name}")
        print(f"    BPM: {track.bpm} | Key: {track.key}{track.scale or ''}")

        if track.clap_matches:
            best = max(track.clap_matches, key=lambda x: x["score"])
            print(f"    CLAP: {best['description']} ({best['score']:.3f})")
        print("-" * 70)


def main():
    print("=== Simple Music Search (Stable Version) ===")
    print(
        "Type a keyword like: psytrance, dark, melodic, festival, night drive, etc.\n"
    )

    while True:
        q = input("\nSearch: ").strip()
        if q.lower() in ["exit", "quit", "q"]:
            break
        if q:
            search(query=q, top_k=10)


if __name__ == "__main__":
    main()
