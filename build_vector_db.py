import pickle
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

from features import Features

BASE_DIR = Path(__file__).parent.resolve()
METADATA_DIR = BASE_DIR / "metadata"
INDEX_DIR = BASE_DIR / "vector_index"
INDEX_DIR.mkdir(exist_ok=True)


def build_index():
    vectors = []
    metadata = []

    json_files = list(METADATA_DIR.glob("*.json"))
    print(f"Found {len(json_files)} tracks. Creating vectors from CLAP matches...\n")

    for json_file in tqdm(json_files):
        try:
            track = Features.load(json_file)

            if not track.clap_matches:
                continue

            # Create a simple vector from CLAP scores
            scores = [m["score"] for m in track.clap_matches]
            vector = np.array(scores, dtype=np.float32)

            # Pad to fixed length (8)
            if len(vector) < 8:
                vector = np.pad(vector, (0, 8 - len(vector)), constant_values=0.0)
            else:
                vector = vector[:8]

            vectors.append(vector)
            metadata.append(track)

        except Exception as e:
            print(f"   Skipped {json_file.name}: {e}")

    if len(vectors) == 0:
        print("❌ No CLAP data found.")
        return

    vectors = np.array(vectors)
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(INDEX_DIR / "tracks_index.faiss"))
    with open(INDEX_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"\n✅ FAISS index built successfully with {len(vectors)} tracks!")
    print(f"   Dimension: {vectors.shape[1]}")
    print(f"   Saved in : {INDEX_DIR}")


if __name__ == "__main__":
    build_index()
