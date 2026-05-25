#!/bin/bash
# Pull Essentia models

set -x
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
REPO_DIR=$(dirname $SCRIPT_DIR)
MODELS_DIR="$REPO_DIR/models"

wget -P $MODELS_DIR https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb
wget -P $MODELS_DIR https://essentia.upf.edu/models/feature-extractors/vggish/audioset-vggish-3.pb