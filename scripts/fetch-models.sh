#!/usr/bin/env bash
# Fetch the canonical ONNX models from the private Hugging Face Hub repo.
# Requires:
#   - HF_TOKEN environment variable, OR
#   - prior `huggingface-cli login`
# Access to the repo must be granted by the repo owner.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="chrisxyz11/elsa-models"
DEST="models"
FILES=(
    "entity-sentiment.onnx"
    "ner.onnx"
    "sentence-sentiment.onnx"
    "subjectivity-gate.onnx"
)

echo "=== ELSA model fetch ==="
echo "Source: huggingface.co/${REPO} (private)"
echo "Target: ${DEST}/"
echo ""

if ! command -v python3 >/dev/null; then
    echo "ERROR: python3 is required."
    exit 1
fi

if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface_hub..."
    python3 -m pip install --quiet "huggingface_hub>=0.24"
fi

if [ -z "${HF_TOKEN:-}" ] && [ ! -f "${HOME}/.cache/huggingface/token" ]; then
    cat <<'EOF'
ERROR: No Hugging Face credentials found.

Set HF_TOKEN with a read token from https://huggingface.co/settings/tokens, e.g.:
    export HF_TOKEN=hf_xxxxxxxxxxxx
or run:
    huggingface-cli login

The repo is private; the owner must grant your account read access.
EOF
    exit 1
fi

mkdir -p "${DEST}"

python3 - "${REPO}" "${DEST}" "${FILES[@]}" <<'PY'
import os, sys
from huggingface_hub import hf_hub_download

repo, dest, *files = sys.argv[1:]
token = os.environ.get("HF_TOKEN")
for fname in files:
    print(f"  fetching {fname}...", flush=True)
    path = hf_hub_download(
        repo_id=repo,
        filename=fname,
        local_dir=dest,
        token=token,
    )
    print(f"    -> {path}")
print("All models downloaded.")
PY

echo ""
echo "Done. Models in ${DEST}/"
ls -lh "${DEST}"/*.onnx
