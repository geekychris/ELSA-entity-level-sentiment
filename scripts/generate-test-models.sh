#!/usr/bin/env bash
# Generate synthetic ONNX models for testing (no GPU/training required)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Generate Synthetic Test Models ==="
echo ""
echo "These models are structurally valid but semantically random."
echo "Use for CI/CD and integration testing only."
echo ""

PYTHON=python3
TRAINING_DIR="training"

# Set up virtual environment if needed
if [ ! -d "$TRAINING_DIR/venv" ]; then
    echo "Creating Python virtual environment..."
    $PYTHON -m venv "$TRAINING_DIR/venv"
fi

source "$TRAINING_DIR/venv/bin/activate"
pip install -q -r "$TRAINING_DIR/requirements.txt"

echo "Generating models..."
$PYTHON "$TRAINING_DIR/create_test_models.py"

echo ""
echo "Synthetic models generated in models/:"
ls -lh models/*.onnx 2>/dev/null || true
