#!/usr/bin/env bash
# Train all ELSA ML models (requires Python 3.10+ with training dependencies)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== ELSA Model Training ==="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+ first."
    exit 1
fi

PYTHON=python3
TRAINING_DIR="training"
MODELS_DIR="models"

# Set up virtual environment if needed
if [ ! -d "$TRAINING_DIR/venv" ]; then
    echo "[0/4] Creating Python virtual environment..."
    $PYTHON -m venv "$TRAINING_DIR/venv"
fi

echo "[0/4] Activating virtual environment and installing dependencies..."
source "$TRAINING_DIR/venv/bin/activate"
pip install -q -r "$TRAINING_DIR/requirements.txt"

mkdir -p "$MODELS_DIR"

echo ""
echo "[1/4] Training Layer 1: Subjectivity Gate (TF-IDF + LogReg)..."
echo "      Dataset: Rotten Tomatoes subjectivity corpus"
$PYTHON "$TRAINING_DIR/layer1_subjectivity/train_subjectivity_gate.py"
echo "      Output: $MODELS_DIR/subjectivity-gate.onnx, $MODELS_DIR/subjectivity-vocab.tsv"
echo ""

echo "[2/4] Training Layer 2: Sentence Sentiment (DistilBERT fine-tune)..."
echo "      Dataset: SST-2 (Stanford Sentiment Treebank)"
$PYTHON "$TRAINING_DIR/layer2_sentence_sentiment/train_sentence_sentiment.py"
echo "      Output: $MODELS_DIR/sentence-sentiment.onnx, $MODELS_DIR/sentence-sentiment-tokenizer/"
echo ""

echo "[3/4] Training Layer 3: Named Entity Recognition (DistilBERT fine-tune)..."
echo "      Dataset: WikiANN (PanX) English NER"
$PYTHON "$TRAINING_DIR/layer3_ner/train_ner.py"
echo "      Output: $MODELS_DIR/ner.onnx, $MODELS_DIR/ner-tokenizer/"
echo ""

echo "[4/4] Training Layer 4: Entity Sentiment (DistilBERT fine-tune)..."
echo "      Dataset: Synthetic templates"
$PYTHON "$TRAINING_DIR/layer4_entity_sentiment/train_targeted_sentiment.py"
echo "      Output: $MODELS_DIR/entity-sentiment.onnx, $MODELS_DIR/entity-sentiment-tokenizer/"
echo ""

echo "=== Training Complete ==="
echo ""
echo "Model files in $MODELS_DIR/:"
ls -lh "$MODELS_DIR"/*.onnx "$MODELS_DIR"/*.tsv 2>/dev/null || true
echo ""
echo "Tokenizer directories:"
ls -d "$MODELS_DIR"/*-tokenizer/ 2>/dev/null || true
