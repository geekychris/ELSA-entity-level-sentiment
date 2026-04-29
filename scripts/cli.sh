#!/usr/bin/env bash
# Launch the ELSA interactive CLI
# Usage: ./scripts/cli.sh [model-dir]
set -euo pipefail
cd "$(dirname "$0")/.."

# Check for models
if [ ! -f "models/subjectivity-gate.onnx" ]; then
    echo "ERROR: No models found in models/."
    echo "Run ./scripts/train-models.sh or ./scripts/generate-test-models.sh first."
    exit 1
fi

# Compile if needed
mvn compile -q 2>/dev/null

# Run CLI (pass model dir as arg if provided)
mvn exec:java -q -Dexec.mainClass="com.hitorro.elsa.cli.ElsaCli" \
    ${1:+-Dexec.args="$1"}
