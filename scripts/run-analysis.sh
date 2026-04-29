#!/usr/bin/env bash
# Run ELSA performance analysis against 24 diverse test inputs
set -euo pipefail
cd "$(dirname "$0")/.."

# Check for models
if [ ! -f "models/subjectivity-gate.onnx" ]; then
    echo "ERROR: No models found in models/."
    echo "Run one of:"
    echo "  ./scripts/train-models.sh          # Train real models"
    echo "  ./scripts/generate-test-models.sh   # Generate synthetic test models"
    exit 1
fi

# Compile if needed
mvn compile -q test-compile -q

# Run
mvn exec:java -q \
    -Dexec.mainClass="com.hitorro.elsa.PerformanceAnalysis" \
    -Dexec.classpathScope="test"
