#!/usr/bin/env bash
# Run all ELSA tests
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== ELSA Test Suite ==="

# Check for models (needed by integration tests)
if [ ! -f "models/subjectivity-gate.onnx" ]; then
    echo "WARNING: No models found in models/. Integration tests will use synthetic models."
    echo "Run ./scripts/generate-test-models.sh first, or ./scripts/train-models.sh for real models."
    echo ""
fi

# Compile test sources and run
echo "Running tests..."
mvn test "$@"

echo ""
echo "All tests passed."
