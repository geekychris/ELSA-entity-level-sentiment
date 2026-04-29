#!/usr/bin/env bash
# Build the ELSA Java project
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== ELSA Build ==="
echo "Java version: $(java -version 2>&1 | head -1)"
echo "Maven version: $(mvn --version 2>&1 | head -1)"
echo ""

echo "[1/3] Cleaning previous build..."
mvn clean -q

echo "[2/3] Compiling sources..."
mvn compile -q

echo "[3/3] Packaging JAR (skipping tests)..."
mvn package -DskipTests -q

JAR=$(ls target/elsa-*.jar 2>/dev/null | head -1)
if [ -n "$JAR" ]; then
    echo ""
    echo "Build successful: $JAR ($(du -h "$JAR" | cut -f1))"
else
    echo "Build failed: no JAR produced"
    exit 1
fi
