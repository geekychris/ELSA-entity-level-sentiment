package com.hitorro.elsa;

import java.nio.file.Path;
import java.util.Set;

public record AnalyzerConfig(
        Path modelDirectory,
        double subjectivityThreshold,
        double sentimentThreshold,
        int maxTextLength,
        int onnxThreads,
        Set<String> entityTypes
) {
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private Path modelDirectory = Path.of("models");
        private double subjectivityThreshold = 0.01;
        private double sentimentThreshold = 0.5;
        private int maxTextLength = 10_000;
        private int onnxThreads = 2;
        private Set<String> entityTypes = Set.of(); // empty = all types

        public Builder modelDirectory(Path p) { this.modelDirectory = p; return this; }
        public Builder subjectivityThreshold(double t) { this.subjectivityThreshold = t; return this; }
        public Builder sentimentThreshold(double t) { this.sentimentThreshold = t; return this; }
        public Builder maxTextLength(int m) { this.maxTextLength = m; return this; }
        public Builder onnxThreads(int t) { this.onnxThreads = t; return this; }
        public Builder entityTypes(Set<String> types) { this.entityTypes = types; return this; }

        public AnalyzerConfig build() {
            return new AnalyzerConfig(modelDirectory, subjectivityThreshold, sentimentThreshold,
                    maxTextLength, onnxThreads, entityTypes);
        }
    }
}
