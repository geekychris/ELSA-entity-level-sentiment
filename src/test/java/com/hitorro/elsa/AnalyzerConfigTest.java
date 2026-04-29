package com.hitorro.elsa;

import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.Set;

import static org.assertj.core.api.Assertions.*;

class AnalyzerConfigTest {

    @Test
    void builder_defaults() {
        AnalyzerConfig config = AnalyzerConfig.builder().build();
        assertThat(config.modelDirectory()).isEqualTo(Path.of("models"));
        assertThat(config.subjectivityThreshold()).isEqualTo(0.6);
        assertThat(config.sentimentThreshold()).isEqualTo(0.5);
        assertThat(config.maxTextLength()).isEqualTo(10_000);
        assertThat(config.onnxThreads()).isEqualTo(2);
        assertThat(config.entityTypes()).isEmpty();
    }

    @Test
    void builder_customValues() {
        AnalyzerConfig config = AnalyzerConfig.builder()
                .modelDirectory(Path.of("/custom/models"))
                .subjectivityThreshold(0.8)
                .sentimentThreshold(0.7)
                .maxTextLength(5000)
                .onnxThreads(4)
                .entityTypes(Set.of("PER", "ORG"))
                .build();

        assertThat(config.modelDirectory()).isEqualTo(Path.of("/custom/models"));
        assertThat(config.subjectivityThreshold()).isEqualTo(0.8);
        assertThat(config.sentimentThreshold()).isEqualTo(0.7);
        assertThat(config.maxTextLength()).isEqualTo(5000);
        assertThat(config.onnxThreads()).isEqualTo(4);
        assertThat(config.entityTypes()).containsExactlyInAnyOrder("PER", "ORG");
    }
}
