package com.hitorro.elsa.model;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class OnnxInferenceEngineTest {

    @Test
    void softmax_uniformInput() {
        float[] logits = {1.0f, 1.0f, 1.0f};
        float[] probs = OnnxInferenceEngine.softmax(logits);

        assertThat(probs).hasSize(3);
        for (float p : probs) {
            assertThat(p).isCloseTo(1.0f / 3, within(0.001f));
        }
    }

    @Test
    void softmax_dominantClass() {
        float[] logits = {10.0f, 0.0f, 0.0f};
        float[] probs = OnnxInferenceEngine.softmax(logits);

        assertThat(probs[0]).isGreaterThan(0.99f);
        assertThat(probs[1]).isLessThan(0.01f);
        assertThat(probs[2]).isLessThan(0.01f);
    }

    @Test
    void softmax_sumsToOne() {
        float[] logits = {2.5f, -1.0f, 0.5f, 3.0f};
        float[] probs = OnnxInferenceEngine.softmax(logits);

        float sum = 0;
        for (float p : probs) sum += p;
        assertThat(sum).isCloseTo(1.0f, within(0.0001f));
    }

    @Test
    void softmax_negativeLogits() {
        float[] logits = {-5.0f, -2.0f, -1.0f};
        float[] probs = OnnxInferenceEngine.softmax(logits);

        // Should still sum to 1 and be properly ordered
        assertThat(probs[2]).isGreaterThan(probs[1]);
        assertThat(probs[1]).isGreaterThan(probs[0]);

        float sum = 0;
        for (float p : probs) sum += p;
        assertThat(sum).isCloseTo(1.0f, within(0.0001f));
    }
}
