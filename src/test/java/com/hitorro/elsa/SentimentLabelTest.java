package com.hitorro.elsa;

import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.*;

class SentimentLabelTest {

    @Test
    void fromIndex_standardMapping() {
        assertThat(SentimentLabel.fromIndex(0)).isEqualTo(SentimentLabel.NEGATIVE);
        assertThat(SentimentLabel.fromIndex(1)).isEqualTo(SentimentLabel.NEUTRAL);
        assertThat(SentimentLabel.fromIndex(2)).isEqualTo(SentimentLabel.POSITIVE);
    }

    @Test
    void fromIndex_outOfRange_defaultsToNeutral() {
        assertThat(SentimentLabel.fromIndex(99)).isEqualTo(SentimentLabel.NEUTRAL);
    }

    @Test
    void fromProbabilities_clearPositive() {
        float[] probs = {0.05f, 0.10f, 0.85f}; // neg, neutral, pos
        assertThat(SentimentLabel.fromProbabilities(probs)).isEqualTo(SentimentLabel.POSITIVE);
    }

    @Test
    void fromProbabilities_clearNegative() {
        float[] probs = {0.90f, 0.05f, 0.05f};
        assertThat(SentimentLabel.fromProbabilities(probs)).isEqualTo(SentimentLabel.NEGATIVE);
    }

    @Test
    void fromProbabilities_neutral() {
        float[] probs = {0.10f, 0.80f, 0.10f};
        assertThat(SentimentLabel.fromProbabilities(probs)).isEqualTo(SentimentLabel.NEUTRAL);
    }

    @Test
    void fromProbabilities_mixed_whenTopTwoCloseAndNonNeutral() {
        float[] probs = {0.42f, 0.10f, 0.48f}; // pos and neg are close
        assertThat(SentimentLabel.fromProbabilities(probs)).isEqualTo(SentimentLabel.MIXED);
    }

    @Test
    void fromProbabilities_binary() {
        float[] probs = {0.2f, 0.8f}; // [negative, positive]
        assertThat(SentimentLabel.fromProbabilities(probs)).isEqualTo(SentimentLabel.POSITIVE);
    }
}
