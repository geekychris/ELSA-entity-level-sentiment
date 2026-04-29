package com.hitorro.elsa;

public enum SentimentLabel {
    POSITIVE,
    NEGATIVE,
    NEUTRAL,
    MIXED;

    public static SentimentLabel fromIndex(int index) {
        return switch (index) {
            case 0 -> NEGATIVE;
            case 1 -> NEUTRAL;
            case 2 -> POSITIVE;
            default -> NEUTRAL;
        };
    }

    public static SentimentLabel fromProbabilities(float[] probs) {
        if (probs.length < 3) {
            // Binary: [negative, positive]
            return probs[1] > probs[0] ? POSITIVE : NEGATIVE;
        }
        int maxIdx = 0;
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > probs[maxIdx]) maxIdx = i;
        }
        float maxProb = probs[maxIdx];
        // Check for mixed: if top two are close and both non-neutral
        if (probs.length >= 3) {
            float[] sorted = probs.clone();
            java.util.Arrays.sort(sorted);
            float second = sorted[sorted.length - 2];
            if (maxProb - second < 0.15f && maxIdx != 1 && indexOfValue(probs, second) != 1) {
                return MIXED;
            }
        }
        return fromIndex(maxIdx);
    }

    private static int indexOfValue(float[] arr, float val) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == val) return i;
        }
        return -1;
    }
}
