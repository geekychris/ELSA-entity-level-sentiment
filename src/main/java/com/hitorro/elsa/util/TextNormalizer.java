package com.hitorro.elsa.util;

import java.text.Normalizer;

public final class TextNormalizer {

    private TextNormalizer() {}

    public static String normalize(String text) {
        if (text == null || text.isEmpty()) return "";

        // Unicode NFC normalization
        String normalized = Normalizer.normalize(text, Normalizer.Form.NFC);

        // Collapse whitespace runs into single spaces
        normalized = normalized.replaceAll("\\s+", " ");

        return normalized.strip();
    }

    public static String truncate(String text, int maxLength) {
        if (text.length() <= maxLength) return text;
        return text.substring(0, maxLength);
    }
}
