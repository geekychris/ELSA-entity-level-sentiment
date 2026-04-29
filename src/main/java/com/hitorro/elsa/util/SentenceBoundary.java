package com.hitorro.elsa.util;

public record SentenceBoundary(String text, int startOffset, int endOffset) {

    public int length() {
        return endOffset - startOffset;
    }
}
