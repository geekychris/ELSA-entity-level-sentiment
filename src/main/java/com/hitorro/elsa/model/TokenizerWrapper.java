package com.hitorro.elsa.model;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.huggingface.tokenizers.jni.CharSpan;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

/**
 * Wraps a HuggingFace tokenizer (Rust-backed via DJL) for fast tokenization.
 * Provides single and batch encoding with padding and truncation.
 */
public class TokenizerWrapper implements AutoCloseable {

    private final HuggingFaceTokenizer tokenizer;
    private final int maxLength;

    public TokenizerWrapper(Path tokenizerPath, int maxLength) throws IOException {
        this.maxLength = maxLength;
        this.tokenizer = HuggingFaceTokenizer.newInstance(tokenizerPath,
                Map.of("padding", "true",
                       "truncation", "true",
                       "maxLength", String.valueOf(maxLength)));
    }

    public TokenizerWrapper(Path tokenizerPath) throws IOException {
        this(tokenizerPath, 512);
    }

    public EncodingResult encode(String text) {
        Encoding encoding = tokenizer.encode(text);
        return new EncodingResult(
                encoding.getIds(),
                encoding.getAttentionMask(),
                charSpansToOffsets(encoding.getCharTokenSpans())
        );
    }

    public BatchEncodingResult encodeBatch(String[] texts) {
        Encoding[] encodings = tokenizer.batchEncode(texts);
        int batchSize = encodings.length;
        int seqLen = encodings[0].getIds().length;

        long[][] inputIds = new long[batchSize][seqLen];
        long[][] attentionMask = new long[batchSize][seqLen];
        long[][][] offsets = new long[batchSize][][];

        for (int i = 0; i < batchSize; i++) {
            inputIds[i] = encodings[i].getIds();
            attentionMask[i] = encodings[i].getAttentionMask();
            offsets[i] = charSpansToOffsets(encodings[i].getCharTokenSpans());
        }
        return new BatchEncodingResult(inputIds, attentionMask, offsets);
    }

    /**
     * Convert DJL CharSpan[] to long[][] offset pairs [[start, end], ...].
     */
    private static long[][] charSpansToOffsets(CharSpan[] spans) {
        if (spans == null) return new long[0][];
        long[][] offsets = new long[spans.length][2];
        for (int i = 0; i < spans.length; i++) {
            if (spans[i] != null) {
                offsets[i][0] = spans[i].getStart();
                offsets[i][1] = spans[i].getEnd();
            }
        }
        return offsets;
    }

    public String decode(long[] tokenIds) {
        return tokenizer.decode(tokenIds);
    }

    public int getMaxLength() {
        return maxLength;
    }

    @Override
    public void close() {
        tokenizer.close();
    }

    public record EncodingResult(long[] inputIds, long[] attentionMask, long[][] offsets) {}

    public record BatchEncodingResult(long[][] inputIds, long[][] attentionMask, long[][][] offsets) {}
}
