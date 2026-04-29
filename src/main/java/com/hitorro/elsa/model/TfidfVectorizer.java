package com.hitorro.elsa.model;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.BufferedReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/**
 * Java-side TF-IDF vectorizer that reads vocabulary and IDF weights
 * exported from a scikit-learn TfidfVectorizer.
 *
 * Used as fallback when sklearn-onnx cannot export the full pipeline
 * with string input support. Produces a dense float vector compatible
 * with a LogisticRegression ONNX model.
 */
public class TfidfVectorizer {

    private final Map<String, Integer> vocabulary;
    private final float[] idfWeights;
    private final int maxFeatures;
    private final boolean sublinearTf;

    public TfidfVectorizer(Map<String, Integer> vocabulary, float[] idfWeights, boolean sublinearTf) {
        this.vocabulary = vocabulary;
        this.idfWeights = idfWeights;
        this.maxFeatures = idfWeights.length;
        this.sublinearTf = sublinearTf;
    }

    /**
     * Load vocabulary and IDF weights from a JSON-like format:
     * Line 1: "sublinear_tf=true|false"
     * Line 2+: "term\tindex\tidf_weight"
     */
    public static TfidfVectorizer load(Path path) throws IOException {
        try (var reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            return load(reader);
        }
    }

    public static TfidfVectorizer load(InputStream is) throws IOException {
        try (var reader = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
            return load(reader);
        }
    }

    private static TfidfVectorizer load(BufferedReader reader) throws IOException {
        String header = reader.readLine();
        boolean sublinearTf = header != null && header.contains("true");

        Map<String, Integer> vocab = new HashMap<>();
        Map<Integer, Float> idfMap = new HashMap<>();
        int maxIdx = 0;

        String line;
        while ((line = reader.readLine()) != null) {
            String[] parts = line.split("\t");
            if (parts.length >= 3) {
                String term = parts[0];
                int idx = Integer.parseInt(parts[1]);
                float idf = Float.parseFloat(parts[2]);
                vocab.put(term, idx);
                idfMap.put(idx, idf);
                if (idx > maxIdx) maxIdx = idx;
            }
        }

        float[] idfWeights = new float[maxIdx + 1];
        for (var entry : idfMap.entrySet()) {
            idfWeights[entry.getKey()] = entry.getValue();
        }
        return new TfidfVectorizer(vocab, idfWeights, sublinearTf);
    }

    /**
     * Transform a single text into a TF-IDF feature vector.
     */
    public float[] transform(String text) {
        float[] features = new float[maxFeatures];
        String lower = text.toLowerCase();

        // Tokenize: split on non-alphanumeric
        String[] tokens = lower.split("[^a-z0-9]+");

        // Count term frequencies
        Map<Integer, Integer> termCounts = new HashMap<>();
        int totalTokens = 0;

        // Unigrams
        for (String token : tokens) {
            if (token.isEmpty()) continue;
            totalTokens++;
            Integer idx = vocabulary.get(token);
            if (idx != null) {
                termCounts.merge(idx, 1, Integer::sum);
            }
        }

        // Bigrams
        for (int i = 0; i < tokens.length - 1; i++) {
            if (tokens[i].isEmpty() || tokens[i + 1].isEmpty()) continue;
            String bigram = tokens[i] + " " + tokens[i + 1];
            Integer idx = vocabulary.get(bigram);
            if (idx != null) {
                termCounts.merge(idx, 1, Integer::sum);
            }
        }

        // Compute TF-IDF
        for (var entry : termCounts.entrySet()) {
            int idx = entry.getKey();
            int count = entry.getValue();
            float tf = sublinearTf ? (float) (1.0 + Math.log(count)) : (float) count;
            features[idx] = tf * idfWeights[idx];
        }

        // L2 normalize
        float norm = 0f;
        for (float f : features) norm += f * f;
        if (norm > 0) {
            norm = (float) Math.sqrt(norm);
            for (int i = 0; i < features.length; i++) {
                features[i] /= norm;
            }
        }

        return features;
    }

    public int getFeatureCount() {
        return maxFeatures;
    }
}
