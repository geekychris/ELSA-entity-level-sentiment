package com.hitorro.elsa.model;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.*;

class TfidfVectorizerTest {

    private static final String VOCAB_DATA = """
            sublinear_tf=true
            hello\t0\t1.500000
            world\t1\t2.000000
            hello world\t2\t3.000000
            test\t3\t1.200000
            """;

    private TfidfVectorizer loadFromString(String data) throws IOException {
        return TfidfVectorizer.load(new ByteArrayInputStream(data.getBytes(StandardCharsets.UTF_8)));
    }

    @Test
    void load_parsesVocabulary() throws IOException {
        TfidfVectorizer vectorizer = loadFromString(VOCAB_DATA);
        assertThat(vectorizer.getFeatureCount()).isEqualTo(4);
    }

    @Test
    void transform_producesFeatureVector() throws IOException {
        TfidfVectorizer vectorizer = loadFromString(VOCAB_DATA);
        float[] features = vectorizer.transform("hello world test");

        assertThat(features).hasSize(4);
        // All three unigrams should be non-zero
        assertThat(features[0]).isGreaterThan(0); // hello
        assertThat(features[1]).isGreaterThan(0); // world
        assertThat(features[3]).isGreaterThan(0); // test
        // Bigram "hello world" should also be non-zero
        assertThat(features[2]).isGreaterThan(0);
    }

    @Test
    void transform_l2Normalized() throws IOException {
        TfidfVectorizer vectorizer = loadFromString(VOCAB_DATA);
        float[] features = vectorizer.transform("hello world");

        // Check L2 norm is ~1.0
        double norm = 0;
        for (float f : features) norm += f * f;
        assertThat(Math.sqrt(norm)).isCloseTo(1.0, within(0.001));
    }

    @Test
    void transform_emptyText_returnsZeroVector() throws IOException {
        TfidfVectorizer vectorizer = loadFromString(VOCAB_DATA);
        float[] features = vectorizer.transform("");
        for (float f : features) {
            assertThat(f).isEqualTo(0f);
        }
    }

    @Test
    void transform_unknownTerms_returnsZeroVector() throws IOException {
        TfidfVectorizer vectorizer = loadFromString(VOCAB_DATA);
        float[] features = vectorizer.transform("completely unknown words");
        for (float f : features) {
            assertThat(f).isEqualTo(0f);
        }
    }
}
