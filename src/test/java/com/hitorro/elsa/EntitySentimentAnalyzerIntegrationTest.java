package com.hitorro.elsa;

import ai.onnxruntime.OrtException;
import org.junit.jupiter.api.*;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Set;

import static org.assertj.core.api.Assertions.*;

/**
 * Integration tests that exercise the full ELSA pipeline end-to-end
 * using the synthetic ONNX models in the models/ directory.
 *
 * Note: Because models are synthetic (randomly initialized), the actual
 * sentiment labels and entity extractions will not match real-world
 * expectations. These tests validate that the pipeline runs without
 * errors, produces structurally valid output, and handles edge cases.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class EntitySentimentAnalyzerIntegrationTest {

    private static final Path MODEL_DIR = Path.of("models");
    private EntitySentimentAnalyzer analyzer;

    @BeforeAll
    void setUp() throws OrtException, IOException {
        analyzer = EntitySentimentAnalyzer.create(
                AnalyzerConfig.builder()
                        .modelDirectory(MODEL_DIR)
                        .build());
    }

    @AfterAll
    void tearDown() {
        if (analyzer != null) analyzer.close();
    }

    // ---- Null / empty / blank input ----

    @Test
    void analyze_nullInput_returnsEmptyResult() {
        AnalysisResult result = analyzer.analyze(null);
        assertThat(result.entities()).isEmpty();
        assertThat(result.hasEntities()).isFalse();
        assertThat(result.subjectiveContent()).isFalse();
    }

    @Test
    void analyze_emptyString_returnsEmptyResult() {
        AnalysisResult result = analyzer.analyze("");
        assertThat(result.entities()).isEmpty();
        assertThat(result.hasEntities()).isFalse();
    }

    @Test
    void analyze_blankString_returnsEmptyResult() {
        AnalysisResult result = analyzer.analyze("   \t\n  ");
        assertThat(result.entities()).isEmpty();
    }

    // ---- Structural validation of pipeline output ----

    @Test
    void analyze_shortOpinionatedText_producesStructurallyValidResult() {
        AnalysisResult result = analyzer.analyze(
                "I absolutely love Apple's new iPhone, it's amazing!");

        assertThat(result.originalText()).isEqualTo("I absolutely love Apple's new iPhone, it's amazing!");
        assertThat(result.elapsed()).isNotNull();
        assertThat(result.elapsed()).isLessThan(Duration.ofSeconds(30));
        assertThat(result.sentencesAnalyzed() + result.sentencesSkipped()).isGreaterThanOrEqualTo(0);

        for (EntitySentiment es : result.entities()) {
            assertThat(es.entity()).isNotBlank();
            assertThat(es.entityType()).isNotBlank();
            assertThat(es.sentiment()).isNotNull();
            assertThat(es.confidence()).isBetween(0.0, 1.0);
            assertThat(es.sourceSentence()).isNotBlank();
            assertThat(es.startOffset()).isGreaterThanOrEqualTo(0);
            assertThat(es.endOffset()).isGreaterThan(es.startOffset());
        }
    }

    @Test
    void analyze_multiSentenceText_reportsConsistentCounts() {
        String text = "Google released a great new product. Microsoft failed miserably. "
                     + "The weather is nice today.";
        AnalysisResult result = analyzer.analyze(text);

        // With synthetic models, subjectivity gate may reject the text.
        // If subjective, sentence counts should be consistent.
        // If objective, both counts should be 0 (pipeline terminated early).
        int totalSentences = result.sentencesAnalyzed() + result.sentencesSkipped();
        if (result.subjectiveContent()) {
            assertThat(totalSentences).isGreaterThanOrEqualTo(1);
        } else {
            assertThat(totalSentences).isEqualTo(0);
        }
    }

    @Test
    void analyze_entitySentimentFieldsAreComplete() {
        AnalysisResult result = analyzer.analyze(
                "Sarah praised Tesla's innovation but criticized Ford's reliability.");

        for (EntitySentiment es : result.entities()) {
            // Sentiment label must be one of the known values
            assertThat(es.sentiment()).isIn(
                    SentimentLabel.POSITIVE,
                    SentimentLabel.NEGATIVE,
                    SentimentLabel.NEUTRAL,
                    SentimentLabel.MIXED);
        }
    }

    // ---- Diverse test data scenarios ----

    @Test
    void analyze_productReview() {
        AnalysisResult result = analyzer.analyze(
                "The Samsung Galaxy S25 has an incredible camera but the battery life is disappointing.");
        assertThat(result).isNotNull();
        assertThat(result.elapsed()).isPositive();
    }

    @Test
    void analyze_financialNews() {
        AnalysisResult result = analyzer.analyze(
                "JPMorgan Chase reported record quarterly profits while Goldman Sachs missed analyst expectations.");
        assertThat(result).isNotNull();
    }

    @Test
    void analyze_politicalOpinion() {
        AnalysisResult result = analyzer.analyze(
                "The mayor handled the crisis brilliantly, but the city council's response was completely inadequate.");
        assertThat(result).isNotNull();
    }

    @Test
    void analyze_socialMediaPost() {
        AnalysisResult result = analyzer.analyze(
                "@elonmusk just ruined Twitter again. Meanwhile Threads by Meta is actually getting better. #tech");
        assertThat(result).isNotNull();
    }

    @Test
    void analyze_restaurantReview() {
        AnalysisResult result = analyzer.analyze(
                "Chef Marco's pasta at Bella Italia was divine! "
                + "However, the dessert from the new pastry chef was bland and overpriced.");
        assertThat(result).isNotNull();
    }

    @Test
    void analyze_sportsCommentary() {
        AnalysisResult result = analyzer.analyze(
                "LeBron James delivered an outstanding performance last night. "
                + "Coach Williams made terrible substitution decisions that nearly cost the Lakers the game.");
        assertThat(result).isNotNull();
    }

    @Test
    void analyze_movieReview() {
        AnalysisResult result = analyzer.analyze(
                "Christopher Nolan's Oppenheimer is a masterpiece of cinema. "
                + "Cillian Murphy's portrayal was breathtaking, though Emily Blunt felt underutilized.");
        assertThat(result).isNotNull();
    }

    @Test
    void analyze_techCompanyComparison() {
        AnalysisResult result = analyzer.analyze(
                "Amazon Web Services dominates the cloud market with excellent reliability. "
                + "Azure from Microsoft is catching up, but Oracle Cloud remains frustratingly behind.");
        assertThat(result).isNotNull();
    }

    // ---- Edge cases ----

    @Test
    void analyze_unicodeText() {
        AnalysisResult result = analyzer.analyze(
                "J'adore le nouveau restaurant de Pierre Gagnaire \u2014 c'est magnifique!");
        assertThat(result).isNotNull();
    }

    @Test
    void analyze_veryLongText_isTruncated() {
        String longText = "I love Apple. ".repeat(2000); // ~28K chars
        AnalysisResult result = analyzer.analyze(longText);
        assertThat(result).isNotNull();
        assertThat(result.elapsed()).isLessThan(Duration.ofMinutes(1));
    }

    @Test
    void analyze_singleWord() {
        AnalysisResult result = analyzer.analyze("Terrible");
        assertThat(result).isNotNull();
    }

    @Test
    void analyze_purelyObjectiveText() {
        AnalysisResult result = analyzer.analyze(
                "The Earth orbits the Sun at an average distance of 93 million miles.");
        assertThat(result).isNotNull();
        // With a real model, subjectiveContent would be false, but synthetic models are unpredictable
    }

    @Test
    void analyze_textWithMixedSentiment() {
        AnalysisResult result = analyzer.analyze(
                "Netflix has both amazing original content like Stranger Things "
                + "and terrible reality shows that no one watches.");
        assertThat(result).isNotNull();
    }

    @Test
    void analyze_textWithSpecialCharacters() {
        AnalysisResult result = analyzer.analyze(
                "AT&T's 5G network is terrible! T-Mobile's coverage is 10x better. #worst #best");
        assertThat(result).isNotNull();
    }

    // ---- Batch analysis ----

    @Test
    void analyzeBatch_multipleTexts() {
        List<AnalysisResult> results = analyzer.analyzeBatch(List.of(
                "I love Google's search engine.",
                "Microsoft Teams is frustrating to use.",
                "The stock market closed unchanged today.",
                ""
        ));

        assertThat(results).hasSize(4);
        // Empty string should return empty result
        assertThat(results.get(3).entities()).isEmpty();
    }

    // ---- Config variations ----

    @Test
    void analyze_withEntityTypeFilter() throws OrtException, IOException {
        try (EntitySentimentAnalyzer filtered = EntitySentimentAnalyzer.create(
                AnalyzerConfig.builder()
                        .modelDirectory(MODEL_DIR)
                        .entityTypes(Set.of("PER", "ORG"))
                        .build())) {

            AnalysisResult result = filtered.analyze(
                    "Tim Cook unveiled Apple's new product at the Cupertino event.");

            for (EntitySentiment es : result.entities()) {
                assertThat(es.entityType()).isIn("PER", "ORG");
            }
        }
    }

    @Test
    void analyze_withCustomThresholds() throws OrtException, IOException {
        try (EntitySentimentAnalyzer strict = EntitySentimentAnalyzer.create(
                AnalyzerConfig.builder()
                        .modelDirectory(MODEL_DIR)
                        .subjectivityThreshold(0.9)
                        .sentimentThreshold(0.9)
                        .build())) {

            AnalysisResult result = strict.analyze("I somewhat like this product.");
            assertThat(result).isNotNull();
        }
    }

    // ---- Thread safety ----

    @Test
    void analyze_concurrentAccess_isThreadSafe() throws InterruptedException {
        String[] texts = {
                "Apple is amazing!",
                "Google is terrible.",
                "Microsoft does okay work.",
                "Amazon delivers fast.",
                "Tesla makes great cars.",
                "Netflix has good shows.",
                "Meta ruined Facebook.",
                "Samsung builds solid phones."
        };

        Thread[] threads = new Thread[texts.length];
        AnalysisResult[] results = new AnalysisResult[texts.length];
        Throwable[] errors = new Throwable[texts.length];

        for (int i = 0; i < texts.length; i++) {
            final int idx = i;
            threads[i] = new Thread(() -> {
                try {
                    results[idx] = analyzer.analyze(texts[idx]);
                } catch (Throwable t) {
                    errors[idx] = t;
                }
            });
        }

        for (Thread t : threads) t.start();
        for (Thread t : threads) t.join(30_000);

        for (int i = 0; i < texts.length; i++) {
            assertThat(errors[i]).as("Thread %d should not throw", i).isNull();
            assertThat(results[i]).as("Thread %d should produce a result", i).isNotNull();
        }
    }
}
