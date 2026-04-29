package com.hitorro.elsa;

import ai.onnxruntime.OrtException;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Performance analysis of the ELSA sentiment engine.
 *
 * Runs the full pipeline against diverse test data and produces
 * a detailed report including:
 * - Per-text output with entities, sentiment labels, and confidence
 * - Timing statistics (per-text, per-layer)
 * - Pipeline behavior analysis (subjectivity gate, entity counts)
 * - Throughput measurement
 */
public class PerformanceAnalysis {

    // ---- Test Data ----
    // Each entry: [category, text, description]
    private static final String[][] TEST_DATA = {

        // === Product Reviews ===
        {"Product Review",
         "I absolutely love Apple's new iPhone 16 Pro - the camera is stunning and the titanium build feels premium. However, the battery barely lasts a full day, which is frustrating for a $1200 phone.",
         "Mixed sentiment toward one product"},

        {"Product Review",
         "The Samsung Galaxy S25 Ultra blows the Pixel 9 out of the water. Google really dropped the ball on the camera this year, while Samsung nailed every aspect of the hardware.",
         "Contrasting sentiment across competing products"},

        {"Product Review",
         "Tesla's Model Y is comfortable and fast, but the build quality is inconsistent. My neighbor's BMW iX4 has much better fit and finish at a similar price point.",
         "Multi-entity comparison with opinion holder"},

        // === Financial / Business ===
        {"Financial",
         "JPMorgan Chase posted record quarterly earnings of $14.3 billion, crushing Wall Street expectations. Meanwhile, Goldman Sachs reported a disappointing 18% drop in trading revenue.",
         "Contrasting financial performance"},

        {"Financial",
         "Nvidia's stock surged 15% after CEO Jensen Huang announced breakthrough AI chip performance. AMD shares fell as investors worried about competitive positioning.",
         "Market reaction with named executives"},

        {"Financial",
         "Warren Buffett praised Apple's long-term value but warned that the broader market is overheated. Berkshire Hathaway reduced its tech holdings by 20% this quarter.",
         "Investor opinion about companies"},

        // === Entertainment ===
        {"Entertainment",
         "Christopher Nolan's Oppenheimer is a masterpiece of cinema. Cillian Murphy delivers a career-defining performance, though Emily Blunt feels criminally underutilized in her limited screen time.",
         "Movie review with multiple actors"},

        {"Entertainment",
         "Taylor Swift's Eras Tour broke every attendance record and was an incredible experience. Drake's recent album, however, was a massive letdown - lazy lyrics and recycled beats throughout.",
         "Contrasting reviews of artists"},

        {"Entertainment",
         "The Last of Us on HBO is the best video game adaptation ever made. Pedro Pascal and Bella Ramsey have incredible chemistry. Netflix's Resident Evil was an embarrassing disaster by comparison.",
         "TV show comparison with actors"},

        // === Sports ===
        {"Sports",
         "LeBron James delivered an absolutely dominant 40-point performance to carry the Lakers past the Celtics. Jayson Tatum had a rough night, shooting just 5-for-22 from the field.",
         "Player performance contrast"},

        {"Sports",
         "Coach Andy Reid's game plan was brilliant - Patrick Mahomes executed it perfectly with three touchdown passes. The Chiefs defense, led by Chris Jones, completely shut down the Bills offense.",
         "Team/coach/player praise"},

        // === Politics / Social ===
        {"Politics",
         "The mayor handled the flood crisis admirably, deploying resources within hours. However, the city council's slow approval of emergency funds drew sharp criticism from residents.",
         "Government response evaluation"},

        {"Politics",
         "Senator Martinez championed the bipartisan infrastructure bill, earning praise from both sides. Representative Davis, however, was widely condemned for blocking the vote on purely partisan grounds.",
         "Political figures with contrasting reception"},

        // === Technology ===
        {"Technology",
         "Amazon Web Services remains the gold standard for cloud computing with its unmatched reliability and service breadth. Microsoft Azure is closing the gap quickly, but Oracle Cloud continues to frustrate developers with poor documentation and frequent outages.",
         "Multi-company tech comparison"},

        {"Technology",
         "ChatGPT from OpenAI revolutionized how people interact with AI, but it still hallucinates frequently. Google's Gemini has better factual accuracy, though its creative writing capabilities lag behind Claude from Anthropic.",
         "AI product comparison"},

        {"Technology",
         "Elon Musk's decision to rebrand Twitter as X was widely mocked, and the platform has lost significant advertiser trust. Mark Zuckerberg's Threads gained 100 million users but failed to retain most of them.",
         "Social media executive decisions"},

        // === Food / Restaurant ===
        {"Restaurant",
         "Chef Marcus Samuelsson's new restaurant in Harlem is extraordinary - every dish is a work of art. The tasting menu at Daniel Boulud's place downtown felt uninspired and overpriced by comparison.",
         "Restaurant comparison with named chefs"},

        // === Travel ===
        {"Travel",
         "Japan Airlines provided impeccable service on our Tokyo flight - the crew was wonderful and the food was outstanding. United Airlines, on the other hand, lost our luggage and the cabin crew was dismissive and rude.",
         "Airline comparison"},

        // === Factual / Neutral (subjectivity gate test) ===
        {"Factual",
         "The Earth orbits the Sun at an average distance of approximately 93 million miles. Water freezes at 0 degrees Celsius at standard atmospheric pressure.",
         "Purely factual - should be filtered by subjectivity gate"},

        {"Factual",
         "Amazon was founded by Jeff Bezos in 1994 in his garage in Bellevue, Washington. The company went public on May 15, 1997 at $18 per share.",
         "Factual with named entities but no opinion"},

        // === Edge Cases ===
        {"Edge Case",
         "Terrible!",
         "Single word with strong sentiment"},

        {"Edge Case",
         "I hate everything about this product from Samsung except the screen, which is actually the best display I've ever seen on any device from any manufacturer including Apple, LG, and Sony.",
         "Complex sentence with many entities"},

        {"Edge Case",
         "lol @elonmusk just mass-fired Twitter engineers again smh... Meanwhile Satya Nadella is over at MSFT quietly building the future of AI with OpenAI partnership. Not even close. #tech #fail #win",
         "Informal social media with @mentions and hashtags"},

        {"Edge Case",
         "Die Qualit\u00e4t von Mercedes-Benz ist hervorragend, aber BMW bietet ein besseres Fahrerlebnis. Audi liegt irgendwo dazwischen.",
         "Non-English (German) text with entities"},
    };

    public static void main(String[] args) throws OrtException, IOException {
        System.out.println("=".repeat(100));
        System.out.println("  ELSA - Entity-Level Sentiment Analysis: Performance Analysis Report");
        System.out.println("=".repeat(100));
        System.out.println();

        Path modelDir = Path.of("models");

        // Warm-up: load models and run one inference
        System.out.println("[1] Loading models and warming up...");
        long loadStart = System.nanoTime();
        EntitySentimentAnalyzer analyzer = EntitySentimentAnalyzer.create(
                AnalyzerConfig.builder()
                        .modelDirectory(modelDir)
                        .build());
        long loadMs = (System.nanoTime() - loadStart) / 1_000_000;
        System.out.printf("    Model load time: %,d ms%n", loadMs);

        // Warm-up inference
        analyzer.analyze("Warm-up text about Apple and Google.");
        System.out.println("    Warm-up inference complete.");
        System.out.println();

        // Run all test data
        System.out.println("[2] Running analysis on " + TEST_DATA.length + " test inputs...");
        System.out.println("=".repeat(100));

        List<AnalysisResult> allResults = new ArrayList<>();
        List<Long> latencies = new ArrayList<>();
        int totalEntities = 0;
        int subjectiveCount = 0;
        int textsWithEntities = 0;
        Map<String, List<Long>> categoryLatencies = new LinkedHashMap<>();
        Map<SentimentLabel, Integer> labelCounts = new EnumMap<>(SentimentLabel.class);
        Map<String, Integer> entityTypeCounts = new LinkedHashMap<>();

        for (String[] entry : TEST_DATA) {
            String category = entry[0];
            String text = entry[1];
            String description = entry[2];

            long start = System.nanoTime();
            AnalysisResult result = analyzer.analyze(text);
            long elapsedMs = (System.nanoTime() - start) / 1_000_000;

            allResults.add(result);
            latencies.add(elapsedMs);
            categoryLatencies.computeIfAbsent(category, k -> new ArrayList<>()).add(elapsedMs);

            if (result.subjectiveContent()) subjectiveCount++;
            if (result.hasEntities()) textsWithEntities++;
            totalEntities += result.entities().size();

            for (EntitySentiment es : result.entities()) {
                labelCounts.merge(es.sentiment(), 1, Integer::sum);
                entityTypeCounts.merge(es.entityType(), 1, Integer::sum);
            }

            // Print detailed output
            System.out.println();
            System.out.printf("  [%s] %s%n", category, description);
            System.out.printf("  Input: \"%s\"%n", truncate(text, 120));
            System.out.printf("  Subjective: %s | Sentences analyzed: %d | Skipped: %d | Time: %,d ms%n",
                    result.subjectiveContent() ? "YES" : "NO",
                    result.sentencesAnalyzed(),
                    result.sentencesSkipped(),
                    elapsedMs);

            if (result.hasEntities()) {
                System.out.printf("  Entities found: %d%n", result.entities().size());
                for (EntitySentiment es : result.entities()) {
                    System.out.printf("    -> %-25s [%-8s]  %s  (conf: %.3f)%s%n",
                            "\"" + es.entity() + "\"",
                            es.entityType(),
                            formatSentiment(es.sentiment()),
                            es.confidence(),
                            es.holder() != null ? "  holder: " + es.holder() : "");
                }
            } else {
                System.out.println("  Entities found: 0 (no entities extracted)");
            }
            System.out.println("  " + "-".repeat(96));
        }

        // Summary statistics
        System.out.println();
        System.out.println("=".repeat(100));
        System.out.println("  SUMMARY STATISTICS");
        System.out.println("=".repeat(100));
        System.out.println();

        // Latency stats
        long totalMs = latencies.stream().mapToLong(l -> l).sum();
        double avgMs = latencies.stream().mapToLong(l -> l).average().orElse(0);
        long minMs = latencies.stream().mapToLong(l -> l).min().orElse(0);
        long maxMs = latencies.stream().mapToLong(l -> l).max().orElse(0);
        List<Long> sorted = latencies.stream().sorted().toList();
        long p50 = sorted.get(sorted.size() / 2);
        long p95 = sorted.get((int)(sorted.size() * 0.95));
        long p99 = sorted.get((int)(sorted.size() * 0.99));

        System.out.println("  LATENCY:");
        System.out.printf("    Total texts analyzed:   %d%n", TEST_DATA.length);
        System.out.printf("    Total wall time:        %,d ms%n", totalMs);
        System.out.printf("    Avg per text:           %.1f ms%n", avgMs);
        System.out.printf("    Min:                    %,d ms%n", minMs);
        System.out.printf("    Max:                    %,d ms%n", maxMs);
        System.out.printf("    P50 (median):           %,d ms%n", p50);
        System.out.printf("    P95:                    %,d ms%n", p95);
        System.out.printf("    P99:                    %,d ms%n", p99);
        System.out.printf("    Throughput:              %.1f texts/sec%n",
                totalMs > 0 ? (TEST_DATA.length * 1000.0 / totalMs) : 0);
        System.out.println();

        // Per-sentence throughput
        int totalSentencesProcessed = allResults.stream()
                .mapToInt(AnalysisResult::sentencesAnalyzed).sum();
        long subjectiveMs = 0;
        for (int i = 0; i < allResults.size(); i++) {
            if (allResults.get(i).subjectiveContent()) subjectiveMs += latencies.get(i);
        }
        List<Double> msPerSentence = new ArrayList<>();
        for (int i = 0; i < allResults.size(); i++) {
            int sa = allResults.get(i).sentencesAnalyzed();
            if (sa > 0) {
                msPerSentence.add((double) latencies.get(i) / sa);
            }
        }
        double avgMsPerSentence = msPerSentence.stream().mapToDouble(d -> d).average().orElse(0);
        double minMsPerSentence = msPerSentence.stream().mapToDouble(d -> d).min().orElse(0);
        double maxMsPerSentence = msPerSentence.stream().mapToDouble(d -> d).max().orElse(0);
        List<Double> sortedPerSent = msPerSentence.stream().sorted().toList();
        double p50PerSent = sortedPerSent.isEmpty() ? 0 : sortedPerSent.get(sortedPerSent.size() / 2);

        System.out.println("  PER-SENTENCE THROUGHPUT:");
        System.out.printf("    Total sentences processed:  %d%n", totalSentencesProcessed);
        System.out.printf("    Wall time (subjective only): %,d ms%n", subjectiveMs);
        System.out.printf("    Gross ms/sentence:          %.1f ms  (wall time / sentences)%n",
                totalSentencesProcessed > 0 ? (double) subjectiveMs / totalSentencesProcessed : 0);
        System.out.printf("    Avg ms/sentence (per text): %.1f ms%n", avgMsPerSentence);
        System.out.printf("    Min ms/sentence (per text): %.1f ms%n", minMsPerSentence);
        System.out.printf("    Max ms/sentence (per text): %.1f ms%n", maxMsPerSentence);
        System.out.printf("    P50 ms/sentence (per text): %.1f ms%n", p50PerSent);
        System.out.printf("    Sentences/sec:              %.1f%n",
                subjectiveMs > 0 ? (totalSentencesProcessed * 1000.0 / subjectiveMs) : 0);
        System.out.println();

        // Per-sentence breakdown by text
        System.out.println("  PER-SENTENCE BREAKDOWN:");
        for (int i = 0; i < allResults.size(); i++) {
            AnalysisResult r = allResults.get(i);
            int sa = r.sentencesAnalyzed();
            if (sa > 0) {
                System.out.printf("    %-14s  %d sent  %,4d ms  -> %5.1f ms/sent  | %d entities%n",
                        TEST_DATA[i][0], sa, latencies.get(i),
                        (double) latencies.get(i) / sa, r.entities().size());
            }
        }
        System.out.println();

        // Latency by category
        System.out.println("  LATENCY BY CATEGORY:");
        for (var e : categoryLatencies.entrySet()) {
            double catAvg = e.getValue().stream().mapToLong(l -> l).average().orElse(0);
            System.out.printf("    %-20s  avg: %6.1f ms  (n=%d)%n", e.getKey(), catAvg, e.getValue().size());
        }
        System.out.println();

        // Pipeline behavior
        System.out.println("  PIPELINE BEHAVIOR:");
        System.out.printf("    Texts classified as subjective:  %d / %d (%.1f%%)%n",
                subjectiveCount, TEST_DATA.length,
                100.0 * subjectiveCount / TEST_DATA.length);
        System.out.printf("    Texts with entities extracted:   %d / %d (%.1f%%)%n",
                textsWithEntities, TEST_DATA.length,
                100.0 * textsWithEntities / TEST_DATA.length);
        System.out.printf("    Total entities extracted:        %d%n", totalEntities);
        System.out.printf("    Avg entities per text:           %.1f%n",
                TEST_DATA.length > 0 ? (double) totalEntities / TEST_DATA.length : 0);
        System.out.printf("    Avg entities per text (w/ ent):  %.1f%n",
                textsWithEntities > 0 ? (double) totalEntities / textsWithEntities : 0);
        System.out.println();

        // Entity type distribution
        if (!entityTypeCounts.isEmpty()) {
            System.out.println("  ENTITY TYPE DISTRIBUTION:");
            entityTypeCounts.entrySet().stream()
                    .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                    .forEach(e -> System.out.printf("    %-15s  %d%n", e.getKey(), e.getValue()));
            System.out.println();
        }

        // Sentiment distribution
        if (!labelCounts.isEmpty()) {
            System.out.println("  SENTIMENT DISTRIBUTION:");
            int totalLabels = labelCounts.values().stream().mapToInt(i -> i).sum();
            for (SentimentLabel label : SentimentLabel.values()) {
                int count = labelCounts.getOrDefault(label, 0);
                System.out.printf("    %s%-10s%s  %3d  (%5.1f%%)  %s%n",
                        "", label, "",
                        count, 100.0 * count / totalLabels,
                        bar(count, totalLabels));
            }
            System.out.println();
        }

        // Qualitative assessment
        System.out.println("  QUALITATIVE ASSESSMENT (synthetic models):");
        System.out.println("    Note: The models/ directory contains synthetic (randomly initialized)");
        System.out.println("    ONNX models generated by create_test_models.py. These are designed");
        System.out.println("    to validate the pipeline's structural integrity, not produce");
        System.out.println("    semantically accurate predictions.");
        System.out.println();
        System.out.println("    With synthetic models, you should expect:");
        System.out.println("    - Random sentiment assignments unrelated to actual text meaning");
        System.out.println("    - Arbitrary entity boundaries (may split or merge incorrectly)");
        System.out.println("    - Subjectivity gate behavior that doesn't correlate with opinion content");
        System.out.println("    - Confidence scores clustered around 0.33 (uniform 3-class distribution)");
        System.out.println();
        System.out.println("    To get meaningful results, train real models using the scripts in training/:");
        System.out.println("      1. python training/layer1_subjectivity/train_subjectivity_gate.py");
        System.out.println("      2. python training/layer2_sentence_sentiment/train_sentence_sentiment.py");
        System.out.println("      3. python training/layer3_ner/train_ner.py");
        System.out.println("      4. python training/layer4_entity_sentiment/train_targeted_sentiment.py");
        System.out.println();

        // Confidence distribution
        if (totalEntities > 0) {
            DoubleSummaryStatistics confStats = allResults.stream()
                    .flatMap(r -> r.entities().stream())
                    .mapToDouble(EntitySentiment::confidence)
                    .summaryStatistics();

            System.out.println("  CONFIDENCE SCORE STATISTICS:");
            System.out.printf("    Mean:   %.3f%n", confStats.getAverage());
            System.out.printf("    Min:    %.3f%n", confStats.getMin());
            System.out.printf("    Max:    %.3f%n", confStats.getMax());

            // Bucket distribution
            int[] buckets = new int[10]; // 0.0-0.1, 0.1-0.2, ..., 0.9-1.0
            allResults.stream()
                    .flatMap(r -> r.entities().stream())
                    .mapToDouble(EntitySentiment::confidence)
                    .forEach(c -> {
                        int bucket = Math.min((int)(c * 10), 9);
                        buckets[bucket]++;
                    });
            System.out.println("    Distribution:");
            for (int i = 0; i < 10; i++) {
                System.out.printf("      [%.1f-%.1f)  %3d  %s%n",
                        i * 0.1, (i + 1) * 0.1, buckets[i],
                        "#".repeat(Math.min(buckets[i], 50)));
            }
            System.out.println();
        }

        System.out.println("=".repeat(100));
        System.out.printf("  Analysis complete. Model load: %,d ms | Total analysis: %,d ms%n", loadMs, totalMs);
        System.out.println("=".repeat(100));

        analyzer.close();
    }

    private static String formatSentiment(SentimentLabel label) {
        return switch (label) {
            case POSITIVE -> "\u2705 POSITIVE ";
            case NEGATIVE -> "\u274c NEGATIVE ";
            case NEUTRAL  -> "\u2796 NEUTRAL  ";
            case MIXED    -> "\u26a0\ufe0f MIXED    ";
        };
    }

    private static String truncate(String s, int maxLen) {
        return s.length() <= maxLen ? s : s.substring(0, maxLen - 3) + "...";
    }

    private static String bar(int count, int total) {
        int width = 30;
        int filled = total > 0 ? (int)(width * (double) count / total) : 0;
        return "\u2588".repeat(filled) + "\u2591".repeat(width - filled);
    }
}
