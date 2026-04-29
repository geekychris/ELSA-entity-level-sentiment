package com.hitorro.elsa;

import ai.onnxruntime.OrtException;
import com.hitorro.elsa.model.ModelRegistry;
import com.hitorro.elsa.pipeline.Pipeline;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;

/**
 * Public entry point for entity-level sentiment analysis.
 *
 * Thread-safe: a single instance can be shared across threads.
 * Models are loaded once at creation and reused for all subsequent calls.
 *
 * <pre>
 * var analyzer = EntitySentimentAnalyzer.create(
 *     AnalyzerConfig.builder()
 *         .modelDirectory(Path.of("models/"))
 *         .build());
 *
 * AnalysisResult result = analyzer.analyze(
 *     "Chris hates Android Phones but loves the iPhone camera.");
 *
 * for (EntitySentiment es : result.entities()) {
 *     System.out.printf("%s -> %s (%.2f)%n",
 *         es.entity(), es.sentiment(), es.confidence());
 * }
 * </pre>
 */
public class EntitySentimentAnalyzer implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(EntitySentimentAnalyzer.class);

    private final Pipeline pipeline;
    private final ModelRegistry registry;

    private EntitySentimentAnalyzer(Pipeline pipeline, ModelRegistry registry) {
        this.pipeline = pipeline;
        this.registry = registry;
    }

    /**
     * Create an analyzer with the given configuration.
     * Loads all ONNX models from the configured model directory.
     * Missing models are tolerated — their layers are simply skipped.
     */
    public static EntitySentimentAnalyzer create(AnalyzerConfig config) throws OrtException, IOException {
        ModelRegistry registry = ModelRegistry.load(config.modelDirectory(), config.onnxThreads());
        Pipeline pipeline = new Pipeline(registry, config);
        log.info("EntitySentimentAnalyzer created with config: modelDir={}, " +
                 "subjectivityThreshold={}, sentimentThreshold={}",
                config.modelDirectory(), config.subjectivityThreshold(), config.sentimentThreshold());
        return new EntitySentimentAnalyzer(pipeline, registry);
    }

    /**
     * Analyze text for entity-level sentiment.
     *
     * @param text the input text (tweet, article, etc.)
     * @return analysis result with entity-sentiment associations
     */
    public AnalysisResult analyze(String text) {
        if (text == null || text.isBlank()) {
            return AnalysisResult.empty(text, java.time.Duration.ZERO);
        }
        return pipeline.analyze(text);
    }

    /**
     * Analyze multiple texts, returning results in the same order.
     */
    public List<AnalysisResult> analyzeBatch(List<String> texts) {
        return texts.stream().map(this::analyze).toList();
    }

    @Override
    public void close() {
        registry.close();
    }
}
