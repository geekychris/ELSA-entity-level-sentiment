package com.hitorro.elsa.pipeline;

import com.hitorro.elsa.AnalysisResult;
import com.hitorro.elsa.AnalyzerConfig;
import com.hitorro.elsa.model.ModelRegistry;
import com.hitorro.elsa.util.TextNormalizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

/**
 * Orchestrates the layered analysis pipeline with early termination.
 * Each layer can short-circuit the pipeline by calling ctx.terminate().
 */
public class Pipeline {

    private static final Logger log = LoggerFactory.getLogger(Pipeline.class);

    private final List<Layer> layers;
    private final AnalyzerConfig config;

    public Pipeline(ModelRegistry registry, AnalyzerConfig config) {
        this.config = config;
        this.layers = new ArrayList<>();

        // Layer 1: Subjectivity gate
        if (registry.hasSubjectivityGate()) {
            layers.add(new SubjectivityGate(
                    registry.getSubjectivityModel(),
                    registry.getTfidfVectorizer(),
                    config.subjectivityThreshold()));
        } else {
            log.info("Subjectivity gate model not available; skipping Layer 1");
        }

        // Layer 2a: Sentence segmentation (always available - uses OpenNLP)
        layers.add(new SentenceSegmenter());

        // Layer 2b: Sentence sentiment filter
        if (registry.hasSentenceSentiment()) {
            layers.add(new SentenceSentimentFilter(
                    registry.getSentenceSentimentModel(),
                    registry.getSentenceSentimentTokenizer(),
                    config.sentimentThreshold()));
        } else {
            log.info("Sentence sentiment model not available; all sentences will pass through");
        }

        // Layer 3: Entity extraction
        if (registry.hasNer()) {
            layers.add(new EntityExtractor(
                    registry.getNerModel(),
                    registry.getNerTokenizer(),
                    config.entityTypes()));
        } else {
            log.info("NER model not available; skipping Layer 3");
        }

        // Layer 4: Entity-sentiment association
        if (registry.hasTargetedSentiment()) {
            layers.add(new EntitySentimentAssociator(
                    registry.getTargetedSentimentModel(),
                    registry.getTargetedSentimentTokenizer()));
        } else {
            log.info("Targeted sentiment model not available; skipping Layer 4");
        }
    }

    public AnalysisResult analyze(String text) {
        long startTime = System.nanoTime();

        // Normalize and truncate
        String normalized = TextNormalizer.normalize(text);
        if (normalized.isEmpty()) {
            return AnalysisResult.empty(text, Duration.ZERO);
        }
        normalized = TextNormalizer.truncate(normalized, config.maxTextLength());

        PipelineContext ctx = new PipelineContext(normalized);

        for (Layer layer : layers) {
            long layerStart = System.nanoTime();
            try {
                layer.process(ctx);
            } catch (Exception e) {
                log.error("Error in layer {}: {}", layer.name(), e.getMessage(), e);
                // Continue with next layer on error - graceful degradation
            }
            long layerMs = (System.nanoTime() - layerStart) / 1_000_000;
            ctx.recordTiming(layer.name(), layerMs);

            if (ctx.isTerminated()) {
                log.debug("Pipeline terminated at layer: {}", layer.name());
                break;
            }
        }

        Duration elapsed = Duration.ofNanos(System.nanoTime() - startTime);

        if (log.isDebugEnabled()) {
            log.debug("Pipeline timings: {}", ctx.getLayerTimings());
        }

        return new AnalysisResult(
                text,
                List.copyOf(ctx.getResults()),
                ctx.isSubjective(),
                ctx.getSentencesAnalyzed(),
                ctx.getSentencesSkipped(),
                elapsed
        );
    }
}
