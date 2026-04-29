package com.hitorro.elsa.pipeline;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import com.hitorro.elsa.model.OnnxInferenceEngine;
import com.hitorro.elsa.model.OnnxModelSession;
import com.hitorro.elsa.model.TfidfVectorizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

/**
 * Layer 1: Subjectivity gate.
 *
 * Uses a TF-IDF + Logistic Regression model (~50KB ONNX) to determine
 * whether text contains subjective/opinionated content.
 * Objective/factual text is rejected immediately (~0.1ms).
 */
public class SubjectivityGate implements Layer {

    private static final Logger log = LoggerFactory.getLogger(SubjectivityGate.class);

    private final OnnxModelSession model;
    private final TfidfVectorizer vectorizer;
    private final double threshold;

    public SubjectivityGate(OnnxModelSession model, TfidfVectorizer vectorizer, double threshold) {
        this.model = model;
        this.vectorizer = vectorizer;
        this.threshold = threshold;
    }

    @Override
    public void process(PipelineContext ctx) {
        float[] features = vectorizer.transform(ctx.getOriginalText());

        Map<String, OnnxTensor> inputs = null;
        try {
            inputs = Map.of("features", OnnxInferenceEngine.createFloatTensor(new float[][]{features}));
            float[][] probs = OnnxInferenceEngine.runClassification(model, inputs);

            // Model outputs [P(objective), P(subjective)]
            double subjectivityScore = probs[0].length > 1 ? probs[0][1] : probs[0][0];
            ctx.setSubjectivityScore(subjectivityScore);

            boolean isSubjective = subjectivityScore >= threshold;
            ctx.setSubjective(isSubjective);

            if (!isSubjective) {
                log.debug("Text classified as objective (score={:.3f}), terminating pipeline",
                        subjectivityScore);
                ctx.terminate();
            }
        } catch (OrtException e) {
            log.error("Subjectivity gate inference failed: {}", e.getMessage());
            // On error, assume subjective (don't gate)
            ctx.setSubjective(true);
            ctx.setSubjectivityScore(1.0);
        } finally {
            if (inputs != null) OnnxInferenceEngine.closeTensors(inputs);
        }
    }

    @Override
    public String name() {
        return "SubjectivityGate";
    }
}
