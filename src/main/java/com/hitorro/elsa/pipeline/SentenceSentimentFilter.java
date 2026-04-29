package com.hitorro.elsa.pipeline;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import com.hitorro.elsa.SentimentLabel;
import com.hitorro.elsa.model.OnnxInferenceEngine;
import com.hitorro.elsa.model.OnnxModelSession;
import com.hitorro.elsa.model.TokenizerWrapper;
import com.hitorro.elsa.util.SentenceBoundary;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;

/**
 * Layer 2b: Sentence-level sentiment filter.
 *
 * Runs a small DistilBERT classifier on each sentence to determine
 * if it carries sentiment. Neutral sentences are filtered out so
 * that expensive NER and targeted sentiment only run on sentences
 * that actually express opinion.
 *
 * Batches all sentences into a single ONNX inference call.
 */
public class SentenceSentimentFilter implements Layer {

    private static final Logger log = LoggerFactory.getLogger(SentenceSentimentFilter.class);

    private final OnnxModelSession model;
    private final TokenizerWrapper tokenizer;
    private final double threshold;

    public SentenceSentimentFilter(OnnxModelSession model, TokenizerWrapper tokenizer, double threshold) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.threshold = threshold;
    }

    @Override
    public void process(PipelineContext ctx) {
        List<SentenceBoundary> sentences = ctx.getAllSentences();
        if (sentences.isEmpty()) {
            ctx.terminate();
            return;
        }

        String[] texts = sentences.stream().map(SentenceBoundary::text).toArray(String[]::new);

        Map<String, OnnxTensor> inputs = null;
        try {
            TokenizerWrapper.BatchEncodingResult encoded = tokenizer.encodeBatch(texts);
            inputs = OnnxInferenceEngine.buildTransformerInputsBatch(
                    encoded.inputIds(), encoded.attentionMask());

            float[][] logits = OnnxInferenceEngine.runClassification(model, inputs);

            for (int i = 0; i < sentences.size(); i++) {
                float[] probs = OnnxInferenceEngine.softmax(logits[i]);
                SentimentLabel label = SentimentLabel.fromProbabilities(probs);

                // Keep sentences where max non-neutral probability exceeds threshold
                float maxSentimentProb = maxNonNeutralProb(probs);
                if (maxSentimentProb >= threshold || label != SentimentLabel.NEUTRAL) {
                    ctx.addSentimentSentence(sentences.get(i), label);
                }
            }

            if (ctx.getSentimentSentences().isEmpty()) {
                log.debug("No sentiment-bearing sentences found, terminating pipeline");
                ctx.terminate();
            }
        } catch (OrtException e) {
            log.error("Sentence sentiment inference failed: {}", e.getMessage());
            // On error, pass all sentences through
            for (SentenceBoundary s : sentences) {
                ctx.addSentimentSentence(s, SentimentLabel.NEUTRAL);
            }
        } finally {
            if (inputs != null) OnnxInferenceEngine.closeTensors(inputs);
        }
    }

    private float maxNonNeutralProb(float[] probs) {
        if (probs.length < 3) {
            return Math.max(probs[0], probs.length > 1 ? probs[1] : 0);
        }
        // probs: [negative, neutral, positive]
        return Math.max(probs[0], probs[2]);
    }

    @Override
    public String name() {
        return "SentenceSentimentFilter";
    }
}
