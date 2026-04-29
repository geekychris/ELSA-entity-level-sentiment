package com.hitorro.elsa.pipeline;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import com.hitorro.elsa.model.OnnxInferenceEngine;
import com.hitorro.elsa.model.OnnxModelSession;
import com.hitorro.elsa.model.TokenizerWrapper;
import com.hitorro.elsa.pipeline.PipelineContext.ExtractedEntity;
import com.hitorro.elsa.util.SentenceBoundary;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Layer 3: Named Entity Recognition.
 *
 * Runs a DistilBERT NER model on sentiment-bearing sentences to extract
 * entity spans. Handles BIO tag decoding and sub-word token reassembly.
 *
 * Standard BIO label scheme:
 * O=0, B-PER=1, I-PER=2, B-ORG=3, I-ORG=4, B-LOC=5, I-LOC=6,
 * B-MISC=7, I-MISC=8, B-PRODUCT=9, I-PRODUCT=10, ...
 */
public class EntityExtractor implements Layer {

    private static final Logger log = LoggerFactory.getLogger(EntityExtractor.class);

    // Standard NER label map (configurable; this covers CoNLL + OntoNotes common types)
    private static final String[] DEFAULT_LABELS = {
            "O",
            "B-PER", "I-PER",
            "B-ORG", "I-ORG",
            "B-LOC", "I-LOC",
            "B-MISC", "I-MISC",
            "B-PRODUCT", "I-PRODUCT",
            "B-EVENT", "I-EVENT",
            "B-WORK_OF_ART", "I-WORK_OF_ART"
    };

    private final OnnxModelSession model;
    private final TokenizerWrapper tokenizer;
    private final Set<String> entityTypeFilter;
    private final String[] labels;

    public EntityExtractor(OnnxModelSession model, TokenizerWrapper tokenizer, Set<String> entityTypeFilter) {
        this(model, tokenizer, entityTypeFilter, DEFAULT_LABELS);
    }

    public EntityExtractor(OnnxModelSession model, TokenizerWrapper tokenizer,
                           Set<String> entityTypeFilter, String[] labels) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.entityTypeFilter = entityTypeFilter;
        this.labels = labels;
    }

    @Override
    public void process(PipelineContext ctx) {
        List<SentenceBoundary> sentences = ctx.getSentimentSentences();
        if (sentences.isEmpty()) {
            ctx.terminate();
            return;
        }

        boolean anyEntities = false;

        for (SentenceBoundary sentence : sentences) {
            List<ExtractedEntity> entities = extractEntities(sentence);
            if (!entities.isEmpty()) {
                ctx.setEntitiesForSentence(sentence, entities);
                anyEntities = true;
            }
        }

        if (!anyEntities) {
            log.debug("No entities found in any sentiment sentence, terminating pipeline");
            ctx.terminate();
        }
    }

    private List<ExtractedEntity> extractEntities(SentenceBoundary sentence) {
        Map<String, OnnxTensor> inputs = null;
        try {
            TokenizerWrapper.EncodingResult encoded = tokenizer.encode(sentence.text());
            inputs = OnnxInferenceEngine.buildTransformerInputs(
                    encoded.inputIds(), encoded.attentionMask());

            float[][][] logits = OnnxInferenceEngine.runTokenLogits(model, inputs);
            float[][] tokenLogits = logits[0]; // single sentence, unbatched

            // Decode BIO tags and reassemble entities
            return decodeBioTags(tokenLogits, encoded.offsets(), sentence);

        } catch (OrtException e) {
            log.error("NER inference failed for sentence: {}", e.getMessage());
            return List.of();
        } finally {
            if (inputs != null) OnnxInferenceEngine.closeTensors(inputs);
        }
    }

    private List<ExtractedEntity> decodeBioTags(float[][] tokenLogits, long[][] offsets,
                                                 SentenceBoundary sentence) {
        List<ExtractedEntity> entities = new ArrayList<>();
        String currentType = null;
        int entityStartChar = -1;
        float entityConfidenceSum = 0;
        int entityTokenCount = 0;

        // Skip [CLS] (index 0) and [SEP] (last token)
        int seqLen = Math.min(tokenLogits.length, offsets.length);

        for (int t = 1; t < seqLen; t++) {
            // Skip special tokens (offset [0,0])
            if (offsets[t][0] == 0 && offsets[t][1] == 0 && t > 0) continue;

            float[] probs = OnnxInferenceEngine.softmax(tokenLogits[t]);
            int labelIdx = argmax(probs);
            String label = labelIdx < labels.length ? labels[labelIdx] : "O";
            float confidence = probs[labelIdx];

            if (label.startsWith("B-")) {
                // Flush previous entity if exists
                if (currentType != null) {
                    flushEntity(entities, currentType, entityStartChar, (int) offsets[t - 1][1],
                            entityConfidenceSum / entityTokenCount, sentence);
                }
                currentType = label.substring(2);
                entityStartChar = (int) offsets[t][0];
                entityConfidenceSum = confidence;
                entityTokenCount = 1;

            } else if (label.startsWith("I-") && currentType != null
                       && label.substring(2).equals(currentType)) {
                // Continue current entity
                entityConfidenceSum += confidence;
                entityTokenCount++;

            } else {
                // O tag or type mismatch: flush current entity
                if (currentType != null) {
                    flushEntity(entities, currentType, entityStartChar, (int) offsets[t - 1][1],
                            entityConfidenceSum / entityTokenCount, sentence);
                    currentType = null;
                }
            }
        }

        // Flush trailing entity
        if (currentType != null && seqLen > 1) {
            flushEntity(entities, currentType, entityStartChar, (int) offsets[seqLen - 1][1],
                    entityConfidenceSum / entityTokenCount, sentence);
        }

        return entities;
    }

    private void flushEntity(List<ExtractedEntity> entities, String type,
                             int startChar, int endChar, double confidence,
                             SentenceBoundary sentence) {
        // Apply entity type filter
        if (!entityTypeFilter.isEmpty() && !entityTypeFilter.contains(type)) return;

        String entityText = sentence.text().substring(startChar, Math.min(endChar, sentence.text().length())).strip();
        if (entityText.isEmpty()) return;

        // Adjust offsets to original text coordinates
        int absStart = sentence.startOffset() + startChar;
        int absEnd = sentence.startOffset() + endChar;

        entities.add(new ExtractedEntity(entityText, type, absStart, absEnd, confidence));
    }

    private static int argmax(float[] arr) {
        int maxIdx = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[maxIdx]) maxIdx = i;
        }
        return maxIdx;
    }

    @Override
    public String name() {
        return "EntityExtractor";
    }
}
