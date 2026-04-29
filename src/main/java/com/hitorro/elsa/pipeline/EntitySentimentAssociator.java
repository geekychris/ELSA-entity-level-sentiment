package com.hitorro.elsa.pipeline;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import com.hitorro.elsa.EntitySentiment;
import com.hitorro.elsa.SentimentLabel;
import com.hitorro.elsa.model.OnnxInferenceEngine;
import com.hitorro.elsa.model.OnnxModelSession;
import com.hitorro.elsa.model.TokenizerWrapper;
import com.hitorro.elsa.pipeline.PipelineContext.ExtractedEntity;
import com.hitorro.elsa.util.SentenceBoundary;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Layer 4: Entity-Sentiment Association (targeted sentiment analysis).
 *
 * For each (sentence, entity) pair, marks the target entity with special
 * tokens [TGT]/[/TGT] and classifies sentiment toward that entity.
 *
 * Also performs holder detection: if a PER entity exists in the same
 * sentence as a non-PER target entity, the PER entity is assigned
 * as the sentiment holder.
 *
 * Input format: "Chris hates [TGT] Android Phones [/TGT] but loves the iPhone camera."
 */
public class EntitySentimentAssociator implements Layer {

    private static final Logger log = LoggerFactory.getLogger(EntitySentimentAssociator.class);
    private static final String TGT_OPEN = " [TGT] ";
    private static final String TGT_CLOSE = " [/TGT] ";

    private final OnnxModelSession model;
    private final TokenizerWrapper tokenizer;

    public EntitySentimentAssociator(OnnxModelSession model, TokenizerWrapper tokenizer) {
        this.model = model;
        this.tokenizer = tokenizer;
    }

    @Override
    public void process(PipelineContext ctx) {
        Map<SentenceBoundary, List<ExtractedEntity>> entitiesBySentence = ctx.getEntitiesBySentence();

        // Collect all (sentence, entity) pairs and their marked inputs
        List<SentenceEntityPair> pairs = new ArrayList<>();
        for (var entry : entitiesBySentence.entrySet()) {
            SentenceBoundary sentence = entry.getKey();
            List<ExtractedEntity> entities = entry.getValue();

            // Collect PER entities for proximity-based holder detection
            List<ExtractedEntity> personEntities = entities.stream()
                    .filter(e -> "PER".equals(e.type()))
                    .toList();

            for (ExtractedEntity entity : entities) {
                String markedInput = markEntity(sentence.text(), entity);
                String holder = entity.type().equals("PER") ? null
                        : findNearestHolder(personEntities, entity);
                pairs.add(new SentenceEntityPair(sentence, entity, markedInput, holder));
            }
        }

        if (pairs.isEmpty()) return;

        // Identify PER entities that are only acting as holders (opinion sources).
        // These are filtered from results since their sentiment row is misleading —
        // it shows sentiment *toward* them, but they're really just the opinion holder.
        Set<String> holderOnlyPersons = findHolderOnlyPersons(pairs);

        // Batch inference for all pairs
        String[] markedTexts = pairs.stream().map(p -> p.markedInput).toArray(String[]::new);

        Map<String, OnnxTensor> inputs = null;
        try {
            TokenizerWrapper.BatchEncodingResult encoded = tokenizer.encodeBatch(markedTexts);
            inputs = OnnxInferenceEngine.buildTransformerInputsBatch(
                    encoded.inputIds(), encoded.attentionMask());

            float[][] logits = OnnxInferenceEngine.runClassification(model, inputs);

            for (int i = 0; i < pairs.size(); i++) {
                SentenceEntityPair pair = pairs.get(i);

                // Skip PER entities that are only acting as opinion holders
                if ("PER".equals(pair.entity.type())
                        && holderOnlyPersons.contains(pair.entity.text())) {
                    continue;
                }

                float[] probs = OnnxInferenceEngine.softmax(logits[i]);
                SentimentLabel label = SentimentLabel.fromProbabilities(probs);
                double confidence = maxProb(probs);

                // If entity-level model is low-confidence (degenerate/collapsed model),
                // fall back to sentence-level sentiment from Layer 2b
                if (confidence < 0.5 || (label == SentimentLabel.NEUTRAL && allProbsSimilar(probs))) {
                    SentimentLabel sentenceLabel = ctx.getSentenceSentiment(pair.sentence);
                    if (sentenceLabel != null && sentenceLabel != SentimentLabel.NEUTRAL) {
                        label = sentenceLabel;
                        confidence = 0.7; // moderate confidence for fallback
                        log.debug("Entity '{}': falling back to sentence sentiment ({})",
                                pair.entity.text(), label);
                    }
                }

                ctx.addResult(new EntitySentiment(
                        pair.entity.text(),
                        pair.entity.type(),
                        label,
                        confidence,
                        pair.holder,
                        pair.entity.startOffset(),
                        pair.entity.endOffset(),
                        pair.sentence.text()
                ));
            }
        } catch (OrtException e) {
            log.error("Targeted sentiment inference failed: {}", e.getMessage());
            // Fallback: use sentence-level sentiment for all entities
            for (SentenceEntityPair pair : pairs) {
                if ("PER".equals(pair.entity.type())
                        && holderOnlyPersons.contains(pair.entity.text())) {
                    continue;
                }
                SentimentLabel fallback = ctx.getSentenceSentiment(pair.sentence);
                if (fallback == null) fallback = SentimentLabel.NEUTRAL;
                ctx.addResult(new EntitySentiment(
                        pair.entity.text(),
                        pair.entity.type(),
                        fallback,
                        0.5,
                        pair.holder,
                        pair.entity.startOffset(),
                        pair.entity.endOffset(),
                        pair.sentence.text()
                ));
            }
        } finally {
            if (inputs != null) OnnxInferenceEngine.closeTensors(inputs);
        }
    }

    /**
     * Mark the target entity in the sentence with [TGT]/[/TGT] tokens.
     */
    String markEntity(String sentenceText, ExtractedEntity entity) {
        // Find entity position within sentence text
        int entityLocalStart = entity.text().isEmpty() ? -1 :
                sentenceText.indexOf(entity.text());

        if (entityLocalStart >= 0) {
            int entityLocalEnd = entityLocalStart + entity.text().length();
            return sentenceText.substring(0, entityLocalStart)
                    + TGT_OPEN + entity.text() + TGT_CLOSE
                    + sentenceText.substring(entityLocalEnd);
        }
        // Fallback: prepend entity
        return TGT_OPEN + entity.text() + TGT_CLOSE + " " + sentenceText;
    }

    /**
     * Identify PER entities that serve as opinion holders rather than sentiment
     * targets. In sentences that contain non-PER entities, all PER entities are
     * treated as holders — they are the subjects expressing opinions about the
     * non-PER targets (e.g., "Steve Jobs loved Apple but Bill Gates was not").
     *
     * PER entities that appear in sentences with ONLY other PER entities are
     * kept, since they may be genuine sentiment targets (e.g., "Everyone hates Bob").
     */
    private Set<String> findHolderOnlyPersons(List<SentenceEntityPair> pairs) {
        // Group pairs by sentence to check for non-PER co-occurrence
        Map<SentenceBoundary, List<SentenceEntityPair>> bySentence = pairs.stream()
                .collect(Collectors.groupingBy(SentenceEntityPair::sentence));

        Set<String> holderOnly = new java.util.HashSet<>();
        for (List<SentenceEntityPair> sentencePairs : bySentence.values()) {
            boolean hasNonPer = sentencePairs.stream()
                    .anyMatch(p -> !"PER".equals(p.entity.type()));
            if (hasNonPer) {
                // All PER entities in this sentence are opinion holders
                sentencePairs.stream()
                        .filter(p -> "PER".equals(p.entity.type()))
                        .forEach(p -> holderOnly.add(p.entity.text()));
            }
        }
        return holderOnly;
    }

    /**
     * Find the nearest PER entity to the target, preferring the closest
     * preceding person. Falls back to the closest following person.
     * This correctly handles sentences like "Chris likes X and Kyle hates Y"
     * where each non-PER entity should be attributed to the nearest person.
     */
    private String findNearestHolder(List<ExtractedEntity> personEntities, ExtractedEntity target) {
        if (personEntities.isEmpty()) return null;

        ExtractedEntity best = null;
        int bestDistance = Integer.MAX_VALUE;
        boolean bestIsPreceding = false;

        for (ExtractedEntity per : personEntities) {
            boolean preceding = per.startOffset() < target.startOffset();
            int distance = preceding
                    ? target.startOffset() - per.endOffset()
                    : per.startOffset() - target.endOffset();

            // Prefer preceding holders; among same direction, prefer closer
            if (best == null
                    || (preceding && !bestIsPreceding)
                    || (preceding == bestIsPreceding && distance < bestDistance)) {
                best = per;
                bestDistance = distance;
                bestIsPreceding = preceding;
            }
        }
        return best.text();
    }

    /**
     * Detect a degenerate model where all probabilities are clustered together
     * (e.g., a collapsed model outputting ~0.33/0.33/0.33 or ~0.01/0.98/0.01).
     * Returns true if the spread between max and min is less than 0.3.
     */
    private boolean allProbsSimilar(float[] probs) {
        float max = probs[0], min = probs[0];
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > max) max = probs[i];
            if (probs[i] < min) min = probs[i];
        }
        // If non-neutral probs are both tiny, model has collapsed to neutral
        if (probs.length >= 3) {
            float nonNeutralMax = Math.max(probs[0], probs[2]); // neg, pos
            return nonNeutralMax < 0.1;
        }
        return (max - min) < 0.3;
    }

    private double maxProb(float[] probs) {
        float max = 0;
        for (float p : probs) if (p > max) max = p;
        return max;
    }

    @Override
    public String name() {
        return "EntitySentimentAssociator";
    }

    private record SentenceEntityPair(
            SentenceBoundary sentence,
            ExtractedEntity entity,
            String markedInput,
            String holder
    ) {}
}
