package com.hitorro.elsa.pipeline;

import com.hitorro.elsa.EntitySentiment;
import com.hitorro.elsa.SentimentLabel;
import com.hitorro.elsa.util.SentenceBoundary;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Mutable context that flows through the pipeline layers,
 * accumulating results at each stage.
 */
public class PipelineContext {

    // Input
    private final String originalText;

    // Layer 1 output
    private double subjectivityScore;
    private boolean subjective;

    // Layer 2a output: all sentences
    private List<SentenceBoundary> allSentences = List.of();

    // Layer 2b output: sentences that carry sentiment
    private List<SentenceBoundary> sentimentSentences = new ArrayList<>();
    private Map<SentenceBoundary, SentimentLabel> sentenceSentiments = new HashMap<>();

    // Layer 3 output: entities per sentence
    private Map<SentenceBoundary, List<ExtractedEntity>> entitiesBySentence = new LinkedHashMap<>();

    // Layer 4 output: final entity-sentiment associations
    private List<EntitySentiment> results = new ArrayList<>();

    // Control
    private boolean terminated;
    private final Map<String, Long> layerTimings = new LinkedHashMap<>();

    public PipelineContext(String originalText) {
        this.originalText = originalText;
    }

    // Accessors

    public String getOriginalText() { return originalText; }

    public double getSubjectivityScore() { return subjectivityScore; }
    public void setSubjectivityScore(double s) { this.subjectivityScore = s; }

    public boolean isSubjective() { return subjective; }
    public void setSubjective(boolean s) { this.subjective = s; }

    public List<SentenceBoundary> getAllSentences() { return allSentences; }
    public void setAllSentences(List<SentenceBoundary> s) { this.allSentences = s; }

    public List<SentenceBoundary> getSentimentSentences() { return sentimentSentences; }
    public void addSentimentSentence(SentenceBoundary s, SentimentLabel label) {
        sentimentSentences.add(s);
        sentenceSentiments.put(s, label);
    }

    public SentimentLabel getSentenceSentiment(SentenceBoundary s) {
        return sentenceSentiments.get(s);
    }

    public Map<SentenceBoundary, List<ExtractedEntity>> getEntitiesBySentence() { return entitiesBySentence; }
    public void setEntitiesForSentence(SentenceBoundary s, List<ExtractedEntity> entities) {
        entitiesBySentence.put(s, entities);
    }

    public List<EntitySentiment> getResults() { return results; }
    public void addResult(EntitySentiment r) { results.add(r); }

    public boolean isTerminated() { return terminated; }
    public void terminate() { this.terminated = true; }

    public void recordTiming(String layerName, long millis) { layerTimings.put(layerName, millis); }
    public Map<String, Long> getLayerTimings() { return layerTimings; }

    public int getSentencesAnalyzed() { return sentimentSentences.size(); }
    public int getSentencesSkipped() { return allSentences.size() - sentimentSentences.size(); }

    /**
     * Entity extracted by NER (Layer 3), before sentiment association.
     */
    public record ExtractedEntity(
            String text,
            String type,
            int startOffset,
            int endOffset,
            double confidence
    ) {}
}
