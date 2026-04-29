package com.hitorro.elsa;

import java.time.Duration;
import java.util.List;

/**
 * Result of analyzing text for entity-level sentiment.
 *
 * @param originalText       the input text
 * @param entities           list of entity-sentiment associations found
 * @param subjectiveContent  whether Layer 1 determined the text is subjective
 * @param sentencesAnalyzed  number of sentences that passed sentiment filter
 * @param sentencesSkipped   number of sentences filtered as neutral
 * @param elapsed            total processing time
 */
public record AnalysisResult(
        String originalText,
        List<EntitySentiment> entities,
        boolean subjectiveContent,
        int sentencesAnalyzed,
        int sentencesSkipped,
        Duration elapsed
) {
    public static AnalysisResult empty(String text, Duration elapsed) {
        return new AnalysisResult(text, List.of(), false, 0, 0, elapsed);
    }

    public boolean hasEntities() {
        return !entities.isEmpty();
    }
}
