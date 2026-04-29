package com.hitorro.elsa;

/**
 * A single entity-sentiment association extracted from text.
 *
 * @param entity         the entity text (e.g., "Android Phones")
 * @param entityType     NER type (e.g., "PRODUCT", "ORG", "PER")
 * @param sentiment      sentiment directed at this entity
 * @param confidence     confidence score [0, 1]
 * @param holder         who holds this sentiment (may be null if unknown)
 * @param startOffset    character offset of entity start in original text
 * @param endOffset      character offset of entity end in original text
 * @param sourceSentence the sentence from which this was extracted
 */
public record EntitySentiment(
        String entity,
        String entityType,
        SentimentLabel sentiment,
        double confidence,
        String holder,
        int startOffset,
        int endOffset,
        String sourceSentence
) {}
