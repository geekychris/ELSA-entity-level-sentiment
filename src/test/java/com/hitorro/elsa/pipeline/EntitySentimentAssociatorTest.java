package com.hitorro.elsa.pipeline;

import com.hitorro.elsa.pipeline.PipelineContext.ExtractedEntity;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class EntitySentimentAssociatorTest {

    @Test
    void markEntity_insertsTargetTokens() {
        // Use a no-model instance just to test the marking logic
        EntitySentimentAssociator associator = new EntitySentimentAssociator(null, null);

        ExtractedEntity entity = new ExtractedEntity("Android Phones", "PRODUCT", 12, 26, 0.9);
        String marked = associator.markEntity("Chris hates Android Phones but loves iPhones", entity);

        assertThat(marked).contains("[TGT]");
        assertThat(marked).contains("[/TGT]");
        assertThat(marked).contains("Android Phones");
        // Entity should be wrapped
        assertThat(marked).contains("[TGT] Android Phones [/TGT]");
    }

    @Test
    void markEntity_entityNotFound_prependsFallback() {
        EntitySentimentAssociator associator = new EntitySentimentAssociator(null, null);

        ExtractedEntity entity = new ExtractedEntity("Tesla", "ORG", 0, 5, 0.9);
        String marked = associator.markEntity("Some unrelated sentence", entity);

        assertThat(marked).startsWith(" [TGT] Tesla [/TGT]");
    }

    @Test
    void findNearestHolder_attributesToClosestPrecedingPerson() {
        EntitySentimentAssociator associator = new EntitySentimentAssociator(null, null);

        // "Chris likes Sinclair Spectrums and Kyle hates ipads"
        // Chris=0..5, Sinclair Spectrums=12..30, Kyle=35..39, ipads=46..51
        var chris = new ExtractedEntity("Chris", "PER", 0, 5, 0.95);
        var spectrums = new ExtractedEntity("Sinclair Spectrums", "ORG", 12, 30, 0.93);
        var kyle = new ExtractedEntity("Kyle", "PER", 35, 39, 0.96);
        var ipads = new ExtractedEntity("ipads", "ORG", 46, 51, 0.98);

        var persons = java.util.List.of(chris, kyle);

        // Use reflection to test the private method indirectly via process(),
        // or test via the package-visible markEntity + holder logic.
        // For now, verify via a lightweight integration approach:
        // Sinclair Spectrums is closer to Chris, ipads is closer to Kyle
        // We test this by checking the pairs built during process()
        // but since process() needs a model, we test the marking logic instead
        // and rely on integration tests for full holder attribution.

        // Direct test using reflection
        try {
            var method = EntitySentimentAssociator.class.getDeclaredMethod(
                    "findNearestHolder", java.util.List.class, ExtractedEntity.class);
            method.setAccessible(true);

            String spectrumsHolder = (String) method.invoke(associator, persons, spectrums);
            String ipadsHolder = (String) method.invoke(associator, persons, ipads);

            assertThat(spectrumsHolder).isEqualTo("Chris");
            assertThat(ipadsHolder).isEqualTo("Kyle");
        } catch (Exception e) {
            fail("Reflection failed: " + e.getMessage());
        }
    }

    @Test
    void markEntity_multipleEntities_marksCorrectOne() {
        EntitySentimentAssociator associator = new EntitySentimentAssociator(null, null);

        String sentence = "Chris hates Android Phones but loves iPhone cameras";
        ExtractedEntity target = new ExtractedEntity("iPhone", "PRODUCT", 38, 44, 0.85);

        String marked = associator.markEntity(sentence, target);

        // Should mark iPhone, not Android
        assertThat(marked).contains("[TGT] iPhone [/TGT]");
        assertThat(marked).doesNotContain("[TGT] Android");
    }
}
