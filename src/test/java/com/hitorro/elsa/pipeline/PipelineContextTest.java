package com.hitorro.elsa.pipeline;

import com.hitorro.elsa.SentimentLabel;
import com.hitorro.elsa.pipeline.PipelineContext.ExtractedEntity;
import com.hitorro.elsa.util.SentenceBoundary;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.*;

class PipelineContextTest {

    @Test
    void initialState() {
        PipelineContext ctx = new PipelineContext("test text");
        assertThat(ctx.getOriginalText()).isEqualTo("test text");
        assertThat(ctx.isTerminated()).isFalse();
        assertThat(ctx.isSubjective()).isFalse();
        assertThat(ctx.getResults()).isEmpty();
        assertThat(ctx.getSentimentSentences()).isEmpty();
    }

    @Test
    void terminate_stopsProcessing() {
        PipelineContext ctx = new PipelineContext("test");
        ctx.terminate();
        assertThat(ctx.isTerminated()).isTrue();
    }

    @Test
    void sentenceCounts() {
        PipelineContext ctx = new PipelineContext("test");
        SentenceBoundary s1 = new SentenceBoundary("sent1", 0, 5);
        SentenceBoundary s2 = new SentenceBoundary("sent2", 6, 11);
        SentenceBoundary s3 = new SentenceBoundary("sent3", 12, 17);

        ctx.setAllSentences(List.of(s1, s2, s3));
        ctx.addSentimentSentence(s1, SentimentLabel.POSITIVE);

        assertThat(ctx.getSentencesAnalyzed()).isEqualTo(1);
        assertThat(ctx.getSentencesSkipped()).isEqualTo(2);
    }

    @Test
    void entityTracking() {
        PipelineContext ctx = new PipelineContext("test");
        SentenceBoundary s1 = new SentenceBoundary("Chris hates Android", 0, 20);

        ExtractedEntity entity = new ExtractedEntity("Android", "PRODUCT", 13, 20, 0.95);
        ctx.setEntitiesForSentence(s1, List.of(entity));

        assertThat(ctx.getEntitiesBySentence()).containsKey(s1);
        assertThat(ctx.getEntitiesBySentence().get(s1)).hasSize(1);
        assertThat(ctx.getEntitiesBySentence().get(s1).get(0).text()).isEqualTo("Android");
    }

    @Test
    void layerTimings() {
        PipelineContext ctx = new PipelineContext("test");
        ctx.recordTiming("Layer1", 1);
        ctx.recordTiming("Layer2", 5);
        assertThat(ctx.getLayerTimings()).hasSize(2);
        assertThat(ctx.getLayerTimings().get("Layer1")).isEqualTo(1L);
    }
}
