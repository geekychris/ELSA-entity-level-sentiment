package com.hitorro.elsa.pipeline;

import com.hitorro.elsa.util.SentenceBoundary;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class SentenceSegmenterTest {

    private final SentenceSegmenter segmenter = new SentenceSegmenter();

    @Test
    void shortText_treatedAsSingleSentence() {
        PipelineContext ctx = new PipelineContext("Chris hates Android Phones");
        segmenter.process(ctx);

        assertThat(ctx.getAllSentences()).hasSize(1);
        assertThat(ctx.getAllSentences().get(0).text()).isEqualTo("Chris hates Android Phones");
    }

    @Test
    void multiSentence_splitCorrectly() {
        String text = "Chris hates Android Phones. But he loves iPhones. The weather is nice today.";
        PipelineContext ctx = new PipelineContext(text);
        segmenter.process(ctx);

        // Should detect at least 2 sentences (fallback or OpenNLP)
        assertThat(ctx.getAllSentences()).hasSizeGreaterThanOrEqualTo(2);
    }

    @Test
    void tweetLength_singleSentence() {
        String tweet = "I absolutely love the new iPhone camera quality #photography";
        PipelineContext ctx = new PipelineContext(tweet);
        segmenter.process(ctx);

        assertThat(ctx.getAllSentences()).hasSize(1);
    }

    @Test
    void emptyText_noSentences() {
        PipelineContext ctx = new PipelineContext("");
        segmenter.process(ctx);

        assertThat(ctx.getAllSentences()).isEmpty();
    }

    @Test
    void sentenceBoundary_offsetsCorrect() {
        PipelineContext ctx = new PipelineContext("Hello world");
        segmenter.process(ctx);

        SentenceBoundary sb = ctx.getAllSentences().get(0);
        assertThat(sb.startOffset()).isEqualTo(0);
        assertThat(sb.endOffset()).isEqualTo("Hello world".length());
    }
}
