package com.hitorro.elsa.pipeline;

import com.hitorro.elsa.util.SentenceBoundary;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Layer 2a: Sentence segmentation using OpenNLP MaxEnt model.
 *
 * For short text (single sentence or < 280 chars), wraps the entire
 * text as a single sentence boundary to avoid unnecessary segmentation.
 */
public class SentenceSegmenter implements Layer {

    private static final Logger log = LoggerFactory.getLogger(SentenceSegmenter.class);
    private static final int SHORT_TEXT_THRESHOLD = 280;

    private final SentenceDetectorME detector;

    public SentenceSegmenter() {
        SentenceDetectorME det = null;
        try (InputStream modelIn = getClass().getResourceAsStream("/opennlp/en-sent.bin")) {
            if (modelIn != null) {
                SentenceModel model = new SentenceModel(modelIn);
                det = new SentenceDetectorME(model);
            } else {
                log.warn("OpenNLP sentence model not found at /opennlp/en-sent.bin; " +
                         "using simple fallback segmentation");
            }
        } catch (IOException e) {
            log.warn("Failed to load OpenNLP sentence model: {}", e.getMessage());
        }
        this.detector = det;
    }

    @Override
    public void process(PipelineContext ctx) {
        String text = ctx.getOriginalText();

        if (text.isEmpty()) {
            ctx.setAllSentences(List.of());
            return;
        }

        // Short text optimization: treat as single sentence
        if (text.length() <= SHORT_TEXT_THRESHOLD && !text.contains(". ") && !text.contains("? ") && !text.contains("! ")) {
            ctx.setAllSentences(List.of(new SentenceBoundary(text, 0, text.length())));
            return;
        }

        if (detector != null) {
            opennlp.tools.util.Span[] spans = detector.sentPosDetect(text);
            List<SentenceBoundary> sentences = new ArrayList<>(spans.length);
            for (opennlp.tools.util.Span span : spans) {
                String sentText = text.substring(span.getStart(), span.getEnd()).strip();
                if (!sentText.isEmpty()) {
                    sentences.add(new SentenceBoundary(sentText, span.getStart(), span.getEnd()));
                }
            }
            ctx.setAllSentences(sentences);
        } else {
            // Fallback: split on sentence-ending punctuation followed by space
            ctx.setAllSentences(fallbackSegment(text));
        }
    }

    private List<SentenceBoundary> fallbackSegment(String text) {
        List<SentenceBoundary> sentences = new ArrayList<>();
        int start = 0;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if ((c == '.' || c == '?' || c == '!') && i + 1 < text.length() && text.charAt(i + 1) == ' ') {
                String sentText = text.substring(start, i + 1).strip();
                if (!sentText.isEmpty()) {
                    sentences.add(new SentenceBoundary(sentText, start, i + 1));
                }
                start = i + 2;
            }
        }
        // Remainder
        if (start < text.length()) {
            String sentText = text.substring(start).strip();
            if (!sentText.isEmpty()) {
                sentences.add(new SentenceBoundary(sentText, start, text.length()));
            }
        }
        return sentences;
    }

    @Override
    public String name() {
        return "SentenceSegmenter";
    }
}
