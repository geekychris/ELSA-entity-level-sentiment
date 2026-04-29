package com.hitorro.elsa.model;

import ai.onnxruntime.OrtException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Loads and caches all ONNX models and their associated tokenizers at startup.
 * Models are loaded from a configured directory.
 */
public class ModelRegistry implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(ModelRegistry.class);

    private final OnnxModelSession subjectivityModel;
    private final TfidfVectorizer tfidfVectorizer;

    private final OnnxModelSession sentenceSentimentModel;
    private final TokenizerWrapper sentenceSentimentTokenizer;

    private final OnnxModelSession nerModel;
    private final TokenizerWrapper nerTokenizer;

    private final OnnxModelSession targetedSentimentModel;
    private final TokenizerWrapper targetedSentimentTokenizer;

    private final int onnxThreads;

    private ModelRegistry(Path modelDir, int onnxThreads) throws OrtException, IOException {
        this.onnxThreads = onnxThreads;
        log.info("Loading models from: {}", modelDir);

        // Layer 1: Subjectivity gate (TF-IDF + LogReg)
        Path subjectivityPath = modelDir.resolve("subjectivity-gate.onnx");
        Path tfidfPath = modelDir.resolve("subjectivity-vocab.tsv");
        this.subjectivityModel = loadModelIfExists(subjectivityPath);
        this.tfidfVectorizer = loadTfidfIfExists(tfidfPath);

        // Layer 2: Sentence sentiment (DistilBERT)
        Path sentSentPath = modelDir.resolve("sentence-sentiment.onnx");
        Path sentSentTokenizerPath = modelDir.resolve("sentence-sentiment-tokenizer");
        this.sentenceSentimentModel = loadModelIfExists(sentSentPath);
        this.sentenceSentimentTokenizer = loadTokenizerIfExists(sentSentTokenizerPath, 128);

        // Layer 3: NER (DistilBERT)
        Path nerPath = modelDir.resolve("ner.onnx");
        Path nerTokenizerPath = modelDir.resolve("ner-tokenizer");
        this.nerModel = loadModelIfExists(nerPath);
        this.nerTokenizer = loadTokenizerIfExists(nerTokenizerPath, 128);

        // Layer 4: Targeted sentiment (DistilBERT)
        Path targetPath = modelDir.resolve("entity-sentiment.onnx");
        Path targetTokenizerPath = modelDir.resolve("entity-sentiment-tokenizer");
        this.targetedSentimentModel = loadModelIfExists(targetPath);
        this.targetedSentimentTokenizer = loadTokenizerIfExists(targetTokenizerPath, 256);

        log.info("Model loading complete");
    }

    public static ModelRegistry load(Path modelDir, int onnxThreads) throws OrtException, IOException {
        return new ModelRegistry(modelDir, onnxThreads);
    }

    private OnnxModelSession loadModelIfExists(Path path) throws OrtException {
        if (Files.exists(path)) {
            return new OnnxModelSession(path, onnxThreads);
        }
        log.warn("Model not found: {} (layer will be skipped)", path);
        return null;
    }

    private TokenizerWrapper loadTokenizerIfExists(Path path, int maxLength) throws IOException {
        if (Files.isDirectory(path)) {
            return new TokenizerWrapper(path, maxLength);
        }
        log.warn("Tokenizer not found: {} (layer will be skipped)", path);
        return null;
    }

    private TfidfVectorizer loadTfidfIfExists(Path path) throws IOException {
        if (Files.exists(path)) {
            return TfidfVectorizer.load(path);
        }
        log.warn("TF-IDF vocab not found: {} (Layer 1 will be skipped)", path);
        return null;
    }

    public OnnxModelSession getSubjectivityModel() { return subjectivityModel; }
    public TfidfVectorizer getTfidfVectorizer() { return tfidfVectorizer; }
    public OnnxModelSession getSentenceSentimentModel() { return sentenceSentimentModel; }
    public TokenizerWrapper getSentenceSentimentTokenizer() { return sentenceSentimentTokenizer; }
    public OnnxModelSession getNerModel() { return nerModel; }
    public TokenizerWrapper getNerTokenizer() { return nerTokenizer; }
    public OnnxModelSession getTargetedSentimentModel() { return targetedSentimentModel; }
    public TokenizerWrapper getTargetedSentimentTokenizer() { return targetedSentimentTokenizer; }

    public boolean hasSubjectivityGate() { return subjectivityModel != null && tfidfVectorizer != null; }
    public boolean hasSentenceSentiment() { return sentenceSentimentModel != null && sentenceSentimentTokenizer != null; }
    public boolean hasNer() { return nerModel != null && nerTokenizer != null; }
    public boolean hasTargetedSentiment() { return targetedSentimentModel != null && targetedSentimentTokenizer != null; }

    @Override
    public void close() {
        if (subjectivityModel != null) subjectivityModel.close();
        if (sentenceSentimentModel != null) sentenceSentimentModel.close();
        if (sentenceSentimentTokenizer != null) sentenceSentimentTokenizer.close();
        if (nerModel != null) nerModel.close();
        if (nerTokenizer != null) nerTokenizer.close();
        if (targetedSentimentModel != null) targetedSentimentModel.close();
        if (targetedSentimentTokenizer != null) targetedSentimentTokenizer.close();
    }
}
