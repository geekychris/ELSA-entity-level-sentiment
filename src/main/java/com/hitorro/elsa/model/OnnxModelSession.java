package com.hitorro.elsa.model;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.util.Map;

/**
 * Thread-safe wrapper around an ONNX Runtime session.
 * Each instance holds a single model loaded into memory.
 * OrtSession.run() is safe for concurrent use.
 */
public class OnnxModelSession implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(OnnxModelSession.class);
    private static final OrtEnvironment ENV = OrtEnvironment.getEnvironment();

    private final OrtSession session;
    private final String modelName;

    public OnnxModelSession(Path modelPath, int intraOpThreads) throws OrtException {
        this.modelName = modelPath.getFileName().toString();
        var opts = new OrtSession.SessionOptions();
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        opts.setIntraOpNumThreads(intraOpThreads);
        this.session = ENV.createSession(modelPath.toString(), opts);
        log.info("Loaded ONNX model: {} (inputs: {}, outputs: {})",
                modelName, session.getInputNames(), session.getOutputNames());
    }

    public OnnxModelSession(Path modelPath) throws OrtException {
        this(modelPath, 2);
    }

    public OrtSession.Result run(Map<String, OnnxTensor> inputs) throws OrtException {
        return session.run(inputs);
    }

    public java.util.Set<String> getInputNames() {
        return session.getInputNames();
    }

    public java.util.Set<String> getOutputNames() {
        return session.getOutputNames();
    }

    public String getModelName() {
        return modelName;
    }

    public static OrtEnvironment getEnvironment() {
        return ENV;
    }

    @Override
    public void close() {
        try {
            session.close();
            log.info("Closed ONNX model: {}", modelName);
        } catch (OrtException e) {
            log.warn("Error closing ONNX session for {}: {}", modelName, e.getMessage());
        }
    }
}
