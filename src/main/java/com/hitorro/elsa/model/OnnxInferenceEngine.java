package com.hitorro.elsa.model;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.nio.LongBuffer;
import java.util.HashMap;
import java.util.Map;

/**
 * Helper for creating tensors and running inference through ONNX models.
 * All tensor creation methods produce tensors that must be closed by the caller.
 */
public final class OnnxInferenceEngine {

    private static final OrtEnvironment ENV = OnnxModelSession.getEnvironment();

    private OnnxInferenceEngine() {}

    public static OnnxTensor createStringTensor(String[] values) throws OrtException {
        return OnnxTensor.createTensor(ENV, values, new long[]{values.length});
    }

    public static OnnxTensor createLongTensor(long[][] values) throws OrtException {
        return OnnxTensor.createTensor(ENV, values);
    }

    public static OnnxTensor createLongTensor1D(long[] values) throws OrtException {
        return OnnxTensor.createTensor(ENV, LongBuffer.wrap(values), new long[]{1, values.length});
    }

    public static OnnxTensor createFloatTensor(float[][] values) throws OrtException {
        return OnnxTensor.createTensor(ENV, values);
    }

    /**
     * Run inference and extract the first output as a 2D float array (batch x classes).
     * Useful for classification models that output logits or probabilities.
     */
    public static float[][] runClassification(OnnxModelSession model, Map<String, OnnxTensor> inputs) throws OrtException {
        try (OrtSession.Result result = model.run(inputs)) {
            Object raw = result.get(0).getValue();
            if (raw instanceof float[][] matrix) {
                return matrix;
            } else if (raw instanceof float[] vector) {
                return new float[][]{vector};
            }
            throw new OrtException("Unexpected output type: " + raw.getClass());
        }
    }

    /**
     * Run inference and extract the first output as a 3D long array (batch x seq_len).
     * Useful for token classification (NER) models.
     */
    public static long[][] runTokenClassification(OnnxModelSession model, Map<String, OnnxTensor> inputs) throws OrtException {
        try (OrtSession.Result result = model.run(inputs)) {
            // Token classification models output logits: [batch, seq_len, num_labels]
            Object raw = result.get(0).getValue();
            if (raw instanceof float[][][] logits) {
                // Argmax over last dimension
                long[][] predictions = new long[logits.length][logits[0].length];
                for (int b = 0; b < logits.length; b++) {
                    for (int s = 0; s < logits[b].length; s++) {
                        int maxIdx = 0;
                        for (int l = 1; l < logits[b][s].length; l++) {
                            if (logits[b][s][l] > logits[b][s][maxIdx]) maxIdx = l;
                        }
                        predictions[b][s] = maxIdx;
                    }
                }
                return predictions;
            } else if (raw instanceof long[][] labels) {
                return labels;
            }
            throw new OrtException("Unexpected output type: " + raw.getClass());
        }
    }

    /**
     * Run inference returning raw logits as 3D array [batch][seq_len][num_labels].
     */
    public static float[][][] runTokenLogits(OnnxModelSession model, Map<String, OnnxTensor> inputs) throws OrtException {
        try (OrtSession.Result result = model.run(inputs)) {
            Object raw = result.get(0).getValue();
            if (raw instanceof float[][][] logits) {
                return logits;
            }
            throw new OrtException("Unexpected output type: " + raw.getClass());
        }
    }

    /**
     * Apply softmax to a logit vector in-place and return probabilities.
     */
    public static float[] softmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) if (v > max) max = v;

        float sum = 0f;
        float[] probs = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = (float) Math.exp(logits[i] - max);
            sum += probs[i];
        }
        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sum;
        }
        return probs;
    }

    /**
     * Build standard transformer input map with input_ids and attention_mask.
     */
    public static Map<String, OnnxTensor> buildTransformerInputs(long[] inputIds, long[] attentionMask) throws OrtException {
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input_ids", createLongTensor1D(inputIds));
        inputs.put("attention_mask", createLongTensor1D(attentionMask));
        return inputs;
    }

    /**
     * Build batched transformer input map.
     */
    public static Map<String, OnnxTensor> buildTransformerInputsBatch(long[][] inputIds, long[][] attentionMask) throws OrtException {
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input_ids", createLongTensor(inputIds));
        inputs.put("attention_mask", createLongTensor(attentionMask));
        return inputs;
    }

    /**
     * Close all tensors in an input map.
     */
    public static void closeTensors(Map<String, OnnxTensor> inputs) {
        for (OnnxTensor tensor : inputs.values()) {
            tensor.close();
        }
    }
}
