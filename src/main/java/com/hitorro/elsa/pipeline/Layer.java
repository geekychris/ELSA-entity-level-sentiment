package com.hitorro.elsa.pipeline;

/**
 * A single layer in the analysis pipeline.
 * Each layer processes the context and may short-circuit
 * by setting context.terminated = true.
 */
public interface Layer {

    void process(PipelineContext ctx);

    String name();
}
