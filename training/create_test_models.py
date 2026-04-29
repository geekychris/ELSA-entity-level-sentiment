"""
Create small synthetic ONNX models for integration testing and benchmarking.
These are NOT trained models - they produce semi-random but structurally correct
outputs to exercise the full Java pipeline.

Layer 1: LogReg classifier (TF-IDF features -> [P(obj), P(subj)])
Layer 2: Sequence classifier (input_ids, attention_mask -> [neg, neutral, pos] logits)
Layer 3: Token classifier (input_ids, attention_mask -> [batch, seq, 9] BIO logits)
Layer 4: Sequence classifier (input_ids, attention_mask -> [neg, neutral, pos] logits)
"""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import os
import json
from pathlib import Path


def create_subjectivity_gate(output_dir: Path, n_features=5000):
    """
    Create a tiny LogReg model: features (1, n_features) -> probabilities (1, 2).
    Uses hand-crafted weights that respond to certain feature indices.
    """
    # Weight matrix [n_features, 2] and bias [2]
    np.random.seed(42)
    weights = np.random.randn(n_features, 2).astype(np.float32) * 0.01
    # Make some features strongly indicate subjectivity
    # Features 0-100 weakly objective, 100-200 strongly subjective
    weights[100:200, 1] += 0.5  # subjective indicator
    weights[100:200, 0] -= 0.5
    bias = np.array([-0.5, 0.5], dtype=np.float32)  # bias toward subjective so pipeline exercises all layers

    # ONNX graph: MatMul -> Add -> Softmax
    X = helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, n_features])
    Y = helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, 2])

    W = numpy_helper.from_array(weights, name="W")
    B = numpy_helper.from_array(bias, name="B")

    matmul = helper.make_node("MatMul", ["features", "W"], ["logits"])
    add = helper.make_node("Add", ["logits", "B"], ["logits_biased"])
    softmax = helper.make_node("Softmax", ["logits_biased"], ["probabilities"], axis=1)

    graph = helper.make_graph([matmul, add, softmax], "subjectivity_gate", [X], [Y],
                               initializer=[W, B])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    onnx.save(model, str(output_dir / "subjectivity-gate.onnx"))
    print(f"  subjectivity-gate.onnx ({os.path.getsize(output_dir / 'subjectivity-gate.onnx') / 1024:.1f} KB)")

    # Create TF-IDF vocabulary file
    # Build a vocab with sentiment-bearing words mapped to high-weight feature indices
    vocab_lines = ["sublinear_tf=true"]
    sentiment_words = {
        "hate": 100, "hates": 101, "love": 102, "loves": 103,
        "terrible": 104, "awful": 105, "amazing": 106, "wonderful": 107,
        "great": 108, "bad": 109, "worst": 110, "best": 111,
        "disappointed": 112, "impressed": 113, "angry": 114, "happy": 115,
        "disgusting": 116, "beautiful": 117, "stupid": 118, "brilliant": 119,
        "annoying": 120, "excellent": 121, "horrible": 122, "fantastic": 123,
        "pathetic": 124, "outstanding": 125, "mediocre": 126, "superb": 127,
        "dreadful": 128, "magnificent": 129, "opinion": 130, "think": 131,
        "feel": 132, "believe": 133, "prefer": 134, "enjoy": 135,
        "dislike": 136, "adore": 137, "despise": 138, "appreciate": 139,
        "regret": 140, "recommend": 141, "absolutely": 142, "really": 143,
        "very": 144, "extremely": 145, "totally": 146, "completely": 147,
    }
    # Add common neutral words at low-weight indices
    neutral_words = {
        "the": 0, "a": 1, "an": 2, "is": 3, "was": 4, "are": 5,
        "were": 6, "be": 7, "been": 8, "being": 9, "have": 10,
        "has": 11, "had": 12, "do": 13, "does": 14, "did": 15,
        "will": 16, "would": 17, "could": 18, "should": 19, "may": 20,
        "might": 21, "shall": 22, "can": 23, "to": 24, "of": 25,
        "in": 26, "for": 27, "on": 28, "with": 29, "at": 30,
        "by": 31, "from": 32, "it": 33, "this": 34, "that": 35,
        "he": 36, "she": 37, "they": 38, "we": 39, "you": 40,
        "and": 41, "or": 42, "but": 43, "not": 44, "no": 45,
        "said": 46, "announced": 47, "released": 48, "reported": 49,
        "according": 50, "percent": 51, "million": 52, "year": 53,
        "company": 54, "new": 55, "market": 56, "stock": 57,
        "price": 58, "yesterday": 59, "today": 60, "billion": 61,
    }

    all_words = {**neutral_words, **sentiment_words}
    idf_base = 3.0
    for word, idx in sorted(all_words.items(), key=lambda x: x[1]):
        idf = idf_base + np.random.uniform(-0.5, 0.5)
        vocab_lines.append(f"{word}\t{idx}\t{idf:.6f}")

    # Add a sentinel entry at the last feature index so the vectorizer
    # produces a vector matching the model's expected input dimension
    vocab_lines.append(f"__pad__\t{n_features - 1}\t0.000000")

    with open(output_dir / "subjectivity-vocab.tsv", "w") as f:
        f.write("\n".join(vocab_lines) + "\n")
    print(f"  subjectivity-vocab.tsv ({len(all_words)} terms + sentinel at idx {n_features - 1})")


def create_transformer_classifier(output_dir: Path, name: str, vocab_size=30522,
                                   hidden_size=64, seq_len=128, num_labels=3):
    """
    Create a small transformer-like classifier:
    input_ids (batch, seq) + attention_mask (batch, seq) -> logits (batch, num_labels)

    Uses embedding lookup + mean pooling + linear classifier.
    """
    np.random.seed(hash(name) % 2**31)

    # Embedding table [vocab_size, hidden_size]
    embeddings = np.random.randn(vocab_size, hidden_size).astype(np.float32) * 0.02
    # Classifier weights [hidden_size, num_labels]
    clf_weights = np.random.randn(hidden_size, num_labels).astype(np.float32) * 0.1
    clf_bias = np.zeros(num_labels, dtype=np.float32)

    # Make certain token embeddings have sentiment signal
    # Common negative tokens in BERT vocab (e.g., "hate"=5765, "terrible"=6659, "awful"=9643)
    neg_tokens = [5765, 5223, 6659, 9643, 4997, 8699]
    pos_tokens = [2293, 3407, 2307, 6919, 14103, 7422]
    for t in neg_tokens:
        if t < vocab_size:
            embeddings[t, :hidden_size//3] += 0.5
    for t in pos_tokens:
        if t < vocab_size:
            embeddings[t, hidden_size//3:2*hidden_size//3] += 0.5

    # Classifier weights: first third -> negative, second third -> positive
    clf_weights[:hidden_size//3, 0] += 0.3   # negative
    clf_weights[hidden_size//3:2*hidden_size//3, 2] += 0.3  # positive

    # ONNX graph: Gather(embeddings, input_ids) -> ReduceMean -> MatMul -> Add
    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [None, None])
    attn_mask = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [None, None])
    output = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, num_labels])

    emb_init = numpy_helper.from_array(embeddings, name="embeddings")
    w_init = numpy_helper.from_array(clf_weights, name="clf_weights")
    b_init = numpy_helper.from_array(clf_bias, name="clf_bias")

    # Gather embeddings: [batch, seq, hidden]
    gather = helper.make_node("Gather", ["embeddings", "input_ids"], ["token_emb"], axis=0)

    # Simple mean pool over sequence dimension (axis=1), ignoring mask for simplicity
    # Use axes as attribute (opset 13 style) rather than as a second input (opset 18+)
    mean_pool = helper.make_node("ReduceMean", ["token_emb"], ["mean_emb"], axes=[1], keepdims=0)

    # Classifier
    matmul = helper.make_node("MatMul", ["mean_emb", "clf_weights"], ["pre_bias"])
    add = helper.make_node("Add", ["pre_bias", "clf_bias"], ["logits"])

    graph = helper.make_graph(
        [gather, mean_pool, matmul, add],
        name,
        [input_ids, attn_mask],
        [output],
        initializer=[emb_init, w_init, b_init]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    onnx.save(model, str(output_dir / f"{name}.onnx"))
    size_kb = os.path.getsize(output_dir / f"{name}.onnx") / 1024
    print(f"  {name}.onnx ({size_kb:.1f} KB)")


def create_token_classifier(output_dir: Path, vocab_size=30522, hidden_size=64,
                             num_labels=9):
    """
    Create a small token classifier for NER:
    input_ids + attention_mask -> logits [batch, seq, num_labels]

    Labels: O=0, B-PER=1, I-PER=2, B-ORG=3, I-ORG=4, B-LOC=5, I-LOC=6, B-MISC=7, I-MISC=8
    """
    np.random.seed(99)

    embeddings = np.random.randn(vocab_size, hidden_size).astype(np.float32) * 0.02
    clf_weights = np.random.randn(hidden_size, num_labels).astype(np.float32) * 0.1
    clf_bias = np.zeros(num_labels, dtype=np.float32)
    # Strong bias toward O tag
    clf_bias[0] = 2.0

    # Make capitalized-name tokens trigger B-PER/B-ORG
    # In BERT vocab, some proper noun tokens:
    # "chris"=4657, "android"=7824, "phone"=5261, "apple"=6207, "google"=8224
    # "samsung"=19102, "tesla"=27854, "iphone"=18059
    per_tokens = [4657, 3156, 4488, 5765]  # name-like tokens
    org_tokens = [6207, 8224, 19102, 27854, 18059, 7824]  # brand/company tokens

    for t in per_tokens:
        if t < vocab_size:
            embeddings[t, :hidden_size//4] += 0.8
    for t in org_tokens:
        if t < vocab_size:
            embeddings[t, hidden_size//4:hidden_size//2] += 0.8

    # Wire embeddings to labels
    clf_weights[:hidden_size//4, 1] += 0.5   # B-PER
    clf_weights[hidden_size//4:hidden_size//2, 3] += 0.5  # B-ORG

    # ONNX graph: Gather -> MatMul -> Add (per token)
    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [None, None])
    attn_mask = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [None, None])
    output = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, None, num_labels])

    emb_init = numpy_helper.from_array(embeddings, name="embeddings")
    w_init = numpy_helper.from_array(clf_weights, name="clf_weights")
    b_init = numpy_helper.from_array(clf_bias, name="clf_bias")

    gather = helper.make_node("Gather", ["embeddings", "input_ids"], ["token_emb"], axis=0)
    matmul = helper.make_node("MatMul", ["token_emb", "clf_weights"], ["pre_bias"])
    add = helper.make_node("Add", ["pre_bias", "clf_bias"], ["logits"])

    graph = helper.make_graph(
        [gather, matmul, add],
        "ner",
        [input_ids, attn_mask],
        [output],
        initializer=[emb_init, w_init, b_init]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    onnx.save(model, str(output_dir / "ner.onnx"))
    size_kb = os.path.getsize(output_dir / "ner.onnx") / 1024
    print(f"  ner.onnx ({size_kb:.1f} KB)")


def create_tokenizer_files(output_dir: Path):
    """
    Create tokenizer directories by saving a pretrained BERT tokenizer.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    for name in ["sentence-sentiment-tokenizer", "ner-tokenizer", "entity-sentiment-tokenizer"]:
        tok_dir = output_dir / name
        tok_dir.mkdir(exist_ok=True)
        tokenizer.save_pretrained(str(tok_dir))
        print(f"  {name}/ (tokenizer saved)")


def main():
    output_dir = Path(__file__).parent.parent / "models"
    output_dir.mkdir(exist_ok=True)

    print("Creating test ONNX models...")
    print()

    print("Layer 1: Subjectivity Gate")
    create_subjectivity_gate(output_dir, n_features=5000)
    print()

    print("Layer 2: Sentence Sentiment Classifier")
    create_transformer_classifier(output_dir, "sentence-sentiment",
                                   num_labels=3, hidden_size=64)
    print()

    print("Layer 3: NER Token Classifier")
    create_token_classifier(output_dir, hidden_size=64, num_labels=9)
    print()

    print("Layer 4: Targeted Entity Sentiment Classifier")
    create_transformer_classifier(output_dir, "entity-sentiment",
                                   num_labels=3, hidden_size=64)
    print()

    print("Tokenizers (using bert-base-uncased)")
    create_tokenizer_files(output_dir)
    print()

    # Verify models load in onnxruntime
    import onnxruntime as ort
    print("Verification:")
    for name in ["subjectivity-gate", "sentence-sentiment", "ner", "entity-sentiment"]:
        path = str(output_dir / f"{name}.onnx")
        sess = ort.InferenceSession(path)
        inputs = [i.name for i in sess.get_inputs()]
        outputs = [o.name for o in sess.get_outputs()]
        print(f"  {name}: inputs={inputs}, outputs={outputs}")

    print("\nAll models created successfully!")


if __name__ == "__main__":
    main()
