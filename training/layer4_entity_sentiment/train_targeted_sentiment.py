"""
Layer 4: Fine-tune DistilBERT for targeted (entity-level) sentiment analysis
and export to ONNX.

Uses SemEval 2014 Task 4 (aspect-based sentiment analysis) and optionally
Twitter targeted sentiment datasets.

Input format: entity is marked with [TGT]/[/TGT] special tokens.
  e.g., "Chris hates [TGT] Android Phones [/TGT] but loves the iPhone camera."

Output: 3-class sentiment toward the marked entity (negative=0, neutral=1, positive=2)
"""

from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
import numpy as np
import xml.etree.ElementTree as ET


MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 3
MAX_LENGTH = 256
TGT_OPEN = "[TGT]"
TGT_CLOSE = "[/TGT]"

SENTIMENT_MAP = {"negative": 0, "neutral": 1, "positive": 2, "conflict": 1}


def load_semeval_data():
    """
    Build targeted sentiment data from SST-2 + synthetic multi-entity examples.
    SST-2 sentences provide natural language variety; we insert [TGT] markers
    around key phrases to teach the model to attend to marked spans.
    """
    texts = []
    labels = []

    # Part 1: SST-2 based - natural language with [TGT] wrapping the whole sentence focus
    try:
        ds = load_dataset("stanfordnlp/sst2", split="train")
        for example in ds:
            sent = example["sentence"]
            label = example["label"]  # 0=neg, 1=pos
            mapped = 0 if label == 0 else 2  # skip neutral for SST

            # Find a noun-like word to wrap (first capitalized word or first long word)
            words = sent.split()
            target = None
            for w in words:
                clean = w.strip(".,!?;:'\"()[]")
                if clean and clean[0].isupper() and len(clean) > 1:
                    target = clean
                    break
            if target is None:
                for w in words:
                    clean = w.strip(".,!?;:'\"()[]")
                    if len(clean) >= 4 and clean.isalpha():
                        target = clean
                        break
            if target is None:
                continue

            marked = mark_entity(sent, target)
            texts.append(marked)
            labels.append(mapped)
        print(f"Loaded {len(texts)} examples from SST-2 with [TGT] markers")
    except Exception as e:
        print(f"SST-2 not available ({e}), using synthetic data only")

    # Part 2: Synthetic multi-entity contrastive examples
    synth_texts, synth_labels = create_synthetic_data()
    texts.extend(synth_texts)
    labels.extend(synth_labels)

    print(f"Total training examples: {len(texts)}")
    return texts, labels


def prepare_from_hf(ds):
    """Prepare data from HuggingFace format."""
    texts = []
    labels = []

    for split in ["train", "test"]:
        if split not in ds:
            continue
        for example in ds[split]:
            sentence = example.get("sentence", example.get("text", ""))
            aspect = example.get("aspect", example.get("term", ""))
            polarity = example.get("polarity", example.get("sentiment", "neutral"))

            if not sentence or not aspect:
                continue

            marked = mark_entity(sentence, aspect)
            texts.append(marked)
            labels.append(SENTIMENT_MAP.get(polarity, 1))

    return texts, labels


def create_synthetic_data():
    """Create synthetic training data for targeted sentiment."""
    patterns = [
        # Negative patterns - diverse phrasing
        ("{holder} hates {tgt}", 0),
        ("{holder} despises {tgt}", 0),
        ("{holder} is disappointed with {tgt}", 0),
        ("{tgt} is terrible according to {holder}", 0),
        ("{holder} can't stand {tgt}", 0),
        ("{tgt} is the worst", 0),
        ("{holder} regrets buying {tgt}", 0),
        ("I really dislike {tgt}", 0),
        ("{tgt} is absolutely awful", 0),
        ("{holder} thinks {tgt} is garbage", 0),
        ("{tgt} is a complete disaster", 0),
        ("{holder} is frustrated with {tgt}", 0),
        ("{tgt} keeps breaking down", 0),
        ("{holder} would never recommend {tgt}", 0),
        ("The quality of {tgt} is unacceptable", 0),
        ("{holder} returned {tgt} because it was defective", 0),
        ("{tgt} fails to deliver on its promises", 0),
        ("{holder} is angry about {tgt}", 0),
        ("Stay away from {tgt}", 0),
        ("{tgt} has gotten worse over time", 0),

        # Positive patterns - diverse phrasing
        ("{holder} loves {tgt}", 2),
        ("{holder} adores {tgt}", 2),
        ("{holder} is impressed by {tgt}", 2),
        ("{tgt} is amazing according to {holder}", 2),
        ("{holder} recommends {tgt}", 2),
        ("{tgt} is the best", 2),
        ("{holder} enjoys using {tgt}", 2),
        ("I really love {tgt}", 2),
        ("{tgt} is absolutely wonderful", 2),
        ("{holder} thinks {tgt} is great", 2),
        ("{tgt} exceeded all expectations", 2),
        ("{holder} is a huge fan of {tgt}", 2),
        ("{tgt} is fantastic and well-designed", 2),
        ("{holder} can't live without {tgt}", 2),
        ("The quality of {tgt} is outstanding", 2),
        ("{holder} is thrilled with {tgt}", 2),
        ("{tgt} delivers exceptional performance", 2),
        ("{holder} strongly endorses {tgt}", 2),
        ("Everyone should try {tgt}", 2),
        ("{tgt} keeps getting better and better", 2),

        # Neutral patterns
        ("{holder} uses {tgt}", 1),
        ("{holder} bought {tgt}", 1),
        ("{tgt} was released yesterday", 1),
        ("{holder} switched from {tgt} to something else", 1),
        ("{tgt} has a new version", 1),
        ("{holder} is considering {tgt}", 1),
        ("{holder} owns {tgt}", 1),
        ("{tgt} is available in stores", 1),
        ("{holder} works at {tgt}", 1),
        ("{tgt} announced quarterly results", 1),

        # Multi-entity with contrasting sentiment (the key case)
        ("{holder} hates {tgt} but loves the alternative", 0),
        ("{holder} loves {tgt} but hates the competition", 2),
        ("{holder} prefers {tgt} over everything else", 2),
        ("{holder} dislikes {tgt} compared to rivals", 0),
        ("{holder} loves {tgt} even though others disagree", 2),
        ("{holder} hates {tgt} while praising the competitor", 0),
        ("Despite the hype {holder} dislikes {tgt}", 0),
        ("While others complain {holder} enjoys {tgt}", 2),
        ("{holder} thinks {tgt} is bad but the service is fine", 0),
        ("{holder} thinks {tgt} is great but overpriced", 2),
    ]

    entities = [
        "Android Phones", "iPhone", "Samsung Galaxy", "Google Pixel",
        "Tesla", "Toyota", "Amazon", "Apple", "Microsoft", "Netflix",
        "the food", "the service", "the battery life", "the camera",
        "the screen", "the software", "the design", "the price",
    ]

    holders = ["Chris", "Sarah", "The reviewer", "Everyone", "Nobody", "Most people"]

    texts = []
    labels = []

    for pattern, sentiment in patterns:
        for entity in entities:
            for holder in holders:
                sentence = pattern.format(tgt=entity, holder=holder)
                marked = mark_entity(sentence, entity)
                texts.append(marked)
                labels.append(sentiment)

    # Multi-entity contrastive examples: two entities in one sentence with
    # opposing sentiments. The model must learn to attend to the [TGT] markers
    # and ignore sentiment words in clauses about the other entity.
    contrastive_texts, contrastive_labels = create_contrastive_data(entities, holders)
    texts.extend(contrastive_texts)
    labels.extend(contrastive_labels)

    # Three-entity / multi-holder contrastive examples: each non-target clause
    # carries an opposing sentiment word so the model cannot pick a label by
    # vote-counting and must attend to [TGT].
    multi_texts, multi_labels = create_multi_entity_contrastive_data(entities, holders)
    texts.extend(multi_texts)
    labels.extend(multi_labels)

    print(f"Created {len(texts)} synthetic training examples")
    return texts, labels


def create_multi_entity_contrastive_data(entities, holders):
    """Three-entity, multi-holder contrastive sentences.

    Each template has three entity slots (a, b, c) with explicit sentiment per
    slot. We emit one example per slot as the target, labeled with that slot's
    sentiment. This forces the model to localize to [TGT] rather than averaging
    sentiment cues across the sentence.
    """
    import itertools, random

    # (template, sent_a, sent_b, sent_c)
    # 0 = NEG, 1 = NEU, 2 = POS
    templates = [
        ("{h1} loves {a} and {h2} loves {b} but hates {c}", 2, 2, 0),
        ("{h1} hates {a} but {h2} loves {b} and adores {c}", 0, 2, 2),
        ("{h1} loves {a} but {h2} hates {b} and dislikes {c}", 2, 0, 0),
        ("{h1} dislikes {a} while {h2} adores {b} and praises {c}", 0, 2, 2),
        ("{h1} adores {a} and {h2} dislikes {b} and despises {c}", 2, 0, 0),
        ("{h1} hates {a} and hates {b} but loves {c}", 0, 0, 2),
        ("{h1} loves {a} and loves {b} but hates {c}", 2, 2, 0),
        ("{h1} thinks {a} is great but {b} is terrible and {c} is awful", 2, 0, 0),
        ("{h1} thinks {a} is terrible but {b} is great and {c} is amazing", 0, 2, 2),
        ("{h1} prefers {a} over {b} and would never use {c}", 2, 0, 0),
        ("{h1} recommends {a} but warns against {b} and avoids {c}", 2, 0, 0),
        ("{h1} enjoys {a} while {h2} despises {b} and tolerates {c}", 2, 0, 1),
        ("{h1} owns {a} and uses {b} but cannot stand {c}", 1, 1, 0),
        ("{h1} bought {a} and recommends {b} but returned {c}", 1, 2, 0),
        ("{h1} loves {a}, hates {b}, and is indifferent about {c}", 2, 0, 1),
        # Four-entity stress patterns (target rotated across first three slots only;
        # the fourth slot stays a distractor)
        ("{h1} loves {a} but {h2} hates {b} while {h1} adores {c} and dislikes {d}", 2, 0, 2),
        ("{h1} hates {a} and {h2} loves {b} while {h1} loves {c} but dislikes {d}", 0, 2, 2),
    ]

    holders_pool = ["Chris", "Sarah", "Bill", "Kyle", "Alex", "Jordan", "Taylor", "Morgan", "Reza", "Priya"]

    rng = random.Random(7)
    texts = []
    labels = []

    # Pick triples (and quads) of distinct entities; bound combinatorics
    triples = list(itertools.combinations(entities, 3))
    rng.shuffle(triples)
    triples = triples[:120]

    quads = list(itertools.combinations(entities, 4))
    rng.shuffle(quads)
    quads = quads[:60]

    for template, sa, sb, sc in templates:
        is_quad = "{d}" in template
        pool = quads if is_quad else triples
        for ents in pool:
            for _ in range(2):  # a couple of holder samples per pool entry
                h1, h2 = rng.sample(holders_pool, 2)
                if is_quad:
                    a, b, c, d = ents
                    sent_kwargs = {"h1": h1, "h2": h2, "a": a, "b": b, "c": c, "d": d}
                else:
                    a, b, c = ents
                    sent_kwargs = {"h1": h1, "h2": h2, "a": a, "b": b, "c": c}
                base = template.format(**sent_kwargs) + "."
                # Rotate target through each labeled slot
                for slot, sent in (("a", sa), ("b", sb), ("c", sc)):
                    target = sent_kwargs[slot]
                    marked = mark_entity(base, target)
                    texts.append(marked)
                    labels.append(sent)

    print(f"  Including {len(texts)} 3+ entity contrastive examples")
    return texts, labels


def create_contrastive_data(entities, holders):
    """
    Generate training examples with two entities and contrasting sentiment.
    For each pattern, one entity is the [TGT] target and gets the label;
    the other entity is a distractor with opposite sentiment.
    """
    # (pattern with {tgt} and {other}, label for {tgt})
    # Each pattern is duplicated with target and distractor swapped
    contrastive_patterns = [
        # Target positive, distractor negative
        ("{holder} loves {tgt} and hates {other}", 2),
        ("{holder} loves {tgt} but dislikes {other}", 2),
        ("{holder} loves {tgt} while despising {other}", 2),
        ("{holder} adores {tgt} but can't stand {other}", 2),
        ("{holder} is a huge fan of {tgt} but thinks {other} is terrible", 2),
        ("{holder} recommends {tgt} over {other} which is awful", 2),
        ("{holder} is thrilled with {tgt} but disappointed by {other}", 2),
        ("{holder} thinks {tgt} is amazing but {other} is garbage", 2),
        ("{holder} enjoys {tgt} even though {other} is better priced", 2),
        ("{holder} prefers {tgt} and would never buy {other}", 2),

        # Target negative, distractor positive
        ("{holder} hates {tgt} and loves {other}", 0),
        ("{holder} hates {tgt} but enjoys {other}", 0),
        ("{holder} dislikes {tgt} while loving {other}", 0),
        ("{holder} can't stand {tgt} but adores {other}", 0),
        ("{holder} thinks {tgt} is terrible but {other} is great", 0),
        ("{holder} returned {tgt} and switched to {other} which is much better", 0),
        ("{holder} is frustrated with {tgt} but impressed by {other}", 0),
        ("{holder} thinks {tgt} is garbage but {other} is amazing", 0),
        ("{holder} regrets buying {tgt} and wishes they got {other} instead", 0),
        ("{holder} would never recommend {tgt} but loves {other}", 0),

        # Two holders with contrasting sentiment about the target
        ("{holder} loves {tgt} but {holder2} hates it", 2),
        ("{holder} hates {tgt} but {holder2} loves it", 0),
        ("{holder} loves {tgt} and {holder2} dislikes {other}", 2),
        ("{holder} hates {tgt} and {holder2} adores {other}", 0),
        ("{holder} is impressed by {tgt} and {holder2} despises {other}", 2),
        ("{holder} is disappointed with {tgt} and {holder2} enjoys {other}", 0),
        ("{holder} loves {tgt} while {holder2} thinks {other} is better", 2),
        ("{holder} dislikes {tgt} while {holder2} praises {other}", 0),
    ]

    holders2 = ["Bill", "Kyle", "Alex", "Jordan", "Taylor", "Morgan"]

    texts = []
    labels = []

    import itertools
    entity_pairs = list(itertools.combinations(entities, 2))

    for pattern, sentiment in contrastive_patterns:
        for tgt_entity, other_entity in entity_pairs[:30]:  # limit combinatorics
            for holder in holders:
                holder2 = holders2[holders.index(holder) % len(holders2)]
                sentence = pattern.format(
                    tgt=tgt_entity, other=other_entity,
                    holder=holder, holder2=holder2
                )
                marked = mark_entity(sentence, tgt_entity)
                texts.append(marked)
                labels.append(sentiment)

    print(f"  Including {len(texts)} contrastive multi-entity examples")
    return texts, labels


def mark_entity(sentence: str, entity: str) -> str:
    """Insert [TGT]/[/TGT] markers around the target entity."""
    idx = sentence.find(entity)
    if idx >= 0:
        return (sentence[:idx] + f" {TGT_OPEN} " + entity +
                f" {TGT_CLOSE} " + sentence[idx + len(entity):])
    return f"{TGT_OPEN} {entity} {TGT_CLOSE} {sentence}"


def train(texts, labels):
    """Fine-tune DistilBERT with special [TGT]/[/TGT] tokens."""
    import torch

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Add special tokens
    special_tokens = {"additional_special_tokens": [TGT_OPEN, TGT_CLOSE]}
    tokenizer.add_special_tokens(special_tokens)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    # Initialize [TGT] from [CLS] and [/TGT] from [SEP] so the model
    # starts with meaningful representations for the markers
    with torch.no_grad():
        cls_id = tokenizer.convert_tokens_to_ids("[CLS]")
        sep_id = tokenizer.convert_tokens_to_ids("[SEP]")
        tgt_open_id = tokenizer.convert_tokens_to_ids(TGT_OPEN)
        tgt_close_id = tokenizer.convert_tokens_to_ids(TGT_CLOSE)
        embeddings = model.distilbert.embeddings.word_embeddings.weight
        embeddings[tgt_open_id] = embeddings[cls_id].clone()
        embeddings[tgt_close_id] = embeddings[sep_id].clone()

    # Tokenize
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
    })
    dataset.set_format("torch")

    # Split 90/10
    split = dataset.train_test_split(test_size=0.1, seed=42)

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labs = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=preds, references=labs)

    output_dir = Path(__file__).parent / "checkpoint"

    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS (Apple Silicon Metal) is not available. This trainer requires "
            "an Apple Silicon Mac with PyTorch built with MPS support."
        )
    print(f"Using MPS device: {torch.backends.mps.is_available()} (built={torch.backends.mps.is_built()})")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        # Trainer auto-uses MPS when available; pin_memory must be off for MPS
        dataloader_pin_memory=False,
        fp16=False,
        bf16=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "best"))
    tokenizer.save_pretrained(str(output_dir / "best"))

    return output_dir / "best", tokenizer


def export_onnx(model_path: Path, output_dir: Path):
    """Export targeted sentiment model to ONNX."""
    import shutil
    from optimum.onnxruntime import ORTModelForSequenceClassification

    ort_model = ORTModelForSequenceClassification.from_pretrained(
        model_path, export=True
    )
    ort_model.save_pretrained(output_dir / "entity_sentiment_onnx_export")

    exported = output_dir / "entity_sentiment_onnx_export" / "model.onnx"
    shutil.copy(exported, output_dir / "entity-sentiment.onnx")

    # Save tokenizer (with special tokens)
    tokenizer_out = output_dir / "entity-sentiment-tokenizer"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(str(tokenizer_out))

    print(f"Entity sentiment ONNX model exported to {output_dir / 'entity-sentiment.onnx'}")
    print(f"Tokenizer saved to {tokenizer_out}")


def main():
    models_dir = Path(__file__).parent.parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    print("Loading training data...")
    texts, labels = load_semeval_data()
    print(f"Training data: {len(texts)} examples")

    print("Training targeted sentiment model...")
    model_path, tokenizer = train(texts, labels)

    print("Exporting to ONNX...")
    export_onnx(model_path, models_dir)

    # Verify
    print("\nVerification:")
    test_cases = [
        f"Chris hates {TGT_OPEN} Android Phones {TGT_CLOSE} but loves the iPhone.",
        f"Chris loves {TGT_OPEN} iPhone {TGT_CLOSE} camera quality.",
        f"{TGT_OPEN} Tesla {TGT_CLOSE} released a new model yesterday.",
    ]
    for text in test_cases:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        import onnxruntime as ort
        sess = ort.InferenceSession(str(models_dir / "entity-sentiment.onnx"))
        inputs = {
            "input_ids": enc["input_ids"].numpy(),
            "attention_mask": enc["attention_mask"].numpy(),
        }
        result = sess.run(None, inputs)
        logits = result[0][0]
        pred = ["NEGATIVE", "NEUTRAL", "POSITIVE"][np.argmax(logits)]
        print(f"  '{text}' -> {pred}")

    print("\nDone!")


if __name__ == "__main__":
    main()
