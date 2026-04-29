"""
Layer 3: Fine-tune DistilBERT for Named Entity Recognition
and export to ONNX.

Uses CoNLL-2003 NER dataset with BIO tagging:
  O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC

Can be extended to OntoNotes for PRODUCT, EVENT, WORK_OF_ART types.
"""

import random
from pathlib import Path
from datasets import (
    load_dataset, Dataset, concatenate_datasets,
    Features, Sequence, Value, ClassLabel,
)
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from optimum.onnxruntime import ORTModelForTokenClassification
import evaluate
import numpy as np
import torch


MODEL_NAME = "distilbert-base-cased"
MAX_LENGTH = 128

LABEL_LIST = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC",
]

# Tag IDs (must match LABEL_LIST positions)
O, B_PER, I_PER, B_ORG, I_ORG, B_LOC, I_LOC = 0, 1, 2, 3, 4, 5, 6


def build_multi_holder_synthetic(n_examples: int = 4000, seed: int = 42) -> Dataset:
    """Adversarial training data for multi-holder boundary detection.

    WikiANN underrepresents conversational sentences with two PER subjects each
    expressing sentiment about different ORGs. The model learns from WikiANN to
    extend I-PER greedily, producing spans like "Steve Jobs loves" when the
    pattern is "{PER} {verb} {ORG}". These templates explicitly teach the model
    that PER spans terminate at verbs and conjunctions.
    """
    persons = [
        ("Bill", "Gates"), ("Steve", "Jobs"), ("Elon", "Musk"), ("Tim", "Cook"),
        ("Larry", "Page"), ("Sergey", "Brin"), ("Mark", "Zuckerberg"),
        ("Jeff", "Bezos"), ("Sundar", "Pichai"), ("Satya", "Nadella"),
        ("Sam", "Altman"), ("Reed", "Hastings"), ("Sheryl", "Sandberg"),
        ("Marissa", "Mayer"), ("Susan", "Wojcicki"), ("Jensen", "Huang"),
        ("Lisa", "Su"), ("Andy", "Jassy"), ("Dara", "Khosrowshahi"),
        ("Brian", "Chesky"), ("Daniel", "Ek"), ("Evan", "Spiegel"),
        ("Jack", "Dorsey"), ("Parag", "Agrawal"), ("Linda", "Yaccarino"),
    ]
    orgs = [
        "Microsoft", "Apple", "Tesla", "Google", "Amazon", "Meta", "Netflix",
        "OpenAI", "SpaceX", "IBM", "Oracle", "Adobe", "Nvidia", "Twitter",
        "YouTube", "LinkedIn", "Salesforce", "Uber", "Airbnb", "Spotify",
        "Snapchat", "Stripe", "GitHub", "Slack", "Zoom",
    ]
    verbs = [
        "loves", "hates", "likes", "dislikes", "owns", "uses", "founded",
        "leads", "criticized", "praised", "endorses", "rejected", "runs",
        "supports", "opposes", "joined", "left", "bought", "sold", "ignored",
        "trusts", "respects", "admires", "fears", "envies",
    ]
    conjunctions = ["and", "but", "while", "though", "whereas"]

    rng = random.Random(seed)
    rows = []

    def per_tokens(name):
        return [name[0], name[1]], [B_PER, I_PER]

    def org_tokens(o):
        return [o], [B_ORG]

    # Single PER + verb + ORG (boundary at verb)
    for _ in range(n_examples // 4):
        p, o, v = rng.choice(persons), rng.choice(orgs), rng.choice(verbs)
        pt, pl = per_tokens(p)
        ot, ol = org_tokens(o)
        tokens = pt + [v] + ot + ["."]
        tags = pl + [O] + ol + [O]
        rows.append({"tokens": tokens, "ner_tags": tags})

    # Two PER each with own verb + ORG (the failing pattern)
    for _ in range(n_examples // 2):
        p1, p2 = rng.sample(persons, 2)
        o1, o2 = rng.sample(orgs, 2)
        v1, v2 = rng.choice(verbs), rng.choice(verbs)
        conj = rng.choice(conjunctions)
        p1t, p1l = per_tokens(p1)
        p2t, p2l = per_tokens(p2)
        o1t, o1l = org_tokens(o1)
        o2t, o2l = org_tokens(o2)
        tokens = p1t + [v1] + o1t + [conj] + p2t + [v2] + o2t + ["."]
        tags = p1l + [O] + o1l + [O] + p2l + [O] + o2l + [O]
        rows.append({"tokens": tokens, "ner_tags": tags})

    # Three-way contrastive
    for _ in range(n_examples // 4):
        p1, p2, p3 = rng.sample(persons, 3)
        o1, o2, o3 = rng.sample(orgs, 3)
        v1, v2, v3 = rng.choice(verbs), rng.choice(verbs), rng.choice(verbs)
        p1t, p1l = per_tokens(p1)
        p2t, p2l = per_tokens(p2)
        p3t, p3l = per_tokens(p3)
        o1t, o1l = org_tokens(o1)
        o2t, o2l = org_tokens(o2)
        o3t, o3l = org_tokens(o3)
        tokens = p1t + [v1] + o1t + [","] + p2t + [v2] + o2t + [","] + ["and"] + p3t + [v3] + o3t + ["."]
        tags = p1l + [O] + o1l + [O] + p2l + [O] + o2l + [O, O] + p3l + [O] + o3l + [O]
        rows.append({"tokens": tokens, "ner_tags": tags})

    rng.shuffle(rows)
    # Match WikiANN's feature schema so concatenate_datasets aligns features.
    # WikiANN's tag set is the first 7 of LABEL_LIST.
    wikiann_tagset = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    features = Features({
        "tokens": Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=wikiann_tagset)),
    })
    return Dataset.from_list(rows, features=features)


def load_and_prepare_data():
    """Load NER data and tokenize with label alignment.
    Uses WikiANN (PanX) English as a readily available NER dataset.
    WikiANN tags: O=0, B-PER=1, I-PER=2, B-ORG=3, I-ORG=4, B-LOC=5, I-LOC=6
    We map these to our label list (which has the same indices for these tags).
    """
    try:
        ds = load_dataset("wikiann", "en")
        # WikiANN has ner_tags that align with our first 7 labels
        # (O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC)
        print(f"Loaded WikiANN: {ds['train'].num_rows} train, {ds['validation'].num_rows} val")
    except Exception as e:
        raise RuntimeError(f"Could not load NER dataset: {e}")

    # Augment training split with multi-holder adversarial examples
    synthetic = build_multi_holder_synthetic(n_examples=4000)
    print(f"Adding {synthetic.num_rows} synthetic multi-holder examples to train split")
    train_aug = concatenate_datasets([
        ds["train"].select_columns(["tokens", "ner_tags"]),
        synthetic,
    ]).shuffle(seed=42)
    ds["train"] = train_aug

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    label2id = {label: i for i, label in enumerate(LABEL_LIST)}

    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples["tokens"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            is_split_into_words=True,
        )

        labels = []
        for i, label_ids in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            label_row = []
            previous_word = None
            for word_id in word_ids:
                if word_id is None:
                    label_row.append(-100)  # special token
                elif word_id != previous_word:
                    label_row.append(label_ids[word_id])
                else:
                    # Sub-word token: use I- tag if current is B-
                    orig_label = label_ids[word_id]
                    if orig_label % 2 == 1:  # B- tag (odd indices)
                        label_row.append(orig_label + 1)  # Convert to I-
                    else:
                        label_row.append(orig_label)
                previous_word = word_id
            labels.append(label_row)

        tokenized["labels"] = labels
        return tokenized

    cols_to_remove = [c for c in ds["train"].column_names if c not in ["input_ids", "attention_mask", "labels"]]
    tokenized = ds.map(tokenize_and_align, batched=True, remove_columns=cols_to_remove)
    return tokenized, tokenizer


def train(tokenized_ds, tokenizer):
    """Fine-tune DistilBERT for NER."""
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label={i: l for i, l in enumerate(LABEL_LIST)},
        label2id={l: i for i, l in enumerate(LABEL_LIST)},
    )

    seqeval = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_labels = []
        true_preds = []
        for pred_row, label_row in zip(predictions, labels):
            t_labels = []
            t_preds = []
            for p, l in zip(pred_row, label_row):
                if l != -100:
                    t_labels.append(LABEL_LIST[l])
                    t_preds.append(LABEL_LIST[p] if p < len(LABEL_LIST) else "O")
            true_labels.append(t_labels)
            true_preds.append(t_preds)

        results = seqeval.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
        }

    output_dir = Path(__file__).parent / "checkpoint"

    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS (Apple Silicon Metal) is not available. This trainer requires "
            "an Apple Silicon Mac with PyTorch built with MPS support."
        )
    print(f"Using MPS device: {torch.backends.mps.is_available()} (built={torch.backends.mps.is_built()})")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        # Trainer auto-uses MPS when available; pin_memory must be off for MPS
        dataloader_pin_memory=False,
        fp16=False,  # MPS does not yet support fp16 cleanly
        bf16=False,
        report_to=[],
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "best"))
    tokenizer.save_pretrained(str(output_dir / "best"))

    return output_dir / "best"


def export_onnx(model_path: Path, output_dir: Path):
    """Export NER model to ONNX."""
    import shutil

    ort_model = ORTModelForTokenClassification.from_pretrained(
        model_path, export=True
    )
    ort_model.save_pretrained(output_dir / "ner_onnx_export")

    # Copy model and tokenizer
    exported = output_dir / "ner_onnx_export" / "model.onnx"
    shutil.copy(exported, output_dir / "ner.onnx")

    tokenizer_out = output_dir / "ner-tokenizer"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(str(tokenizer_out))

    print(f"NER ONNX model exported to {output_dir / 'ner.onnx'}")
    print(f"Tokenizer saved to {tokenizer_out}")

    # Save label map
    label_map = {i: l for i, l in enumerate(LABEL_LIST)}
    import json
    with open(output_dir / "ner-labels.json", "w") as f:
        json.dump(label_map, f, indent=2)


def main():
    models_dir = Path(__file__).parent.parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    print("Loading and preparing CoNLL-2003...")
    tokenized_ds, tokenizer = load_and_prepare_data()

    print("Training DistilBERT for NER...")
    model_path = train(tokenized_ds, tokenizer)

    print("Exporting to ONNX...")
    export_onnx(model_path, models_dir)

    print("Done!")


if __name__ == "__main__":
    main()
