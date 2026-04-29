"""
Layer 2: Fine-tune DistilBERT for 3-class sentence sentiment classification
and export to ONNX.

Uses SST-2 (Stanford Sentiment Treebank) mapped to 3 classes:
  0 = NEGATIVE, 1 = NEUTRAL, 2 = POSITIVE

The model is then quantized (dynamic int8) for faster inference.
"""

from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
import evaluate
import numpy as np


MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 3  # negative, neutral, positive
MAX_LENGTH = 128


def load_and_prepare_data():
    """Load SST-2 and map to 3 classes."""
    ds = load_dataset("glue", "sst2")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    # SST-2 is binary (0=neg, 1=pos). Remap: 0->0 (neg), 1->2 (pos).
    # Need to cast label to int first since it's a ClassLabel feature.
    from datasets import Features, Value
    for split in ds:
        ds[split] = ds[split].map(
            lambda ex: {"label": ex["label"] * 2},
            features=Features({
                **{k: v for k, v in ds[split].features.items() if k != "label"},
                "label": Value("int64"),
            }),
        )
    tokenized = ds.map(tokenize, batched=True)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return tokenized, tokenizer


def train(tokenized_ds, tokenizer):
    """Fine-tune DistilBERT."""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=predictions, references=labels)

    output_dir = Path(__file__).parent / "checkpoint"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "best"))
    tokenizer.save_pretrained(str(output_dir / "best"))

    return output_dir / "best"


def export_onnx(model_path: Path, output_dir: Path):
    """Export to ONNX and quantize."""
    # Export to ONNX using Optimum
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        model_path, export=True
    )
    onnx_path = output_dir / "sentence-sentiment.onnx"
    ort_model.save_pretrained(output_dir / "onnx_export")

    # Copy the ONNX model
    import shutil
    exported = output_dir / "onnx_export" / "model.onnx"
    shutil.copy(exported, onnx_path)

    # Save tokenizer for Java
    tokenizer_out = output_dir / "sentence-sentiment-tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(str(tokenizer_out))

    # Quantize (dynamic int8)
    quantizer = ORTQuantizer.from_pretrained(output_dir / "onnx_export")
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=output_dir / "quantized", quantization_config=qconfig)

    quantized_path = output_dir / "quantized" / "model_quantized.onnx"
    if quantized_path.exists():
        shutil.copy(quantized_path, output_dir / "sentence-sentiment-quantized.onnx")
        print(f"Quantized model: {quantized_path.stat().st_size / 1024 / 1024:.1f} MB")

    print(f"ONNX model exported to {onnx_path}")
    print(f"Tokenizer saved to {tokenizer_out}")

    return onnx_path


def main():
    models_dir = Path(__file__).parent.parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    print("Loading and preparing data...")
    tokenized_ds, tokenizer = load_and_prepare_data()

    print("Training DistilBERT for sentence sentiment...")
    model_path = train(tokenized_ds, tokenizer)

    print("Exporting to ONNX...")
    export_onnx(model_path, models_dir)

    print("Done!")


if __name__ == "__main__":
    main()
