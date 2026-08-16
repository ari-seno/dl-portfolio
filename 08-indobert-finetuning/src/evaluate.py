# src/evaluate.py

import argparse

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from src.dataset import LABEL2ID, build_dataset_dict, get_tokenizer, tokenize_dataset
from src.train import compute_metrics, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model on the test set.")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["test", "validation"])
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = get_tokenizer(args.model_dir)
    raw = build_dataset_dict()
    tokenized = tokenize_dataset(raw, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)

    eval_args = TrainingArguments(
        output_dir="outputs/eval_tmp",
        per_device_eval_batch_size=32,
        report_to="none",
        use_cpu=(device.type == "cpu"),
    )
    trainer = Trainer(model=model, args=eval_args, compute_metrics=compute_metrics)
    results = trainer.evaluate(tokenized[args.split])
    print("Metrics:", results)

    preds = trainer.predict(tokenized[args.split]).predictions
    preds = np.argmax(preds, axis=-1)
    labels = np.array(tokenized[args.split]["label"])

    print("\nClassification report:")
    print(classification_report(labels, preds, labels=list(LABEL2ID.values()), target_names=list(LABEL2ID.keys()), digits=4, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(labels, preds))
    print("Accuracy:", accuracy_score(labels, preds))


if __name__ == "__main__":
    main()