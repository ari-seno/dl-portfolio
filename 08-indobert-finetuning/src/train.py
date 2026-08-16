# src/train.py

import argparse

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from src.dataset import (
    DEFAULT_MODEL_NAME,
    LABEL2ID,
    build_dataset_dict,
    compute_class_weights,
    get_tokenizer,
    tokenize_dataset,
)
from src.model import build_model


def get_device():
    """Select the best available device: XPU (Intel Arc) > CUDA > CPU."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1_macro": f1, "precision": precision, "recall": recall}


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune IndoBERT variants on SmSA.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--class-weights", action="store_true", help="Balanced loss weights")
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--tag", type=str, default="", help="Optional run tag for output naming")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")
    print(f"Config: {vars(args)}")

    raw = build_dataset_dict()
    tokenizer = get_tokenizer(args.model_name)
    tokenized = tokenize_dataset(raw, tokenizer, max_length=args.max_length)
    model = build_model(args.model_name).to(device)

    class_weights = None
    if args.class_weights:
        class_weights = compute_class_weights(tokenized["train"])
        print(f"Class weights: {class_weights}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    output_dir = args.output_dir
    if args.tag:
        output_dir = f"{args.output_dir}/{args.tag}"
        import os
        os.makedirs(output_dir, exist_ok=True)
    best_model_dir = f"{output_dir}/best_model"

    total_steps = args.num_epochs * len(tokenized["train"]) // args.batch_size
    warmup_steps = int(total_steps * args.warmup_ratio)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        use_cpu=(device.type == "cpu"),
        save_total_limit=3,
    )

    class WeightedTrainer(Trainer):
        def __init__(self, *targs, class_weights=None, **kwargs):
            super().__init__(*targs, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    trainer_cls = WeightedTrainer if args.class_weights else Trainer

    callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]

    trainer_kwargs = {}
    if args.class_weights:
        trainer_kwargs["class_weights"] = torch.tensor(
            class_weights, device=device, dtype=torch.float32
        )

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        **trainer_kwargs,
    )

    trainer.train()
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    test_results = trainer.evaluate(tokenized["test"])
    print("Test results:", test_results)

    # Per-class report on the best model
    preds = trainer.predict(tokenized["test"]).predictions
    preds = np.argmax(preds, axis=-1)
    labels = np.array(tokenized["test"]["label"])
    per_class = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
    for cls_name, i in LABEL2ID.items():
        print(
            f"{cls_name}: precision={per_class[0][i]:.4f}, recall={per_class[1][i]:.4f}, f1={per_class[2][i]:.4f}"
        )


if __name__ == "__main__":
    main()