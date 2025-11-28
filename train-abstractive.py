"""
train_abstractive.py

Fine-tune a T5 model on the CNN/DailyMail dataset (abisee/cnn_dailymail).
Computes ROUGE-1 and ROUGE-L (precision/recall/f1) and saves metrics to JSON.

Usage example:
    python train_abstractive.py \
        --model_name t5-small \
        --output_dir ./t5_cnn_model \
        --epochs 1 \
        --per_device_train_batch_size 4
"""

import os
import argparse
import json
import numpy as np
from datasets import load_dataset
import evaluate
from transformers import (
    T5TokenizerFast,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="t5-small")
    p.add_argument("--output_dir", type=str, default="./t5_cnn_model")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--max_input_length", type=int, default=512)
    p.add_argument("--max_target_length", type=int, default=150)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def preprocess_examples(examples, tokenizer, max_input_length, max_target_length):
    # prepend task prefix used by T5
    inputs = ["summarize: " + doc for doc in examples["document"]]
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )
    # tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["summary"],
            max_length=max_target_length,
            truncation=True,
            padding="max_length",
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def safe_extract_rouge(result):
    """
    The evaluate.load("rouge").compute(...) can return objects in a few forms
    depending on version. Normalize to dict of floats for precision/recall/fmeasure.
    """
    out = {}
    for k, v in result.items():
        # v may be a dict like {'precision':..., 'recall':..., 'fmeasure':...}
        # or v may be an object with .mid.fmeasure (datasets.metric older API)
        try:
            if isinstance(v, dict) and "precision" in v:
                out[f"{k}_precision"] = float(v["precision"])
                out[f"{k}_recall"] = float(v["recall"])
                out[f"{k}_fmeasure"] = float(v["fmeasure"])
            else:
                # try object style (has .mid)
                mid = getattr(v, "mid", None)
                if mid is not None:
                    out[f"{k}_precision"] = float(mid.precision)
                    out[f"{k}_recall"] = float(mid.recall)
                    out[f"{k}_fmeasure"] = float(mid.fmeasure)
                else:
                    # fallback: try numeric cast
                    out[k] = float(v)
        except Exception:
            out[k] = str(v)
    return out

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load dataset
    logger.info("Loading CNN/DailyMail (abisee/cnn_dailymail)")

    raw = load_dataset("abisee/cnn_dailymail", "3.0.0")

    # Select smaller subsets for faster training
    train_ds = raw["train"].select(range(10000))
    val_ds = raw["validation"].select(range(2000))
    test_ds = raw["test"].select(range(2000))

    # Rename columns
    train_ds = train_ds.rename_columns({"article": "document", "highlights": "summary"})
    val_ds = val_ds.rename_columns({"article": "document", "highlights": "summary"})
    test_ds = test_ds.rename_columns({"article": "document", "highlights": "summary"})

    # 2) Load tokenizer and model
    logger.info(f"Loading model/tokenizer: {args.model_name}")
    tokenizer = T5TokenizerFast.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    # 3) Preprocess (map)
    logger.info("Tokenizing dataset (this can take a while)...")
    preprocess_fn = lambda examples: preprocess_examples(
        examples, tokenizer, min(args.max_input_length, 256), min(args.max_target_length, 100)
    )
    train_tok = train_ds.map(preprocess_fn, batched=True, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(preprocess_fn, batched=True, remove_columns=val_ds.column_names)
    test_tok = test_ds.map(preprocess_fn, batched=True, remove_columns=test_ds.column_names)

    # 4) Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 5) Metrics: ROUGE
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        normalized = safe_extract_rouge(result)
        try:
            pred_lens = [len(tokenizer.encode(p)) for p in decoded_preds]
            normalized["gen_len"] = float(np.mean(pred_lens))
        except Exception:
            normalized["gen_len"] = None
        return normalized

    # 6) Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=False,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=200,
    )

    # 7) Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 8) Train
    logger.info("Starting training...")
    trainer.train()

    # 9) Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.predict(test_tok)
    metrics = test_metrics.metrics if hasattr(test_metrics, "metrics") else test_metrics

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved metrics to {metrics_path}")
    logger.info("Metrics snapshot:\n" + json.dumps(metrics, indent=2))

    # 10) Save model and tokenizer
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Done.")

if __name__ == "__main__":
    main()
