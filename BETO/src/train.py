import json
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .config import DataConfig, TrainConfig, ensure_output_dirs
from .data_utils import load_binary_dataset, stratified_splits


def _build_hf_dataset(df: pd.DataFrame, tokenizer, max_length: int):
    ds = Dataset.from_pandas(df, preserve_index=False)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    ds = ds.map(tokenize_batch, batched=True)
    return ds


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall": recall_score(labels, preds, average="binary", zero_division=0),
    }


def train_beto(
    dataset_path: Path,
    output_dir: Path,
    train_cfg: TrainConfig | None = None,
    data_cfg: DataConfig | None = None,
):
    train_cfg = train_cfg or TrainConfig()
    data_cfg = data_cfg or DataConfig(dataset_path=dataset_path, output_dir=output_dir)

    dirs = ensure_output_dirs(Path(output_dir))
    set_seed(train_cfg.seed)

    print(f"[DATA] Cargando dataset desde {dataset_path}")
    df = load_binary_dataset(dataset_path)
    train_df, val_df, test_df = stratified_splits(
        df,
        test_size=data_cfg.test_size,
        val_size=data_cfg.val_size,
        random_state=data_cfg.random_state,
    )

    print(f"[DATA] Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(train_cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        train_cfg.model_name,
        num_labels=2,
        id2label={0: "REAL", 1: "FAKE"},
        label2id={"REAL": 0, "FAKE": 1},
        use_safetensors=True,
    )

    train_ds = _build_hf_dataset(train_df, tokenizer, train_cfg.max_length)
    val_ds = _build_hf_dataset(val_df, tokenizer, train_cfg.max_length)

    train_ds = train_ds.remove_columns(
        [
            c
            for c in train_ds.column_names
            if c not in {"input_ids", "attention_mask", "label"}
        ]
    )
    val_ds = val_ds.remove_columns(
        [
            c
            for c in val_ds.column_names
            if c not in {"input_ids", "attention_mask", "label"}
        ]
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_kwargs = {
        "output_dir": str(dirs["model"]),
        "learning_rate": train_cfg.learning_rate,
        "weight_decay": train_cfg.weight_decay,
        "per_device_train_batch_size": train_cfg.train_batch_size,
        "per_device_eval_batch_size": train_cfg.eval_batch_size,
        "num_train_epochs": train_cfg.num_train_epochs,
        "gradient_accumulation_steps": train_cfg.gradient_accumulation_steps,
        "warmup_ratio": train_cfg.warmup_ratio,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "logging_strategy": "steps",
        "logging_steps": 50,
        "report_to": "none",
        "fp16": torch.cuda.is_available(),
        "seed": train_cfg.seed,
    }

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in ta_params:
        training_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in ta_params:
        training_kwargs["eval_strategy"] = "epoch"

    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": data_collator,
        "compute_metrics": _compute_metrics,
    }

    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    print("[TRAIN] Iniciando fine-tuning de BETO...")
    trainer.train()

    eval_metrics = trainer.evaluate()
    print("[EVAL] Métricas de validación:", eval_metrics)

    trainer.save_model(str(dirs["model"]))
    tokenizer.save_pretrained(str(dirs["model"]))

    train_df.to_csv(dirs["base"] / "train_split.csv", index=False)
    val_df.to_csv(dirs["base"] / "val_split.csv", index=False)
    test_df.to_csv(dirs["base"] / "test_split.csv", index=False)

    with open(dirs["metrics"] / "val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2, ensure_ascii=False)

    print(f"[OK] Modelo y artefactos guardados en {dirs['base']}")
    return {
        "model_dir": str(dirs["model"]),
        "splits": {
            "train": str(dirs["base"] / "train_split.csv"),
            "val": str(dirs["base"] / "val_split.csv"),
            "test": str(dirs["base"] / "test_split.csv"),
        },
        "metrics": str(dirs["metrics"] / "val_metrics.json"),
    }
