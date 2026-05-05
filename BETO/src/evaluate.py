import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from .config import ensure_output_dirs
from .data_utils import load_binary_dataset, stratified_splits


def _score_map_from_pipeline_sample(sample):
    if isinstance(sample, list):
        return {item["label"].upper(): float(item["score"]) for item in sample}

    if isinstance(sample, dict) and "label" in sample and "score" in sample:
        label = str(sample["label"]).upper()
        score = float(sample["score"])
        if label in {"FAKE", "LABEL_1"}:
            return {"FAKE": score, "REAL": 1.0 - score}
        return {"REAL": score, "FAKE": 1.0 - score}

    return {}


def evaluate_model(model_dir: Path, dataset_path: Path, output_dir: Path):
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    dirs = ensure_output_dirs(output_dir)

    df = load_binary_dataset(dataset_path)
    _, _, test_df = stratified_splits(df)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
    )

    probs_fake = []
    preds = []

    for text in test_df["text"].tolist():
        scores = clf(text, truncation=True, max_length=256)
        sample = scores[0] if isinstance(scores, list) and len(scores) == 1 else scores
        score_map = _score_map_from_pipeline_sample(sample)

        p_fake = score_map.get("FAKE", score_map.get("LABEL_1", 0.0))
        probs_fake.append(p_fake)
        preds.append(1 if p_fake >= 0.5 else 0)

    y_true = test_df["label"].values
    y_pred = np.array(preds)

    report = classification_report(
        y_true,
        y_pred,
        target_names=["REAL", "FAKE"],
        output_dict=True,
        zero_division=0,
    )

    try:
        auc = roc_auc_score(y_true, np.array(probs_fake))
    except ValueError:
        auc = None

    metrics = {
        "classification_report": report,
        "roc_auc": auc,
        "n_test": int(len(test_df)),
    }

    with open(dirs["metrics"] / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    pd.DataFrame(
        {
            "text": test_df["text"],
            "y_true": y_true,
            "y_pred": y_pred,
            "p_fake": probs_fake,
        }
    ).to_csv(dirs["metrics"] / "test_predictions.csv", index=False)

    print("[OK] Evaluación completada")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics
