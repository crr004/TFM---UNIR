import json
from pathlib import Path

import pandas as pd

from .config import ensure_output_dirs
from .data_utils import load_binary_dataset, stratified_splits
from .xai import BetoExplainer


def _series_stats(series: pd.Series):
    clean = series.dropna()
    if clean.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "p25": None,
            "p75": None,
            "min": None,
            "max": None,
        }

    return {
        "count": int(clean.shape[0]),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
        "median": float(clean.median()),
        "p25": float(clean.quantile(0.25)),
        "p75": float(clean.quantile(0.75)),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def _build_faithfulness_summary(df: pd.DataFrame):
    metrics = [
        "mean_comprehensiveness",
        "mean_sufficiency",
        "deletion_auc",
        "insertion_auc",
        "base_pred_prob",
        "n_words",
    ]

    overall = {metric: _series_stats(df[metric]) for metric in metrics}

    by_class = {}
    for pred_class, group in df.groupby("predicted_class"):
        by_class[str(pred_class)] = {
            metric: _series_stats(group[metric]) for metric in metrics
        }

    return {
        "n_samples": int(df.shape[0]),
        "overall": overall,
        "by_predicted_class": by_class,
    }


def run_explanations(
    model_dir: Path,
    output_dir: Path,
    dataset_path: Path | None = None,
    text: str | None = None,
    n_samples: int = 3,
    faithfulness_scope: str = "sample",
    faithfulness_max_samples: int | None = None,
    random_state: int = 42,
):
    dirs = ensure_output_dirs(Path(output_dir))
    explainer = BetoExplainer(model_dir)

    if text:
        sample_texts = [text]
    else:
        if dataset_path is None:
            raise ValueError(
                "Si no se pasa texto directo, dataset_path es obligatorio."
            )
        df = load_binary_dataset(dataset_path)
        _, _, test_df = stratified_splits(df)

        if faithfulness_scope == "full_test":
            work_df = test_df.copy()
            if faithfulness_max_samples is not None and faithfulness_max_samples > 0:
                work_df = work_df.sample(
                    n=min(int(faithfulness_max_samples), len(work_df)),
                    random_state=random_state,
                )
            sample_texts = work_df["text"].tolist()
        else:
            sample_texts = test_df["text"].head(n_samples).tolist()

    rows = []
    for i, sample in enumerate(sample_texts, start=1):
        lime_path = dirs["explanations"] / f"lime_sample_{i}.html"
        ig_path = dirs["explanations"] / f"ig_sample_{i}.json"
        faithfulness_path = dirs["explanations"] / f"faithfulness_sample_{i}.json"

        lime_result = explainer.explain_lime(sample, lime_path)
        ig_result = explainer.explain_integrated_gradients(sample, ig_path)
        faith_result = explainer.explain_faithfulness(sample, faithfulness_path)

        rows.append(
            {
                "sample_id": i,
                "text": sample,
                "lime_html": lime_result["path"],
                "ig_json": ig_result["path"],
                "faithfulness_json": faith_result["path"],
                "predicted_class": ig_result["predicted_class"],
                "mean_comprehensiveness": faith_result["mean_comprehensiveness"],
                "mean_sufficiency": faith_result["mean_sufficiency"],
                "deletion_auc": faith_result["deletion_auc"],
                "insertion_auc": faith_result["insertion_auc"],
                "n_words": faith_result["n_words"],
                "base_pred_prob": faith_result["base_pred_prob"],
            }
        )

    shap_path = dirs["explanations"] / "shap_summary.json"
    explainer.explain_shap(
        sample_texts, shap_path, max_samples=min(50, len(sample_texts))
    )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(dirs["explanations"] / "explanations_index.csv", index=False)

    faithfulness_summary = _build_faithfulness_summary(summary_df)
    faithfulness_summary["scope"] = faithfulness_scope
    faithfulness_summary["max_samples"] = faithfulness_max_samples

    faithfulness_summary_path = (
        dirs["explanations"] / "faithfulness_global_summary.json"
    )
    with open(faithfulness_summary_path, "w", encoding="utf-8") as f:
        json.dump(faithfulness_summary, f, indent=2, ensure_ascii=False)

    print("[OK] Explicaciones generadas")
    print(f"- Carpeta: {dirs['explanations']}")
    return {
        "index": str(dirs["explanations"] / "explanations_index.csv"),
        "shap": str(shap_path),
        "faithfulness_global": str(faithfulness_summary_path),
    }
