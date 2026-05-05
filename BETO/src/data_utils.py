import re
import unicodedata
from html import unescape
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_MAP = {
    "fake": 1,
    "false": 1,
    "1": 1,
    "true": 0,
    "real": 0,
    "no fake": 0,
    "0": 0,
}


def minimal_beto_clean(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    text = unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ").replace("\u200b", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _pick_text_column(df: pd.DataFrame) -> str:
    preferred = ["text", "text_beto", "full_text", "Content"]
    for col in preferred:
        if col in df.columns:
            return col

    if "Title" in df.columns and "Content" in df.columns:
        df["full_text"] = (
            (
                df["Title"].fillna("").astype(str).str.strip()
                + " "
                + df["Content"].fillna("").astype(str).str.strip()
            )
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        return "full_text"

    raise KeyError(
        "No se encontró una columna de texto usable. Esperadas: text, text_beto, full_text o Content."
    )


def _normalize_labels(df: pd.DataFrame) -> pd.Series:
    if "label" in df.columns:
        label_series = pd.to_numeric(df["label"], errors="coerce")
        if label_series.notna().all():
            label_series = label_series.astype(int)
            if set(label_series.unique()).issubset({0, 1}):
                return label_series

    if "State" not in df.columns:
        raise KeyError(
            "No existe la columna 'State' y tampoco una columna 'label' válida."
        )

    state = df["State"].astype(str).str.strip().str.lower()
    labels = state.map(LABEL_MAP)

    if labels.isna().any():
        unknown = df.loc[labels.isna(), "State"].value_counts()
        raise ValueError(f"Etiquetas State no mapeadas: {unknown.to_dict()}")

    return labels.astype(int)


def load_binary_dataset(dataset_path: Path) -> pd.DataFrame:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"No se encontró dataset en: {dataset_path}")

    df = pd.read_csv(dataset_path)
    text_col = _pick_text_column(df)

    df = df.copy()
    df["text"] = df[text_col].map(minimal_beto_clean)
    df = df[df["text"].str.len() > 0].copy()
    df["label"] = _normalize_labels(df)

    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df[["text", "label"]]


def stratified_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
):
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )

    val_relative_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        stratify=train_val_df["label"],
        random_state=random_state,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
