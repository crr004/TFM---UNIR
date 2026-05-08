from collections import Counter
from html import unescape
from pathlib import Path
import re
import unicodedata

import pandas as pd
import spacy
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "./Bin/Data_Bin_Classifier.csv"
OUTPUT_PATH = SCRIPT_DIR / "dataset_preprocesado_binario.csv"

KEEP_SHORT_TOKENS = {"no", "si", "sí", "ya", "ok", "xa", "eh"}
NEGATION_TOKENS = {"no", "nunca", "jamas", "jamás", "sin", "tampoco"}
BOILERPLATE_PATTERNS = [
    r"\bleer mas\b",
    r"\bleer más\b",
    r"\bfuente\s*:\s*.*$",
    r"\bsuscribete\b",
    r"\bsuscríbete\b",
    r"\bcompartir\b",
    r"\bcomentarios?\b",
]

stats = Counter()


# -------------------------
# CARGAR MODELO
# -------------------------
def load_spacy_model():
    """Carga spaCy en modo ligero y asegura un límite de longitud razonable."""
    try:
        nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
    except OSError as exc:
        raise RuntimeError(
            "No se encontró el modelo spaCy 'es_core_news_sm'. Instálalo antes de ejecutar el script."
        ) from exc

    nlp.max_length = 2_000_000
    return nlp


# -------------------------
# UTILIDADES DE NORMALIZACIÓN
# -------------------------
def normalize_unicode(text):
    text = unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", " ")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text


def build_full_text(df):
    if "Title" not in df.columns or "Content" not in df.columns:
        raise KeyError("El dataset debe contener las columnas 'Title' y 'Content'.")

    title = df["Title"].fillna("").astype(str).str.strip()
    content = df["Content"].fillna("").astype(str).str.strip()
    full_text = (title + " " + content).str.replace(r"\s+", " ", regex=True).str.strip()
    return full_text


def build_features(text_series):
    features = pd.DataFrame(index=text_series.index)
    features["num_chars"] = text_series.str.len()
    features["num_words"] = text_series.str.split().str.len().fillna(0).astype(int)
    features["num_exclamaciones"] = text_series.str.count("!")
    features["num_interrogaciones"] = text_series.str.count(r"\?")
    features["num_urls"] = text_series.str.count(r"https?://\S+|www\.\S+")
    features["num_emails"] = text_series.str.count(r"\b\S+@\S+\.\S+\b")
    features["num_mayusculas"] = text_series.str.count(r"[A-ZÁÉÍÓÚÑ]")
    features["num_digitos"] = text_series.str.count(r"\d")
    features["num_ellipsis"] = text_series.str.count(r"\.\.\.|…")
    features["num_quotes"] = text_series.str.count(r'["“”‘’\']')
    return features


def clean_for_ml(text):
    """Limpieza fuerte para ML clásico (SVM, RF, LR, NB con TF-IDF).
    Normaliza, tokeniza a palabras explícitas, reemplaza números/URLs.
    """
    if pd.isna(text):
        stats["vacios"] += 1
        return ""

    text = str(text).strip()
    if not text:
        stats["vacios"] += 1
        return ""

    original = text
    text = normalize_unicode(text)
    text = text.lower()

    # Cuenta señales antes de reemplazarlas
    url_matches = re.findall(r"https?://\S+|www\.\S+", text)
    email_matches = re.findall(r"\b\S+@\S+\.\S+\b", text)
    num_matches = re.findall(r"\d+", text)

    stats["urls"] += len(url_matches)
    stats["emails"] += len(email_matches)
    stats["numeros"] += len(num_matches)

    # Reemplaza con tokens explícitos
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"\b\S+@\S+\.\S+\b", " email ", text)
    text = re.sub(r"\d+([\.,]\d+)?", " numero ", text)

    # Normaliza repeticiones (holaaaa -> holaa)
    collapsed = re.sub(r"(.)\1{2,}", r"\1\1", text)
    if collapsed != text:
        stats["repeticiones_normalizadas"] += 1
    text = collapsed

    # Elimina boilerplate
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.MULTILINE)

    # Limpia caracteres especiales
    text = re.sub(r"[^a-záéíóúñü\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text != original.lower():
        stats["textos_normalizados"] += 1

    return text


def clean_for_dl(text):
    """Limpieza suave para modelos de Deep Learning (LSTM, CNN, RNN, etc).
    Normaliza Unicode, minúsculas, reemplaza números/URLs con tokens, pero mantiene más contexto.
    Sin lematización, sin eliminar stopwords agresivamente. Ideal para preservar información semántica.
    """
    if pd.isna(text):
        return ""

    text = str(text).strip()
    if not text:
        return ""

    text = normalize_unicode(text)
    text = text.lower()

    # Reemplaza URLs y números con tokens
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"\b\S+@\S+\.\S+\b", " email ", text)
    text = re.sub(r"\d+([\.,]\d+)?", " numero ", text)

    # Normaliza repeticiones
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Elimina boilerplate
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.MULTILINE)

    # Mantiene más caracteres (guiones, comillas, etc.)
    text = re.sub(r"[^a-záéíóúñü\s\-''\".,;:!?()—–]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_for_beto(text):
    """Limpieza mínima para BETO/transformers.
    Solo normalización Unicode, espacios, HTML. Mantiene casi todo el texto original.
    """
    if pd.isna(text):
        return ""

    text = str(text).strip()
    if not text:
        return ""

    text = normalize_unicode(text)
    # BETO puede trabajar con mayúsculas, no lowercaseamos
    # Solo limpia espacios múltiples
    text = re.sub(r"\s+", " ", text).strip()

    return text


def lemmatize_and_filter(texts, nlp):
    """Lematiza y filtra agresivamente para ML clásico (text_ml).
    Elimina stopwords (excepto negaciones), tokens cortos, caracteres no-alfa.
    """
    cleaned_docs = []
    docs = nlp.pipe(texts, batch_size=256, n_process=1)

    for doc in tqdm(docs, total=len(texts), desc="Lematizando"):
        tokens = []
        for token in doc:
            token_text = token.text.strip()
            lemma = token.lemma_.strip().lower()

            if token.is_space or token.is_punct or token.is_quote:
                stats["puntuacion"] += 1
                continue

            if not token_text:
                continue

            if token.like_num or lemma == "numero":
                tokens.append("numero")
                continue

            if token.like_url or lemma == "url":
                tokens.append("url")
                continue

            if token.like_email or lemma == "email":
                tokens.append("email")
                continue

            if lemma in NEGATION_TOKENS:
                tokens.append(lemma)
                continue

            if token.is_stop and lemma not in NEGATION_TOKENS:
                stats["stopwords"] += 1
                continue

            if len(lemma) < 3 and lemma not in KEEP_SHORT_TOKENS:
                stats["tokens_cortos"] += 1
                continue

            if not any(ch.isalpha() for ch in lemma):
                stats["no_alpha"] += 1
                continue

            tokens.append(lemma)

        cleaned_text = " ".join(tokens).strip()
        if not cleaned_text:
            stats["vacios_post"] += 1
        cleaned_docs.append(cleaned_text)

    return cleaned_docs


# -------------------------
# PIPELINE PRINCIPAL
# -------------------------
def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {INPUT_PATH}")

    print(f"Cargando dataset desde: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, encoding="utf-8")
    print("Columnas del dataset:")
    print(df.columns.tolist())

    df = df.copy()
    df["full_text"] = build_full_text(df)
    df["full_text"] = (
        df["full_text"]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df[df["full_text"].ne("")].copy()

    print("\nExtrayendo características estructurales...")
    features = build_features(df["full_text"])
    df = pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)

    print("\nNormalizando texto para deduplicación...")
    df["text_ml_temp"] = df["full_text"].map(clean_for_ml)

    before = len(df)
    df["dedup_key"] = (
        df["text_ml_temp"].str.replace(r"\s+", " ", regex=True).str.strip()
    )

    if "State" in df.columns:
        conflict_mask = df.groupby("dedup_key")["State"].transform("nunique") > 1
        conflict_rows = int(conflict_mask.sum())
        if conflict_rows:
            stats["conflictos_etiqueta"] = conflict_rows
        df = df[~conflict_mask].copy()

    df = df.drop_duplicates(subset=["dedup_key"])
    print(f"Duplicados/conflictos eliminados: {before - len(df)}")

    print("\nCargando spaCy...")
    nlp = load_spacy_model()

    print("\nProcesando textos para diferentes modelos...")
    df["text_ml"] = lemmatize_and_filter(df["text_ml_temp"].tolist(), nlp)
    df["text_dl"] = df["full_text"].map(clean_for_dl)
    df["text_beto"] = df["full_text"].map(clean_for_beto)
    df = df.drop(columns=["text_ml_temp"])

    if "State" not in df.columns:
        raise KeyError(
            "El dataset debe contener la columna 'State' para crear la etiqueta binaria."
        )

    print("\nValores únicos en State:")
    print(df["State"].dropna().unique())
    state_norm = df["State"].astype(str).str.strip().str.lower()

    label_map = {
        "fake": 1,
        "false": 1,
        "1": 1,
        "no fake": 0,
        "real": 0,
        "true": 0,
        "0": 0,
    }

    df["label"] = state_norm.map(label_map)

    if df["label"].isna().any():
        print("Valores de State no mapeados:")
        print(df.loc[df["label"].isna(), "State"].value_counts())
        raise ValueError("Hay etiquetas sin mapear en State.")

    # Filtra textos que quedaron vacíos tras el procesamiento ML.
    valid_mask = df["text_ml"].str.len() > 0
    removed_empty = int((~valid_mask).sum())
    if removed_empty:
        stats["eliminados_vacios"] = removed_empty
        df = df[valid_mask].copy()

    # Orden final: columnas originales + features + salidas del pipeline.
    preferred_columns = [
        "ID",
        "Title",
        "Content",
        "State",
        "Dataset",
        "label",
        "full_text",
        "text_ml",
        "text_dl",
        "text_beto",
        "num_chars",
        "num_words",
        "num_exclamaciones",
        "num_interrogaciones",
        "num_urls",
        "num_emails",
        "num_mayusculas",
        "num_digitos",
        "num_ellipsis",
        "num_quotes",
    ]
    # Limpia columna de deduplicación
    df = df.drop(columns=["dedup_key"], errors="ignore")
    remaining_columns = [col for col in df.columns if col not in preferred_columns]
    df = df[
        [col for col in preferred_columns if col in df.columns] + remaining_columns
    ].copy()

    print("\nESTADISTICAS DEL PREPROCESADO:\n")
    for key, value in stats.most_common():
        print(f"{key}: {value}")

    print("\nCOLUMNAS GENERADAS:")
    print("  - full_text: Texto original (Title + Content)")
    print("  - text_ml: Para SVM, RF, LR, NB + TF-IDF (lematizado, sin stopwords)")
    print("  - text_dl: Para LSTM, CNN, RNN (sin lematizar, preserva contexto)")
    print("  - text_beto: Para BETO/Transformers (minima limpieza)")
    print("  - 10 features estructurales (URLs, exclamaciones, mayusculas, etc.)")

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nPreprocesado completado correctamente")
    print(f"Archivo guardado en: {OUTPUT_PATH}")
    print(f"Registros finales: {len(df)}")


if __name__ == "__main__":
    main()
