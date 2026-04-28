# ==============================================================================
# preprocessing.py — Carga, Limpieza y Preparación de Datos
# ==============================================================================
#
# Responsabilidades:
#   - Cargar el dataset CSV de noticias.
#   - Agrupar los 212 tópicos originales en 9 categorías principales.
#   - Limpiar el texto con expresiones regulares.
#   - Procesar el texto con spaCy (lematización + eliminación de stopwords).
#   - Generar representación TF-IDF y dividir en train/test.
#   - Serializar los artefactos (vectorizador TF-IDF, LabelEncoder).
#
# ==============================================================================

import os
import re
import joblib

import numpy as np
import pandas as pd
import spacy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm.auto import tqdm


# ==============================================================================
# 1. CARGA DEL MODELO SPACY
# ==============================================================================

def _load_spacy_model(model_name="es_core_news_sm"):
    """
    Carga el modelo spaCy especificado.
    Si no está instalado, lo descarga automáticamente.
    """
    try:
        nlp = spacy.load(model_name)
        print(f"[spaCy] Modelo '{model_name}' cargado correctamente.")
    except OSError:
        print(f"[spaCy] El modelo '{model_name}' no está instalado. Descargando...")
        from spacy.cli import download as spacy_download
        spacy_download(model_name)
        nlp = spacy.load(model_name)
        print(f"[spaCy] Modelo '{model_name}' descargado y cargado correctamente.")

    nlp.max_length = 2_000_000
    return nlp


# ==============================================================================
# 2. MAPEO DE TÓPICOS
# ==============================================================================

def map_topic(topic):
    """
    Agrupa los 212 tópicos originales del dataset MLSUM en 9 categorías
    principales para reducir la dimensionalidad del problema de clasificación.
    """
    topic = str(topic).lower()

    if any(k in topic for k in [
        'deporte', 'futbol', 'baloncesto', 'olimpico',
        'champions', 'eurocopa', 'adrenalina'
    ]):
        return 'Deportes'

    elif any(k in topic for k in [
        'politica', 'congreso', 'opinion', 'genova', 'eldebate'
    ]):
        return 'Política'

    elif any(k in topic for k in [
        'economia', 'negocio', 'empleo', 'vivienda',
        'finanzas', 'ahorro', 'mercado'
    ]):
        return 'Economía'

    elif any(k in topic for k in [
        'tecnologia', 'ciencia', 'motor', 'ciberpais', 'techie', 'digital'
    ]):
        return 'Ciencia y Tecnología'

    elif any(k in topic for k in [
        'cultura', 'cine', 'television', 'babelia', 'tentaciones',
        'eps', 'icon', 'vinetas', 'arte', 'album'
    ]):
        return 'Cultura y Entretenimiento'

    elif any(k in topic for k in [
        'internacional', 'america', 'mexico', 'estados_unidos',
        'colombia', 'argentina', 'mundo'
    ]):
        return 'Internacional'

    elif any(k in topic for k in [
        'sociedad', 'gente', 'estilo', 'viajero', 'salud',
        'educacion', 'mamas', 'buenavida', 'planeta_futuro'
    ]):
        return 'Sociedad y Estilo de Vida'

    elif any(k in topic for k in [
        'ccaa', 'catalunya', 'madrid', 'valencia', 'paisvasco',
        'galicia', 'andalucia', 'espana', 'cvalenciana', 'cat '
    ]) or 'elpais actualidad' in topic:
        return 'España / Local'

    else:
        return 'Otros'


# ==============================================================================
# 3. LIMPIEZA DE TEXTO
# ==============================================================================

def regex_clean(text):
    """Limpieza básica con expresiones regulares."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-záéíóúñü\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def spacy_clean_batch(texts, nlp, batch_size=256):
    """
    Procesamiento avanzado con spaCy (lematización + filtrado de stopwords).
    Utiliza nlp.pipe para procesamiento en lotes eficiente.
    """
    cleaned_docs = []
    for doc in tqdm(
        nlp.pipe(texts, batch_size=batch_size, disable=["parser", "ner"]),
        total=len(texts),
        desc="Procesando NLP"
    ):
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space
        ]
        cleaned_docs.append(" ".join(tokens))
    return cleaned_docs


# ==============================================================================
# 4. PIPELINE COMPLETO DE PREPROCESAMIENTO
# ==============================================================================

def load_and_prepare_data(csv_path):
    """
    Carga el dataset CSV, aplica agrupación de tópicos,
    limpieza de nulos/duplicados, y procesamiento NLP completo.

    Retorna el DataFrame con las columnas:
        - Title, Content, Topic_Grouped, Cleaned_Content
    """
    print(f"[Preprocesamiento] Cargando dataset desde: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[Preprocesamiento] Registros iniciales: {len(df)}")

    # 1. Agrupación de tópicos
    df['Topic_Grouped'] = df['Topic'].apply(map_topic)

    # 2. Eliminación de nulos y vacíos en campos críticos
    df = df.dropna(subset=['Title', 'Content', 'Topic_Grouped']).copy()
    df = df[(df['Title'].str.strip() != '') & (df['Content'].str.strip() != '')]

    # 3. Eliminación de duplicados
    initial_count = len(df)
    df = df.drop_duplicates(subset=['Title', 'Content'])
    print(f"[Preprocesamiento] Duplicados eliminados: {initial_count - len(df)}")

    # 4. Unión de Título + Contenido
    df['Texto_Completo'] = df['Title'] + " " + df['Content']

    # 5. Limpieza con Regex
    print("[Preprocesamiento] Aplicando limpieza Regex...")
    tqdm.pandas(desc="Limpieza Regex")
    df['Regex_Content'] = df['Texto_Completo'].progress_apply(regex_clean)

    # 6. Procesamiento con spaCy
    print("[Preprocesamiento] Aplicando procesamiento spaCy (lematización + stopwords)...")
    nlp = _load_spacy_model()
    texts_to_process = df['Regex_Content'].tolist()
    df['Cleaned_Content'] = spacy_clean_batch(texts_to_process, nlp)

    # 7. Limpieza de columnas intermedias
    df = df.drop(columns=['Regex_Content', 'Texto_Completo'])
    df = df.reset_index(drop=True)

    print(f"[Preprocesamiento] Registros finales: {len(df)}")
    print(f"[Preprocesamiento] Distribución de clases:")
    print(df['Topic_Grouped'].value_counts())

    return df


# ==============================================================================
# 5. TF-IDF + SPLIT + GUARDADO DE ARTEFACTOS
# ==============================================================================

def build_tfidf_and_split(df, max_features=30000, test_size=0.2, random_state=42):
    """
    1. Codifica las etiquetas con LabelEncoder.
    2. Divide el dataset en train/test estratificado.
    3. Entrena TfidfVectorizer sobre el set de entrenamiento.
    4. Transforma ambos conjuntos.

    Retorna:
        X_train_tfidf, X_test_tfidf, y_train, y_test,
        vectorizer, label_encoder
    """
    # Label Encoding
    print("[Preprocesamiento] Codificando etiquetas...")
    label_encoder = LabelEncoder()
    df['Target'] = label_encoder.fit_transform(df['Topic_Grouped'])

    print("[Preprocesamiento] Mapeo de clases:")
    for idx, cls in enumerate(label_encoder.classes_):
        print(f"  {idx}: {cls}")

    # Train-Test Split
    print(f"\n[Preprocesamiento] Dividiendo dataset ({int((1-test_size)*100)}% train, {int(test_size*100)}% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['Cleaned_Content'],
        df['Target'],
        test_size=test_size,
        random_state=random_state,
        stratify=df['Target']
    )
    print(f"[Preprocesamiento] Set de Entrenamiento: {len(X_train)} muestras")
    print(f"[Preprocesamiento] Set de Prueba: {len(X_test)} muestras")

    # TF-IDF
    print(f"\n[Preprocesamiento] Vectorizando texto con TF-IDF (max_features={max_features})...")
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Conversión a float32 para compatibilidad con GPU
    X_train_tfidf = X_train_tfidf.astype(np.float32)
    X_train_tfidf.sort_indices()
    X_test_tfidf = X_test_tfidf.astype(np.float32)
    X_test_tfidf.sort_indices()

    print(f"[Preprocesamiento] Dimensiones TF-IDF (train): {X_train_tfidf.shape}")
    print(f"[Preprocesamiento] Dimensiones TF-IDF (test):  {X_test_tfidf.shape}")

    return X_train_tfidf, X_test_tfidf, y_train.values, y_test.values, vectorizer, label_encoder


def save_artifacts(vectorizer, label_encoder, output_dir):
    """Serializa el vectorizador TF-IDF y el LabelEncoder en disco."""
    os.makedirs(output_dir, exist_ok=True)

    vec_path = os.path.join(output_dir, "tfidf_vectorizer.pkl")
    enc_path = os.path.join(output_dir, "label_encoder.pkl")

    joblib.dump(vectorizer, vec_path)
    joblib.dump(label_encoder, enc_path)

    print(f"[Preprocesamiento] Artefactos guardados en '{output_dir}':")
    print(f"  - {vec_path}")
    print(f"  - {enc_path}")
