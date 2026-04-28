# ==============================================================================
# predict.py — Pipeline de Inferencia
# ==============================================================================
#
# Responsabilidades:
#   - Cargar los artefactos serializados (modelo, vectorizador, encoder).
#   - Exponer una función de clasificación end-to-end que recibe un título
#     y un contenido en texto crudo y devuelve la categoría predicha.
#
# Este módulo es INDEPENDIENTE del dataset original y del entrenamiento.
# Solo necesita los artefactos en la carpeta configurada y spaCy instalado.
#
# ==============================================================================

import os

import numpy as np
import joblib
import xgboost as xgb

from .preprocessing import regex_clean, spacy_clean_batch, _load_spacy_model


# ==============================================================================
# CARGA DE ARTEFACTOS
# ==============================================================================

def load_inference_pipeline(artifacts_dir):
    """
    Carga todos los artefactos necesarios para inferencia:
        - Modelo XGBoost
        - Vectorizador TF-IDF
        - LabelEncoder

    Retorna un diccionario con las claves: 'model', 'vectorizer', 'encoder', 'nlp'.
    """
    # Buscar modelo: primero .ubj (formato nuevo), luego .pkl (formato legacy del notebook)
    model_path_ubj = os.path.join(artifacts_dir, "xgboost_topic_classifier.ubj")
    model_path_pkl = os.path.join(artifacts_dir, "xgboost_topic_classifier.pkl")

    if os.path.isfile(model_path_ubj):
        model_path = model_path_ubj
    elif os.path.isfile(model_path_pkl):
        model_path = model_path_pkl
    else:
        raise FileNotFoundError(
            f"[Predict] No se encontró el modelo XGBoost en:\n"
            f"  - {model_path_ubj}\n"
            f"  - {model_path_pkl}\n"
            f"  Ejecuta primero los comandos 'preprocess' y 'train'."
        )

    vec_path = os.path.join(artifacts_dir, "tfidf_vectorizer.pkl")
    enc_path = os.path.join(artifacts_dir, "label_encoder.pkl")

    # Verificar que existen vectorizer y encoder
    for path, name in [
        (vec_path, "Vectorizador TF-IDF"),
        (enc_path, "LabelEncoder"),
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"[Predict] No se encontró el artefacto '{name}' en: {path}\n"
                f"  Ejecuta primero los comandos 'preprocess' y 'train'."
            )

    print("[Predict] Cargando artefactos de inferencia...")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print(f"  - Modelo cargado: {model_path}")

    vectorizer = joblib.load(vec_path)
    print(f"  - Vectorizador cargado: {vec_path}")

    encoder = joblib.load(enc_path)
    print(f"  - LabelEncoder cargado: {enc_path}")

    nlp = _load_spacy_model()

    return {
        'model': model,
        'vectorizer': vectorizer,
        'encoder': encoder,
        'nlp': nlp,
    }


# ==============================================================================
# CLASIFICACIÓN DE UNA NOTICIA
# ==============================================================================

def classify_news(title, content, pipeline):
    """
    Pipeline completo de inferencia para clasificar una noticia en crudo.

    Parámetros:
        title:    Título de la noticia (str).
        content:  Contenido/cuerpo de la noticia (str).
        pipeline: Diccionario devuelto por load_inference_pipeline().

    Retorna:
        La categoría predicha como string (ej: 'Economía', 'Deportes'...).
    """
    model = pipeline['model']
    vectorizer = pipeline['vectorizer']
    encoder = pipeline['encoder']
    nlp = pipeline['nlp']

    # Paso 1: Unir título y contenido
    texto_combinado = title + " " + content

    # Paso 2: Limpieza con Regex
    texto_regex = regex_clean(texto_combinado)

    # Paso 3: Procesamiento NLP con spaCy
    texto_nlp = spacy_clean_batch([texto_regex], nlp, batch_size=1)[0]

    # Paso 4: Vectorización TF-IDF
    texto_tfidf = vectorizer.transform([texto_nlp])

    # Paso 5: Adaptar formato (float32 + ordenar índices)
    texto_tfidf = texto_tfidf.astype(np.float32)
    texto_tfidf.sort_indices()

    # Paso 6: Predicción
    clase_numerica = model.predict(texto_tfidf)

    # Paso 7: Decodificar a texto
    categoria_final = encoder.inverse_transform(clase_numerica)[0]

    return categoria_final
