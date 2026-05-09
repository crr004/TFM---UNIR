# ==============================================================================
# preprocessing.py — Carga, Limpieza y Preparación de Datos
# ==============================================================================
#
# Responsabilidades:
#   - Cargar el dataset CSV de noticias.
#   - Agrupar los 212 tópicos originales en 9 categorías principales.
#   - Filtrar textos no españoles (langdetect).
#   - Limpiar el texto con expresiones regulares mejoradas.
#   - Procesar el texto con spaCy (lematización + stopwords + reemplazo NER).
#   - Generar representación TF-IDF y dividir en train/test.
#   - Serializar los artefactos (vectorizador TF-IDF, LabelEncoder).
#
# ==============================================================================

import os
import re
import unicodedata
import joblib

import numpy as np
import pandas as pd
import spacy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm.auto import tqdm

# Importación condicional de langdetect (puede no estar instalado en inferencia)
try:
    from langdetect import detect as langdetect_detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


# ==============================================================================
# CONSTANTES
# ==============================================================================

# Modelo spaCy a usar (lg = más pesado y preciso para NER)
SPACY_MODEL_NAME = "es_core_news_lg"

# Mapeo de etiquetas NER de spaCy → placeholders para el vocabulario TF-IDF.
# Esto generaliza el vocabulario: "Pedro Sánchez" y "Pablo Casado" se convierten
# ambos en "__PERSONA__", ayudando al modelo a aprender que la presencia de
# personas es relevante para el tópico, sin depender de nombres específicos.
NER_MAP = {
    'PER': '__PERSONA__',
    'ORG': '__ORGANIZACION__',
    'LOC': '__LUGAR__',
    'GPE': '__LUGAR__',          # Entidades geopolíticas (ciudades, países)
    'MISC': '__ENTIDAD__',
}

# Whitelist de stopwords cortas que SÍ aportan significado semántico y no deben
# eliminarse durante el filtrado. Estas palabras, pese a ser stopwords y/o tener
# menos de 3 caracteres, alteran el sentido de la frase.
STOPWORD_WHITELIST = {'no', 'sí', 'ni'}


# ==============================================================================
# 1. CARGA DEL MODELO SPACY
# ==============================================================================

def _load_spacy_model(model_name=None):
    """
    Carga el modelo spaCy especificado.
    Si no está instalado, lo descarga automáticamente.
    Se usa el modelo 'es_core_news_lg' por defecto para maximizar la
    precisión del reconocimiento de entidades nombradas (NER).
    """
    if model_name is None:
        model_name = SPACY_MODEL_NAME

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

    Esto es necesario porque el dataset original tiene etiquetas muy granulares
    (ej: 'ccaa catalunya', 'ccaa madrid', 'ccaa valencia') que representan
    variantes del mismo concepto. Agruparlas mejora el balance de clases y
    la capacidad de generalización del modelo.
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
# 3. DETECCIÓN DE IDIOMA
# ==============================================================================

def _detect_language_safe(text):
    """
    Detecta el idioma de un texto usando langdetect.
    Retorna el código de idioma ('es', 'en', 'fr', etc.) o 'unknown' si falla.

    NOTA: langdetect usa un enfoque probabilístico basado en n-gramas.
    Un texto en español que contenga nombres en inglés (equipos NBA,
    anglicismos, etc.) será correctamente identificado como español
    porque el grueso del texto sigue siendo castellano.
    Solo se filtran textos donde la lengua DOMINANTE no es español.
    """
    try:
        return langdetect_detect(text)
    except Exception:
        return 'unknown'


def filter_non_spanish(df, text_column='Content', min_length=200):
    """
    Filtra registros cuyo texto no esté en español.

    Parámetros:
        df:           DataFrame con los datos.
        text_column:  Columna sobre la que detectar el idioma.
        min_length:   Longitud mínima de texto para aplicar detección.
                      Textos más cortos se mantienen (langdetect es impreciso
                      con textos muy cortos).

    Retorna:
        DataFrame filtrado y el número de registros eliminados.
    """
    if not LANGDETECT_AVAILABLE:
        print("[Preprocesamiento] AVISO: 'langdetect' no instalado. "
              "Se omite el filtro de idioma.")
        return df, 0

    print(f"[Preprocesamiento] Detectando idioma en {len(df)} textos...")
    tqdm.pandas(desc="Detectando idioma")

    # Solo aplicamos detección a textos suficientemente largos
    mask_long = df[text_column].str.len() >= min_length
    languages = pd.Series('es', index=df.index)  # por defecto: español
    languages[mask_long] = df.loc[mask_long, text_column].progress_apply(
        _detect_language_safe
    )

    # Mantener textos en español o con idioma desconocido (beneficio de la duda)
    mask_spanish = languages.isin(['es', 'unknown'])
    n_removed = (~mask_spanish).sum()

    # Reportar distribución de idiomas detectados
    lang_counts = languages[~mask_spanish].value_counts()
    if len(lang_counts) > 0:
        print(f"[Preprocesamiento] Textos no españoles eliminados: {n_removed}")
        print(f"[Preprocesamiento] Distribución de idiomas eliminados:")
        for lang, count in lang_counts.head(10).items():
            print(f"  - {lang}: {count}")
    else:
        print(f"[Preprocesamiento] Todos los textos están en español.")

    return df[mask_spanish].copy(), n_removed


# ==============================================================================
# 4. LIMPIEZA DE TEXTO
# ==============================================================================

def regex_clean(text):
    """
    Limpieza exhaustiva de texto con expresiones regulares.

    Pasos aplicados (en orden):
    1. Conversión a minúsculas.
    2. Normalización de caracteres Unicode (comillas tipográficas, guiones
       largos, etc.) a sus equivalentes ASCII.
    3. Eliminación de URLs (http, https, www y dominios sueltos).
    4. Eliminación de emails.
    5. Eliminación de tags HTML.
    6. Normalización de repeticiones de caracteres (holaaaa → hola).
    7. Eliminación de secuencias numéricas.
    8. Eliminación de caracteres no alfabéticos (conservando tildes y ñ).
    9. Colapso de espacios múltiples.
    """
    text = str(text).lower()

    # Normalización Unicode: convertir caracteres tipográficos a ASCII
    # (comillas "curvas", guiones largos, etc.)
    text = unicodedata.normalize('NFKD', text)
    # Recomponer caracteres descompuestos (mantener tildes como un solo char)
    text = unicodedata.normalize('NFC', text)

    # URLs (incluyendo dominios sueltos como example.com)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+\.\w{2,4}(?:/\S*)?', '', text)  # dominios sueltos

    # Emails
    text = re.sub(r'\S+@\S+', '', text)

    # Tags HTML
    text = re.sub(r'<.*?>', '', text)

    # Normalizar repeticiones de caracteres (holaaaa → hola)
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    # Eliminar secuencias numéricas
    text = re.sub(r'\d+', ' ', text)

    # Solo letras españolas y espacios
    text = re.sub(r'[^a-záéíóúñü\s]', ' ', text)

    # Colapsar espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ==============================================================================
# 5. PROCESAMIENTO NLP CON SPACY (LEMATIZACIÓN + NER)
# ==============================================================================

def spacy_clean_batch(texts, nlp, batch_size=256):
    """
    Procesamiento avanzado con spaCy:
    1. Reconocimiento de Entidades Nombradas (NER): las entidades se reemplazan
       por placeholders genéricos (__PERSONA__, __LUGAR__, __ORGANIZACION__, etc.)
       para generalizar el vocabulario y que el modelo aprenda a discriminar
       por tipo de entidad, no por nombres propios específicos.
    2. Lematización: reduce cada palabra a su forma base (corrieron → correr).
    3. Filtrado de stopwords (excepto las de la whitelist: 'no', 'sí', 'ni').
    4. Filtrado de tokens residuales (< 3 caracteres, excepto whitelist).

    Utiliza nlp.pipe para procesamiento en lotes eficiente.
    El componente 'parser' se desactiva por eficiencia (no se necesita análisis
    de dependencias). El componente 'ner' se MANTIENE ACTIVO.
    """
    cleaned_docs = []

    for doc in tqdm(
        nlp.pipe(texts, batch_size=batch_size, disable=["parser"]),
        total=len(texts),
        desc="Procesando NLP (lematización + NER)"
    ):
        # 1. Identificar qué tokens pertenecen a entidades NER
        ent_token_indices = {}
        for ent in doc.ents:
            for token in ent:
                ent_token_indices[token.i] = ent

        # 2. Procesar cada token
        tokens = []
        processed_ents = set()  # evitar duplicar entidades multi-token

        for token in doc:
            if token.i in ent_token_indices:
                # Token es parte de una entidad NER
                ent = ent_token_indices[token.i]
                ent_id = (ent.start, ent.end)

                if ent_id not in processed_ents:
                    processed_ents.add(ent_id)
                    placeholder = NER_MAP.get(ent.label_, '__ENTIDAD__')
                    tokens.append(placeholder)
                # Si ya procesamos esta entidad, simplemente skip
            else:
                # Token normal: aplicar lematización y filtrado
                lemma = token.lemma_.lower()

                # Conservar stopwords de la whitelist
                if token.is_stop and lemma not in STOPWORD_WHITELIST:
                    continue

                # Filtrar puntuación y espacios
                if token.is_punct or token.is_space:
                    continue

                # Filtrar tokens no alfabéticos
                if not token.is_alpha:
                    continue

                # Filtrar tokens muy cortos (< 3 chars), excepto whitelist
                if len(lemma) < 3 and lemma not in STOPWORD_WHITELIST:
                    continue

                tokens.append(lemma)

        cleaned_docs.append(" ".join(tokens))

    return cleaned_docs


# ==============================================================================
# 6. PIPELINE COMPLETO DE PREPROCESAMIENTO
# ==============================================================================

def load_and_prepare_data(csv_path):
    """
    Carga el dataset CSV, aplica toda la cadena de preprocesamiento:
    1. Agrupación de tópicos (212 → 9 categorías).
    2. Limpieza estructural (nulos, duplicados, textos vacíos).
    3. Filtro de idioma (descarta textos no españoles).
    4. Limpieza regex (URLs, emails, HTML, Unicode, repeticiones, números).
    5. Procesamiento NLP con spaCy (lematización + NER + filtrado).

    Retorna el DataFrame con las columnas:
        - Title, Content, Topic_Grouped, Cleaned_Content
    """
    print(f"[Preprocesamiento] Cargando dataset desde: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[Preprocesamiento] Registros iniciales: {len(df)}")

    # === PASO 1: Agrupación de tópicos ===
    print("\n[Preprocesamiento] Paso 1/6: Agrupando tópicos (212 → 9 categorías)...")
    df['Topic_Grouped'] = df['Topic'].apply(map_topic)

    # === PASO 2: Limpieza estructural ===
    print("[Preprocesamiento] Paso 2/6: Limpieza estructural...")
    # Eliminar nulos
    n_before = len(df)
    df = df.dropna(subset=['Title', 'Content', 'Topic_Grouped']).copy()
    df = df[(df['Title'].str.strip() != '') & (df['Content'].str.strip() != '')]
    print(f"  - Nulos/vacíos eliminados: {n_before - len(df)}")

    # Eliminar duplicados
    n_before = len(df)
    df = df.drop_duplicates(subset=['Title', 'Content'])
    print(f"  - Duplicados eliminados: {n_before - len(df)}")

    # Filtrar textos extremadamente cortos (probables errores de parseo CSV)
    n_before = len(df)
    df = df[df['Content'].str.len() >= 100]
    print(f"  - Textos cortos (< 100 chars) eliminados: {n_before - len(df)}")

    # === PASO 3: Filtro de idioma ===
    print("\n[Preprocesamiento] Paso 3/6: Filtrado de idioma...")
    df, n_lang_removed = filter_non_spanish(df)
    print(f"  - Total tras filtro de idioma: {len(df)}")

    # === PASO 4: Unión de Título + Contenido ===
    print("\n[Preprocesamiento] Paso 4/6: Uniendo Título + Contenido...")
    df['Texto_Completo'] = df['Title'] + " " + df['Content']

    # === PASO 5: Limpieza con Regex ===
    print("[Preprocesamiento] Paso 5/6: Aplicando limpieza Regex exhaustiva...")
    tqdm.pandas(desc="Limpieza Regex")
    df['Regex_Content'] = df['Texto_Completo'].progress_apply(regex_clean)

    # === PASO 6: Procesamiento con spaCy (Lematización + NER) ===
    print("\n[Preprocesamiento] Paso 6/6: Procesamiento NLP con spaCy...")
    print(f"  - Modelo: {SPACY_MODEL_NAME}")
    print("  - NER ACTIVO: las entidades se reemplazarán por tipo genérico")
    nlp = _load_spacy_model()
    texts_to_process = df['Regex_Content'].tolist()
    df['Cleaned_Content'] = spacy_clean_batch(texts_to_process, nlp)

    # Limpiar columnas intermedias
    df = df.drop(columns=['Regex_Content', 'Texto_Completo'])
    df = df.reset_index(drop=True)

    # Eliminar registros que quedaron vacíos tras el procesamiento
    n_before = len(df)
    df = df[df['Cleaned_Content'].str.strip() != '']
    print(f"\n[Preprocesamiento] Textos vacíos tras NLP eliminados: {n_before - len(df)}")

    print(f"\n[Preprocesamiento] === RESUMEN ===")
    print(f"  Registros finales: {len(df)}")
    print(f"  Distribución de clases:")
    print(df['Topic_Grouped'].value_counts().to_string(header=False))

    return df


# ==============================================================================
# 7. TF-IDF + SPLIT + GUARDADO DE ARTEFACTOS
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
