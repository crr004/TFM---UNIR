import pandas as pd
import re
import unicodedata
from collections import Counter
import spacy
from tqdm import tqdm

# -------------------------
# CARGAR MODELO (optimizado)
# -------------------------
nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])

# -------------------------
# CARGAR DATASET
# -------------------------
df = pd.read_csv("Data_Bin_Classifier.csv", encoding="utf-8")

print("Columnas del dataset:")
print(df.columns)

# -------------------------
# CREAR TEXTO COMPLETO
# -------------------------
df['full_text'] = df['Title'] + " " + df['Content']

# -------------------------
# ELIMINAR DUPLICADOS
# -------------------------
num_antes = len(df)
df = df.drop_duplicates(subset='full_text')
num_despues = len(df)

print(f"\nDuplicados eliminados: {num_antes - num_despues}")

# -------------------------
# CONTADOR DE MÉTRICAS
# -------------------------
stats = Counter()

# -------------------------
# FUNCIÓN LIMPIEZA
# -------------------------
def limpiar_texto(texto):

    if pd.isnull(texto):
        stats['vacios'] += 1
        return ""

    texto = texto.lower()

    # Normalizar repeticiones (holaaaa → hola)
    nuevo = re.sub(r'(.)\1{2,}', r'\1', texto)
    if nuevo != texto:
        stats['repeticiones_normalizadas'] += 1
    texto = nuevo

    # URLs
    urls = re.findall(r'http\S+|www\S+', texto)
    stats['urls'] += len(urls)
    texto = re.sub(r'http\S+|www\S+', '', texto)

    # Emails
    emails = re.findall(r'\S+@\S+', texto)
    stats['emails'] += len(emails)
    texto = re.sub(r'\S+@\S+', '', texto)

    # Números
    numeros = re.findall(r'\d+', texto)
    stats['numeros'] += len(numeros)
    texto = re.sub(r'\d+', ' numero ', texto)

    # Mantener tildes (importante en español)

    # Caracteres raros
    texto = re.sub(r'[^a-zA-Záéíóúñü\s]', ' ', texto)

    # Espacios
    texto = re.sub(r'\s+', ' ', texto).strip()

    return texto

# -------------------------
# LIMPIEZA INICIAL
# -------------------------
print("\nIniciando limpieza...")

textos_limpios = []

for i, texto in enumerate(df['full_text']):
    if i % 5000 == 0:
        print(f"Procesados {i} textos...")

    textos_limpios.append(limpiar_texto(texto))

print("Limpieza completada")

# -------------------------
# LEMATIZACIÓN (rápida)
# -------------------------
print("\nIniciando lematización...")

docs = nlp.pipe(textos_limpios, batch_size=100)

resultado_final = []

for doc in tqdm(docs, total=len(textos_limpios)):
    tokens = []

    for token in doc:

        # NO eliminar "no"
        if token.is_stop and token.text != "no":
            stats['stopwords'] += 1
            continue

        if len(token.text) < 3:
            stats['tokens_cortos'] += 1
            continue

        if not token.is_alpha:
            stats['no_alpha'] += 1
            continue

        tokens.append(token.lemma_)

    texto_final = " ".join(tokens)

    if texto_final == "":
        stats['vacios_post'] += 1

    resultado_final.append(texto_final)

df['text_clean'] = resultado_final

print("Lematización completada")

# -------------------------
# TARGET
# -------------------------
print("\nValores únicos en State:")
print(df['State'].unique())

df['label'] = df['State'].astype(int)

# -------------------------
# MÉTRICAS
# -------------------------
print("\nESTADÍSTICAS DEL PREPROCESADO:\n")

for k, v in stats.items():
    print(f"{k}: {v}")

# -------------------------
# GUARDAR
# -------------------------
df.to_csv("dataset_preprocesado_binario.csv", index=False, encoding="utf-8-sig")

print("\nPreprocesado completado correctamente")
