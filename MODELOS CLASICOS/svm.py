import pandas as pd
import numpy as np
import time
import json

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from sklearn.preprocessing import StandardScaler

from scipy.sparse import hstack

# -------------------------
# 1. Cargar datos
# -------------------------
df = pd.read_csv("../Data/dataset_preprocesado_binario.csv")

numeric_features = [
    'num_chars',
    'num_words',
    'num_exclamaciones',
    'num_interrogaciones',
    'num_urls',
    'num_emails',
    'num_mayusculas',
    'num_digitos',
    'num_ellipsis',
    'num_quotes'
]

df = df[['text_ml', 'full_text', 'label'] + numeric_features].dropna()

# -------------------------
# 2. Variables
# -------------------------
X_text = df['text_ml']
X_text_full = df['full_text']
X_num = df[numeric_features]
y = df['label']

# =========================================================
# SPLIT 70 / 10 / 20
# =========================================================

# -------------------------
# 3. TRAIN (70) y TEMP (30)
# -------------------------
X_text_train, X_text_temp, \
X_text_full_train, X_text_full_temp, \
X_num_train, X_num_temp, \
y_train, y_temp = train_test_split(

    X_text,
    X_text_full,
    X_num,
    y,

    test_size=0.30,
    random_state=42,
    stratify=y
)

# -------------------------
# 4. VALIDATION (10) y TEST (20)
# -------------------------
X_text_val, X_text_test, \
X_text_full_val, X_text_full_test, \
X_num_val, X_num_test, \
y_val, y_test = train_test_split(

    X_text_temp,
    X_text_full_temp,
    X_num_temp,
    y_temp,

    test_size=2/3,
    random_state=42,
    stratify=y_temp
)

# -------------------------
# 5. TF-IDF
# -------------------------
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.85
)

# FIT SOLO TRAIN
X_train_tfidf = vectorizer.fit_transform(X_text_train)

# TRANSFORM
X_val_tfidf = vectorizer.transform(X_text_val)
X_test_tfidf = vectorizer.transform(X_text_test)

# -------------------------
# 6. Normalización numérica
# -------------------------
scaler = StandardScaler()

# FIT SOLO TRAIN
X_train_num = scaler.fit_transform(X_num_train)

# TRANSFORM
X_val_num = scaler.transform(X_num_val)
X_test_num = scaler.transform(X_num_test)

# -------------------------
# 7. Combinar
# -------------------------
X_train_final = hstack([X_train_tfidf, X_train_num])
X_val_final = hstack([X_val_tfidf, X_val_num])
X_test_final = hstack([X_test_tfidf, X_test_num])

# -------------------------
# 8. Modelo SVM (Linear)
# -------------------------
model = LinearSVC(
    C=0.5,
    class_weight=None,
    max_iter=3000
)

# -------------------------
# 9. Entrenamiento
# -------------------------
start_time = time.time()

model.fit(X_train_final, y_train)

end_time = time.time()

training_time = end_time - start_time

# =========================================================
# VALIDACIÓN
# =========================================================

# -------------------------
# 10. Predicción VALIDACIÓN
# -------------------------
y_val_pred = model.predict(X_val_final)

# =========================================================
# TEST
# =========================================================

# -------------------------
# 11. Predicción TEST
# -------------------------
y_pred = model.predict(X_test_final)

# =========================================================
# MÉTRICAS TEST
# =========================================================

# -------------------------
# 12. Métricas principales
# -------------------------
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average='macro'
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average='weighted'
)

# -------------------------
# 13. ROC-AUC
# -------------------------
# LinearSVC no tiene predict_proba
# usamos decision_function

y_scores = model.decision_function(X_test_final)

roc_auc = roc_auc_score(
    y_test,
    y_scores
)

# -------------------------
# 14. Classification Report
# -------------------------
class_report = classification_report(
    y_test,
    y_pred
)

# -------------------------
# 15. Confusion Matrix
# -------------------------
conf_matrix = confusion_matrix(
    y_test,
    y_pred
)

# =========================================================
# RESULTADOS
# =========================================================

# -------------------------
# 16. Resultados finales
# -------------------------
print("\n================================================")
print("RESULTADOS - SVM (LinearSVC)")
print("================================================")

print(f"\nTraining Time: {training_time:.4f} segundos")

print("\n--- MÉTRICAS ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Weighted F1-score: {weighted_f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\n--- CLASSIFICATION REPORT ---")
print(class_report)

print("\n--- CONFUSION MATRIX ---")
print(conf_matrix)

# =========================================================
# OVERFITTING CHECK
# =========================================================

# -------------------------
# 17. Predicción TRAIN
# -------------------------
y_train_pred = model.predict(X_train_final)

# -------------------------
# 18. Métricas TRAIN
# -------------------------
train_accuracy = accuracy_score(y_train, y_train_pred)

train_precision = precision_score(y_train, y_train_pred)

train_recall = recall_score(y_train, y_train_pred)

train_f1 = f1_score(y_train, y_train_pred)

# -------------------------
# 19. Comparación TRAIN vs TEST
# -------------------------
print("\n================================================")
print("COMPARACIÓN TRAIN vs TEST")
print("================================================")

print("\nTRAIN:")
print(f"Accuracy: {train_accuracy:.4f}")
print(f"Precision: {train_precision:.4f}")
print(f"Recall: {train_recall:.4f}")
print(f"F1-score: {train_f1:.4f}")

print("\nTEST:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# =========================================================
# GUARDAR RESULTADOS JSON
# =========================================================

# -------------------------
# 20. Diccionario resultados
# -------------------------
results = {

    "model": "LinearSVC",
    "training_time_seconds": round(float(training_time), 4),
    "metrics": {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "macro_f1_score": round(float(macro_f1), 4),
        "weighted_f1_score": round(float(weighted_f1), 4),
        "roc_auc": round(float(roc_auc), 4)

    },

    "train_metrics": {
        "accuracy": round(float(train_accuracy), 4),
        "precision": round(float(train_precision), 4),
        "recall": round(float(train_recall), 4),
        "f1_score": round(float(train_f1), 4)

    },

    "confusion_matrix": conf_matrix.tolist(),
    "classification_report": class_report

}

# -------------------------
# 21. Guardar JSON
# -------------------------
with open("svm_results.json", "w", encoding="utf-8") as f:

    json.dump(
        results,
        f,
        indent=4,
        ensure_ascii=False
    )

print("\nResultados guardados en:")
print("svm_results.json")

# =========================================================
# 22. EXPLICABILIDAD (SHAP + LIME)
# =========================================================

# INSTALAR:
# pip install shap lime

import shap
import matplotlib.pyplot as plt
import os

from lime.lime_text import LimeTextExplainer

# =========================================================
# CREAR CARPETA
# =========================================================

os.makedirs("xai/svm", exist_ok=True)

# =========================================================
# FEATURE NAMES
# =========================================================

tfidf_feature_names = vectorizer.get_feature_names_out()

all_feature_names = list(tfidf_feature_names) + numeric_features

# =========================================================
# SHAP
# =========================================================

print("\n================================================")
print("GENERANDO EXPLICACIONES SHAP")
print("================================================")

# convertir a CSR
X_train_final = X_train_final.tocsr()
X_test_final = X_test_final.tocsr()

# sample pequeño para RAM
sample_size = 200

X_shap_sample = X_test_final[:sample_size]

# =========================================================
# SHAP EXPLAINER
# =========================================================

explainer = shap.LinearExplainer(
    model,
    X_train_final
)

# =========================================================
# SHAP VALUES
# =========================================================

shap_values = explainer.shap_values(X_shap_sample)

# =========================================================
# SHAP SUMMARY PLOT
# =========================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_shap_sample,
    feature_names=all_feature_names,
    max_display=30,
    show=False
)

plt.tight_layout()

plt.savefig(
    "xai/svm/shap_summary_plot.png",
    bbox_inches='tight'
)

plt.close()

print("[OK] shap_summary_plot.png")

# =========================================================
# SHAP BAR PLOT
# =========================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_shap_sample,
    feature_names=all_feature_names,
    plot_type="bar",
    max_display=30,
    show=False
)

plt.tight_layout()

plt.savefig(
    "xai/svm/shap_bar_plot.png",
    bbox_inches='tight'
)

plt.close()

print("[OK] shap_bar_plot.png")

# =========================================================
# SHAP WATERFALL
# =========================================================

sample_index = 0

sample_dense = X_shap_sample[sample_index].toarray()[0]

explanation = shap.Explanation(
    values=shap_values[sample_index],
    base_values=explainer.expected_value,
    data=sample_dense,
    feature_names=all_feature_names
)

plt.figure()

shap.plots.waterfall(
    explanation,
    show=False
)

plt.savefig(
    "xai/svm/shap_waterfall_plot.png",
    bbox_inches='tight'
)

plt.close()

print("[OK] shap_waterfall_plot.png")

with open(
    "xai/svm/shap_explained_text.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("===== TEXTO ORIGINAL =====\n\n")
    f.write(X_text_full_test.iloc[sample_index])

    f.write("\n\n===== TEXTO PROCESADO =====\n\n")
    f.write(X_text_test.iloc[sample_index])

# =========================================================
# TOP FEATURES JSON
# =========================================================

mean_abs_shap = np.abs(shap_values).mean(axis=0)

top_idx = np.argsort(mean_abs_shap)[::-1][:20]

top_features = []

for idx in top_idx:

    top_features.append({
        "feature": all_feature_names[idx],
        "importance": float(mean_abs_shap[idx])
    })

with open(
    "xai/svm/top_features_shap.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        top_features,
        f,
        indent=4,
        ensure_ascii=False
    )

print("[OK] top_features_shap.json")

# =========================================================
# LIME
# =========================================================

print("\n================================================")
print("GENERANDO EXPLICACIONES LIME")
print("================================================")

# =========================================================
# MAPPING FULL_TEXT -> TEXT_ML
# =========================================================

text_mapping_dict = dict(
    zip(
        X_text_full_test,
        X_text_test
    )
)

# =========================================================
# FUNCIÓN PREDICT PARA LIME
# =========================================================

def predict_proba_lime(texts):

    processed_texts = []

    for text in texts:

        if text in text_mapping_dict:

            processed_texts.append(
                text_mapping_dict[text]
            )

        else:

            processed_texts.append(text)

    # TF-IDF
    tfidf = vectorizer.transform(processed_texts)

    # features numéricas vacías
    numeric_zeros = np.zeros(
        (len(processed_texts), len(numeric_features))
    )

    # combinar
    final = hstack([tfidf, numeric_zeros])

    final = final.tocsr()

    # scores SVM
    scores = model.decision_function(final)

    # pseudo-probabilidades
    probs_fake = 1 / (1 + np.exp(-scores))

    probs_real = 1 - probs_fake

    probs = np.vstack([
        probs_real,
        probs_fake
    ]).T

    return probs

# =========================================================
# LIME EXPLAINER
# =========================================================

lime_explainer = LimeTextExplainer(
    class_names=["REAL", "FAKE"]
)

# =========================================================
# SAMPLE TEXT
# =========================================================

confidence_scores = np.abs(y_scores - 0)

sample_index = np.argmax(confidence_scores)

sample_text = X_text_full_test.iloc[sample_index]

# =========================================================
# GENERAR EXPLICACIÓN
# =========================================================

lime_exp = lime_explainer.explain_instance(
    sample_text,
    predict_proba_lime,
    num_features=15
)

# =========================================================
# GUARDAR HTML
# =========================================================

lime_exp.save_to_file(
    "xai/svm/lime_explanation.html"
)

print("[OK] lime_explanation.html")

# =========================================================
# GUARDAR FEATURES LIME JSON
# =========================================================

lime_features = []

for feature, weight in lime_exp.as_list():

    lime_features.append({
        "feature": feature,
        "weight": float(weight)
    })

with open(
    "xai/svm/lime_features.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        lime_features,
        f,
        indent=4,
        ensure_ascii=False
    )

print("[OK] lime_features.json")

# =========================================================
# RESUMEN FINAL
# =========================================================

print("\n================================================")
print("XAI COMPLETADO")
print("================================================")

print("\nArchivos generados:")

print("\nSHAP:")
print("- shap_summary_plot.png")
print("- shap_bar_plot.png")
print("- shap_waterfall_plot.png")
print("- top_features_shap.json")

print("\nLIME:")
print("- lime_explanation.html")
print("- lime_features.json")