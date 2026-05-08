import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix
)

from scipy.sparse import hstack

# -------------------------
# 1. Cargar datos
# -------------------------
df = pd.read_csv("../Data/dataset_preprocesado_binario.csv")

# -------------------------
# 2. Features numéricas
# -------------------------
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

# -------------------------
# 3. Selección de columnas
# -------------------------
df = df[['text_ml', 'full_text', 'label'] + numeric_features].dropna()

# -------------------------
# 4. Variables
# -------------------------
X_text = df['text_ml']
X_text_full = df['full_text']
X_num = df[numeric_features]
y = df['label']

# =========================================================
# SPLIT 70 / 10 / 20
# =========================================================

# -------------------------
# 5. Primer split -> TRAIN (70) y TEMP (30)
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
# 6. Segundo split -> VALIDATION (10) y TEST (20)
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

# =========================================================
# TF-IDF
# =========================================================

# -------------------------
# 7. TF-IDF
# -------------------------
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9
)

# FIT SOLO EN TRAIN
X_train_tfidf = vectorizer.fit_transform(X_text_train)

# TRANSFORM EN VAL Y TEST
X_val_tfidf = vectorizer.transform(X_text_val)
X_test_tfidf = vectorizer.transform(X_text_test)

# =========================================================
# NORMALIZACIÓN
# =========================================================

# -------------------------
# 8. Normalización numérica
# -------------------------
scaler = StandardScaler()

# FIT SOLO TRAIN
X_train_num = scaler.fit_transform(X_num_train)

# TRANSFORM
X_val_num = scaler.transform(X_num_val)
X_test_num = scaler.transform(X_num_test)

# =========================================================
# COMBINACIÓN
# =========================================================

# -------------------------
# 9. Combinar features
# -------------------------
X_train_final = hstack([X_train_tfidf, X_train_num])
X_val_final = hstack([X_val_tfidf, X_val_num])
X_test_final = hstack([X_test_tfidf, X_test_num])

# convertir a CSR para permitir slicing
X_train_final = X_train_final.tocsr()
X_val_final = X_val_final.tocsr()
X_test_final = X_test_final.tocsr()

# =========================================================
# MODELO
# =========================================================

# -------------------------
# 10. Modelo
# -------------------------
model = LogisticRegression(
    max_iter=2000,
    C=2,
    class_weight="balanced",
    solver="liblinear"
)

# =========================================================
# ENTRENAMIENTO
# =========================================================

# -------------------------
# 11. Entrenamiento
# -------------------------
start_time = time.time()

model.fit(X_train_final, y_train)

end_time = time.time()

training_time = end_time - start_time

# =========================================================
# OPTIMIZACIÓN DE THRESHOLD (VALIDACIÓN)
# =========================================================

# -------------------------
# 12. Probabilidades VALIDACIÓN
# -------------------------
y_val_probs = model.predict_proba(X_val_final)[:, 1]

best_f1 = 0
best_threshold = 0
best_metrics = {}

# -------------------------
# 13. Buscar mejor threshold
# -------------------------
for t in np.arange(0.3, 0.7, 0.01):

    y_val_pred_temp = (y_val_probs >= t).astype(int)

    precision_temp = precision_score(y_val, y_val_pred_temp)
    recall_temp = recall_score(y_val, y_val_pred_temp)
    f1_temp = f1_score(y_val, y_val_pred_temp)

    if f1_temp > best_f1:

        best_f1 = f1_temp
        best_threshold = t

# =========================================================
# EVALUACIÓN FINAL EN TEST
# =========================================================

# -------------------------
# 14. Predicción TEST
# -------------------------
y_test_probs = model.predict_proba(X_test_final)[:, 1]

y_test_pred = (y_test_probs >= best_threshold).astype(int)

# =========================================================
# MÉTRICAS
# =========================================================

# -------------------------
# 15. Métricas principales
# -------------------------
accuracy = accuracy_score(y_test, y_test_pred)

precision = precision_score(y_test, y_test_pred)

recall = recall_score(y_test, y_test_pred)

f1 = f1_score(y_test, y_test_pred)

macro_f1 = f1_score(
    y_test,
    y_test_pred,
    average='macro'
)

weighted_f1 = f1_score(
    y_test,
    y_test_pred,
    average='weighted'
)

# -------------------------
# 16. ROC-AUC
# -------------------------
roc_auc = roc_auc_score(
    y_test,
    y_test_probs
)

# -------------------------
# 17. Classification report
# -------------------------
class_report = classification_report(
    y_test,
    y_test_pred
)

# -------------------------
# 18. Confusion Matrix
# -------------------------
conf_matrix = confusion_matrix(
    y_test,
    y_test_pred
)

# =========================================================
# RESULTADOS
# =========================================================

# -------------------------
# 19. Resultados finales
# -------------------------
print("\n================================================")
print("RESULTADOS - REGRESIÓN LOGÍSTICA")
print("================================================")

print(f"\nMejor threshold (VALIDACIÓN): {best_threshold:.2f}")

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
# 20. Predicción TRAIN
# -------------------------
y_train_probs = model.predict_proba(X_train_final)[:, 1]
y_train_pred = (y_train_probs >= best_threshold).astype(int)

# -------------------------
# 21. Métricas TRAIN
# -------------------------
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

# -------------------------
# 22. Comparación TRAIN vs TEST
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
# LEARNING CURVE
# =========================================================

# -------------------------
# 23. Pipeline
# -------------------------
pipeline = Pipeline([

    ("tfidf", TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9
    )),

    ("clf", LogisticRegression(
        max_iter=2000,
        C=2,
        class_weight="balanced",
        solver="liblinear"
    ))

])

# -------------------------
# 24. Learning curve
# -------------------------
train_sizes, train_scores, test_scores = learning_curve(

    pipeline,

    X_text_train,
    y_train,

    cv=5,

    scoring='f1',
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5)
)

# -------------------------
# 25. Media scores
# -------------------------
train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

# =========================================================
# GRÁFICA
# =========================================================

os.makedirs("graficas/logistic_regression", exist_ok=True)

# -------------------------
# 26. Plot Learning Curve F1
# -------------------------
plt.figure(figsize=(8, 6))

plt.plot(
    train_sizes,
    train_mean,
    label="Train F1"
)

plt.plot(
    train_sizes,
    test_mean,
    label="Validation F1"
)

plt.xlabel("Tamaño del conjunto de entrenamiento")
plt.ylabel("F1-score")
plt.title("Learning Curve - F1 - Regresión Logística")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "graficas/logistic_regression/learning_curve_f1.png",
    bbox_inches='tight'
)
plt.close()
print("[OK] learning_curve_f1.png")

# =========================================================
# 27. GUARDAR RESULTADOS EN JSON
# =========================================================

results = {

    "model": "Logistic Regression",
    "best_threshold": round(float(best_threshold), 4),
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
# Guardar JSON
# -------------------------
with open("logistic_regression_results.json", "w", encoding="utf-8") as f:

    json.dump(
        results,
        f,
        indent=4,
        ensure_ascii=False
    )

print("\nResultados guardados en:")
print("logistic_regression_results.json")

# =========================================================
# 28. GRÁFICAS DE EVALUACIÓN
# =========================================================

print("\n================================================")
print("GENERANDO GRÁFICAS DE EVALUACIÓN")
print("================================================")

# -------------------------
# 28a. Learning Curve - Precisión
# -------------------------
train_sizes_p, train_scores_p, test_scores_p = learning_curve(

    pipeline,

    X_text_train,
    y_train,

    cv=5,

    scoring='precision',
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5)
)

train_mean_p = train_scores_p.mean(axis=1)
test_mean_p = test_scores_p.mean(axis=1)

plt.figure(figsize=(8, 6))

plt.plot(train_sizes_p, train_mean_p, label="Train Precision")
plt.plot(train_sizes_p, test_mean_p, label="Validation Precision")

plt.xlabel("Tamaño del conjunto de entrenamiento")
plt.ylabel("Precision")
plt.title("Learning Curve - Precisión - Regresión Logística")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "graficas/logistic_regression/learning_curve_precision.png",
    bbox_inches='tight'
)
plt.close()

print("[OK] learning_curve_precision.png")

# -------------------------
# 28b. Learning Curve - Loss
# -------------------------
train_sizes_l, train_scores_l, test_scores_l = learning_curve(

    pipeline,

    X_text_train,
    y_train,

    cv=5,

    scoring='neg_log_loss',
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5)
)

train_mean_l = -train_scores_l.mean(axis=1)
test_mean_l = -test_scores_l.mean(axis=1)

plt.figure(figsize=(8, 6))

plt.plot(train_sizes_l, train_mean_l, label="Train Loss")
plt.plot(train_sizes_l, test_mean_l, label="Validation Loss")

plt.xlabel("Tamaño del conjunto de entrenamiento")
plt.ylabel("Log Loss")
plt.title("Learning Curve - Loss - Regresión Logística")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "graficas/logistic_regression/learning_curve_loss.png",
    bbox_inches='tight'
)
plt.close()

print("[OK] learning_curve_loss.png")

# -------------------------
# 28c. Matriz de Confusión
# -------------------------
plt.figure(figsize=(7, 5))

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['REAL', 'FAKE'],
    yticklabels=['REAL', 'FAKE']
)

plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión - Regresión Logística")
plt.tight_layout()
plt.savefig(
    "graficas/logistic_regression/confusion_matrix.png",
    bbox_inches='tight'
)
plt.close()

print("[OK] confusion_matrix.png")

# -------------------------
# 28d. Curva ROC-AUC
# -------------------------
fpr, tpr, _ = roc_curve(y_test, y_test_probs)

plt.figure(figsize=(8, 6))

plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], 'k--', label="Clasificador aleatorio")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC-AUC - Regresión Logística")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "graficas/logistic_regression/roc_auc_curve.png",
    bbox_inches='tight'
)
plt.close()

print("[OK] roc_auc_curve.png")

print("\nGráficas guardadas en: graficas/logistic_regression/")

# =========================================================
# 29. EXPLICABILIDAD (SHAP + LIME)
# =========================================================

# INSTALAR:
# pip install shap lime

import shap

from lime.lime_text import LimeTextExplainer

# =========================================================
# CREAR CARPETA
# =========================================================

os.makedirs("xai/logistic_regression", exist_ok=True)

# =========================================================
# FEATURE NAMES
# =========================================================

# nombres TF-IDF
tfidf_feature_names = vectorizer.get_feature_names_out()

# nombres numéricos
all_feature_names = list(tfidf_feature_names) + numeric_features

# =========================================================
# SHAP
# =========================================================

print("\n================================================")
print("GENERANDO EXPLICACIONES SHAP")
print("================================================")

# =========================================================
# SAMPLE PARA SHAP
# =========================================================

# para no consumir demasiada RAM
sample_size = 200

X_shap_sample = X_test_final[:sample_size]

# =========================================================
# SHAP EXPLAINER
# =========================================================

explainer = shap.LinearExplainer(
    model,
    X_train_final,
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
    "xai/logistic_regression/shap_summary_plot.png",
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
    "xai/logistic_regression/shap_bar_plot.png",
    bbox_inches='tight'
)

plt.close()

print("[OK] shap_bar_plot.png")

# =========================================================
# SHAP WATERFALL
# =========================================================

sample_index = 0

# convertir sparse -> dense SOLO 1 sample
sample_dense = X_shap_sample[sample_index].toarray()[0]

# explicación
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
    "xai/logistic_regression/shap_waterfall_plot.png",
    bbox_inches='tight'
)

plt.close()

print("[OK] shap_waterfall_plot.png")

# guardar texto explicado
with open(
    "xai/logistic_regression/shap_explained_text.txt",
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
    "xai/logistic_regression/top_features_shap.json",
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
# FUNCIÓN PREDICT PROBA PARA LIME
# =========================================================

def predict_proba_lime(texts):

    processed_texts = []

    for text in texts:

        # usar texto procesado correspondiente
        if text in text_mapping_dict:

            processed_texts.append(
                text_mapping_dict[text]
            )

        else:
            # fallback
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

    # probabilidades
    probs = model.predict_proba(final)

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
# mapping entre texto original y procesado
xai_text_mapping = pd.DataFrame({
    "full_text": X_text_full_test.reset_index(drop=True),
    "processed_text": X_text_test.reset_index(drop=True)
})

sample_full_text = xai_text_mapping.iloc[15]["full_text"]

sample_processed_text = xai_text_mapping.iloc[15]["processed_text"]

# =========================================================
# GENERAR EXPLICACIÓN
# =========================================================

lime_exp = lime_explainer.explain_instance(
    sample_full_text,
    predict_proba_lime,
    num_features=15
)

# =========================================================
# GUARDAR HTML
# =========================================================

lime_exp.save_to_file(
    "xai/logistic_regression/lime_explanation.html"
)

print("[OK] lime_explanation.html")

# =========================================================
# GUARDAR FEATURES LIME JSON
# =========================================================

lime_data = {
    "original_text": sample_full_text,
    "processed_text": sample_processed_text,
    "features": []
}

for feature, weight in lime_exp.as_list():

    lime_data["features"].append({
        "feature": feature,
        "weight": float(weight)
    })

with open(
    "xai/logistic_regression/lime_features.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        lime_data,
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