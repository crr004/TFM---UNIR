import warnings
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from sklearn.exceptions import ConvergenceWarning

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    log_loss,
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
# SPLIT 70 / 20 / 10  (train / test / val)
# =========================================================

# -------------------------
# 5. Primer split -> TRAIN (70%) y TEMP (30%)
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
# 6. Segundo split -> TEST (20%) y VALIDATION (10%)
#    test_size=2/3 del 30% restante -> test=20%, val=10%
# -------------------------
X_text_test, X_text_val, \
X_text_full_test, X_text_full_val, \
X_num_test, X_num_val, \
y_test, y_val = train_test_split(

    X_text_temp,
    X_text_full_temp,
    X_num_temp,
    y_temp,

    test_size=1/3,
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

# -------------------------
# 13. Buscar mejor threshold
# -------------------------
for t in np.arange(0.3, 0.7, 0.01):

    y_val_pred_temp = (y_val_probs >= t).astype(int)

    f1_temp = f1_score(y_val, y_val_pred_temp)

    if f1_temp > best_f1:

        best_f1 = f1_temp
        best_threshold = t

# -------------------------
# 14. Predicción VALIDACIÓN con mejor threshold
# -------------------------
y_val_pred = (y_val_probs >= best_threshold).astype(int)

# =========================================================
# EVALUACIÓN FINAL EN TEST
# =========================================================

# -------------------------
# 15. Predicción TEST
# -------------------------
y_test_probs = model.predict_proba(X_test_final)[:, 1]

y_test_pred = (y_test_probs >= best_threshold).astype(int)

# =========================================================
# MÉTRICAS
# =========================================================

# -------------------------
# 16. Métricas TEST
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
# 17. ROC-AUC
# -------------------------
roc_auc = roc_auc_score(
    y_test,
    y_test_probs
)

# -------------------------
# 18. Classification report
# -------------------------
class_report = classification_report(
    y_test,
    y_test_pred
)

# -------------------------
# 19. Confusion Matrix
# -------------------------
conf_matrix = confusion_matrix(
    y_test,
    y_test_pred
)

# =========================================================
# RESULTADOS
# =========================================================

# -------------------------
# 20. Resultados finales
# -------------------------
print("\n================================================")
print("RESULTADOS - REGRESIÓN LOGÍSTICA")
print("================================================")

print(f"\nMejor threshold (VALIDACIÓN): {best_threshold:.2f}")

print(f"\nTraining Time: {training_time:.4f} segundos")

print("\n--- MÉTRICAS TEST ---")
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
# 21. Predicción TRAIN
# -------------------------
y_train_probs = model.predict_proba(X_train_final)[:, 1]
y_train_pred = (y_train_probs >= best_threshold).astype(int)

# -------------------------
# 22. Métricas TRAIN
# -------------------------
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

# -------------------------
# 23. Métricas VALIDACIÓN
# -------------------------
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred)
val_recall = recall_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred)

# -------------------------
# 24. Comparación TRAIN vs VAL vs TEST
# -------------------------
print("\n================================================")
print("COMPARACIÓN TRAIN vs VAL vs TEST")
print("================================================")

print("\nTRAIN:")
print(f"Accuracy: {train_accuracy:.4f}")
print(f"Precision: {train_precision:.4f}")
print(f"Recall: {train_recall:.4f}")
print(f"F1-score: {train_f1:.4f}")

print("\nVALIDACIÓN:")
print(f"Accuracy: {val_accuracy:.4f}")
print(f"Precision: {val_precision:.4f}")
print(f"Recall: {val_recall:.4f}")
print(f"F1-score: {val_f1:.4f}")

print("\nTEST:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# =========================================================
# GUARDAR RESULTADOS EN JSON
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

    "val_metrics": {
        "accuracy": round(float(val_accuracy), 4),
        "precision": round(float(val_precision), 4),
        "recall": round(float(val_recall), 4),
        "f1_score": round(float(val_f1), 4)
    },

    "confusion_matrix": conf_matrix.tolist(),
    "classification_report": class_report

}

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
# GRÁFICAS DE EVALUACIÓN
# =========================================================

print("\n================================================")
print("GENERANDO GRÁFICAS DE EVALUACIÓN")
print("================================================")

os.makedirs("graficas/logistic_regression", exist_ok=True)

# -------------------------
# 25. Curvas de convergencia (Loss y Precision vs iteraciones)
# -------------------------
# Se entrena con saga y max_iter creciente para simular la convergencia
# del solver, equivalente a "epochs" en ML clásico

iter_checkpoints = [10, 25, 50, 100, 200, 500, 1000, 2000]

conv_train_losses      = []
conv_val_losses        = []
conv_train_precisions  = []
conv_val_precisions    = []

with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    for n_iter in iter_checkpoints:

        conv_model = LogisticRegression(
            max_iter=n_iter,
            C=2,
            class_weight="balanced",
            solver="saga",
            tol=1e-10
        )

        conv_model.fit(X_train_final, y_train)

        tp = conv_model.predict_proba(X_train_final)
        vp = conv_model.predict_proba(X_val_final)

        conv_train_losses.append(log_loss(y_train, tp))
        conv_val_losses.append(log_loss(y_val, vp))

        train_pred_conv = (tp[:, 1] >= best_threshold).astype(int)
        val_pred_conv   = (vp[:, 1] >= best_threshold).astype(int)

        conv_train_precisions.append(precision_score(y_train, train_pred_conv))
        conv_val_precisions.append(precision_score(y_val,   val_pred_conv))

# Loss curve
plt.figure(figsize=(8, 6))
plt.plot(iter_checkpoints, conv_train_losses, marker='o', label="Train Loss")
plt.plot(iter_checkpoints, conv_val_losses,   marker='o', label="Validación Loss")
plt.xscale("log")
plt.xlabel("Iteraciones del solver")
plt.ylabel("Log Loss")
plt.title("Curva de Aprendizaje - Loss - Regresión Logística")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "graficas/logistic_regression/learning_curve_loss.png",
    bbox_inches='tight'
)
plt.close()

print("[OK] learning_curve_loss.png")

# Precision curve
plt.figure(figsize=(8, 6))
plt.plot(iter_checkpoints, conv_train_precisions, marker='o', label="Train Precision")
plt.plot(iter_checkpoints, conv_val_precisions,   marker='o', label="Validación Precision")
plt.xscale("log")
plt.xlabel("Iteraciones del solver")
plt.ylabel("Precision")
plt.title("Curva de Aprendizaje - Precisión - Regresión Logística")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "graficas/logistic_regression/learning_curve_precision.png",
    bbox_inches='tight'
)
plt.close()

print("[OK] learning_curve_precision.png")

# =========================================================
# Curvas por tamaño del conjunto de entrenamiento
# =========================================================
# A diferencia de sklearn.learning_curve, aquí se usa el vectorizer y
# el scaler ya ajustados sobre el 100% de train, por lo que solo varía
# la cantidad de muestras que ve el modelo en cada paso.
# Eje X = número de muestras de entrenamiento (10% ... 100% de X_train_final)

n_train_total = X_train_final.shape[0]
lc_fractions  = np.linspace(0.1, 1.0, 10)
lc_sizes      = [max(50, int(f * n_train_total)) for f in lc_fractions]

lc_train_losses     = []
lc_val_losses       = []
lc_train_precisions = []
lc_val_precisions   = []
lc_train_f1s        = []
lc_val_f1s          = []

for n in lc_sizes:

    X_sub = X_train_final[:n]
    y_sub = y_train.iloc[:n]

    lc_model = LogisticRegression(
        max_iter=2000,
        C=2,
        class_weight="balanced",
        solver="liblinear"
    )
    lc_model.fit(X_sub, y_sub)

    tp = lc_model.predict_proba(X_sub)
    vp = lc_model.predict_proba(X_val_final)

    lc_train_losses.append(log_loss(y_sub, tp))
    lc_val_losses.append(log_loss(y_val,  vp))

    train_pred_lc = (tp[:, 1] >= best_threshold).astype(int)
    val_pred_lc   = (vp[:, 1] >= best_threshold).astype(int)

    lc_train_precisions.append(precision_score(y_sub, train_pred_lc))
    lc_val_precisions.append(precision_score(y_val,   val_pred_lc))

    lc_train_f1s.append(f1_score(y_sub, train_pred_lc))
    lc_val_f1s.append(f1_score(y_val,   val_pred_lc))

# Loss vs tamaño
plt.figure(figsize=(8, 6))
plt.plot(lc_sizes, lc_train_losses, marker='o', label="Train Loss")
plt.plot(lc_sizes, lc_val_losses,   marker='o', label="Validación Loss")
plt.xlabel("Tamaño del conjunto de entrenamiento")
plt.ylabel("Log Loss")
plt.title("Curva por Tamaño - Loss - Regresión Logística")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "graficas/logistic_regression/lc_size_loss.png",
    bbox_inches='tight'
)
plt.close()
print("[OK] lc_size_loss.png")

# Precision vs tamaño
plt.figure(figsize=(8, 6))
plt.plot(lc_sizes, lc_train_precisions, marker='o', label="Train Precision")
plt.plot(lc_sizes, lc_val_precisions,   marker='o', label="Validación Precision")
plt.xlabel("Tamaño del conjunto de entrenamiento")
plt.ylabel("Precision")
plt.title("Curva por Tamaño - Precisión - Regresión Logística")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "graficas/logistic_regression/lc_size_precision.png",
    bbox_inches='tight'
)
plt.close()
print("[OK] lc_size_precision.png")

# F1 vs tamaño
plt.figure(figsize=(8, 6))
plt.plot(lc_sizes, lc_train_f1s, marker='o', label="Train F1")
plt.plot(lc_sizes, lc_val_f1s,   marker='o', label="Validación F1")
plt.xlabel("Tamaño del conjunto de entrenamiento")
plt.ylabel("F1-score")
plt.title("Curva por Tamaño - F1 - Regresión Logística")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "graficas/logistic_regression/lc_size_f1.png",
    bbox_inches='tight'
)
plt.close()
print("[OK] lc_size_f1.png")

# -------------------------
# 26. Barras: métricas Train / Validación / Test
# -------------------------
metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-score']

train_values = [train_accuracy, train_precision, train_recall, train_f1]
val_values   = [val_accuracy,   val_precision,   val_recall,   val_f1]
test_values  = [accuracy,       precision,       recall,       f1]

x = np.arange(len(metrics_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 6))

ax.bar(x - width, train_values, width, label='Train')
ax.bar(x,         val_values,   width, label='Validación')
ax.bar(x + width, test_values,  width, label='Test')

ax.set_xlabel("Métrica")
ax.set_ylabel("Valor")
ax.set_title("Métricas Train / Validación / Test - Regresión Logística")
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis='y')
plt.tight_layout()
plt.savefig(
    "graficas/logistic_regression/metrics_comparison.png",
    bbox_inches='tight'
)
plt.close()

print("[OK] metrics_comparison.png")

# -------------------------
# 26. Matriz de Confusión
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
# 27. Curva ROC-AUC
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
# 28. EXPLICABILIDAD (SHAP + LIME)
# =========================================================

import shap

from lime.lime_text import LimeTextExplainer

# =========================================================
# CREAR CARPETA
# =========================================================

os.makedirs("xai/logistic_regression", exist_ok=True)

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

sample_size = 200

X_shap_sample = X_test_final[:sample_size]

explainer = shap.LinearExplainer(
    model,
    X_train_final,
)

shap_values = explainer.shap_values(X_shap_sample)

# -------------------------
# SHAP SUMMARY PLOT
# -------------------------
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

# -------------------------
# SHAP BAR PLOT
# -------------------------
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

# -------------------------
# SHAP WATERFALL
# -------------------------
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
    "xai/logistic_regression/shap_waterfall_plot.png",
    bbox_inches='tight'
)

plt.close()

print("[OK] shap_waterfall_plot.png")

with open(
    "xai/logistic_regression/shap_explained_text.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("===== TEXTO ORIGINAL =====\n\n")
    f.write(X_text_full_test.iloc[sample_index])

    f.write("\n\n===== TEXTO PROCESADO =====\n\n")
    f.write(X_text_test.iloc[sample_index])

# -------------------------
# TOP FEATURES JSON
# -------------------------
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
# SAMPLE INDEX Y VALORES NUMÉRICOS REALES
# =========================================================

lime_sample_idx = 15

# valores numéricos del sample real ya normalizados (shape 1 x n_numeric)
lime_sample_num = X_test_num[lime_sample_idx: lime_sample_idx + 1]

# =========================================================
# FUNCIÓN PREDICT PROBA PARA LIME
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

    # repetir los valores numéricos reales del sample para cada perturbación
    # LIME solo perturba el texto; las features numéricas se mantienen constantes
    n = len(processed_texts)
    numeric_repeated = np.tile(lime_sample_num, (n, 1))

    # combinar
    final = hstack([tfidf, numeric_repeated])

    final = final.tocsr()

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

xai_text_mapping = pd.DataFrame({
    "full_text": X_text_full_test.reset_index(drop=True),
    "processed_text": X_text_test.reset_index(drop=True)
})

sample_full_text = xai_text_mapping.iloc[lime_sample_idx]["full_text"]

sample_processed_text = xai_text_mapping.iloc[lime_sample_idx]["processed_text"]

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
