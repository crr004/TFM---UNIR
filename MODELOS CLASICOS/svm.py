import pandas as pd
import numpy as np
import time
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    hinge_loss,
    classification_report,
    confusion_matrix
)

from sklearn.preprocessing import StandardScaler

from sklearn.calibration import CalibratedClassifierCV

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
# SPLIT 70 / 20 / 10  (train / test / val)
# =========================================================

# -------------------------
# 3. TRAIN (70%) y TEMP (30%)
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
# 4. TEST (20%) y VALIDATION (10%)
#    test_size=1/3 del 30% restante -> test=20%, val=10%
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
X_train_final = hstack([X_train_tfidf, X_train_num]).tocsr()
X_val_final   = hstack([X_val_tfidf,   X_val_num  ]).tocsr()
X_test_final  = hstack([X_test_tfidf,  X_test_num ]).tocsr()

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
# PREDICCIONES
# =========================================================

# -------------------------
# 10. Predicción TEST
# -------------------------
y_pred = model.predict(X_test_final)

# -------------------------
# 11. Scores TEST (decision_function)
# LinearSVC no tiene predict_proba
# -------------------------
y_scores = model.decision_function(X_test_final)

# =========================================================
# MÉTRICAS TEST
# =========================================================

# -------------------------
# 12. Métricas principales TEST
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
# 17. Predicción y métricas TRAIN
# -------------------------
y_train_pred = model.predict(X_train_final)

train_accuracy  = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall    = recall_score(y_train, y_train_pred)
train_f1        = f1_score(y_train, y_train_pred)

# -------------------------
# 18. Predicción y métricas VALIDACIÓN
# -------------------------
y_val_pred = model.predict(X_val_final)

val_accuracy  = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred)
val_recall    = recall_score(y_val, y_val_pred)
val_f1        = f1_score(y_val, y_val_pred)

# -------------------------
# 19. Comparación TRAIN vs VAL vs TEST
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
# GUARDAR RESULTADOS JSON
# =========================================================

# -------------------------
# 20. Guardar JSON
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

    "val_metrics": {
        "accuracy": round(float(val_accuracy), 4),
        "precision": round(float(val_precision), 4),
        "recall": round(float(val_recall), 4),
        "f1_score": round(float(val_f1), 4)
    },

    "confusion_matrix": conf_matrix.tolist(),
    "classification_report": class_report

}

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
# GRÁFICAS DE EVALUACIÓN
# =========================================================

print("\n================================================")
print("GENERANDO GRÁFICAS DE EVALUACIÓN")
print("================================================")

os.makedirs("graficas/svm", exist_ok=True)

# =========================================================
# 21. Curvas de aprendizaje por época (SGDClassifier hinge)
# =========================================================
# LinearSVC con coordenadas descenso converge casi inmediatamente
# en datos de texto linealmente separables -> curvas planas.
# SGDClassifier con loss='hinge' es matemáticamente equivalente
# al SVM lineal pero entrena por épocas estocásticas, mostrando
# una convergencia gradual real. warm_start=True permite acumular
# épocas llamando fit() varias veces.

n_epochs = 50
epochs_axis = list(range(1, n_epochs + 1))

conv_train_losses     = []
conv_val_losses       = []
conv_train_precisions = []
conv_val_precisions   = []

sgd_conv = SGDClassifier(
    loss='hinge',
    alpha=1e-4,
    max_iter=1,
    warm_start=True,
    shuffle=True,
    random_state=42,
    tol=None
)

for epoch in range(n_epochs):

    sgd_conv.fit(X_train_final, y_train)

    train_scores_conv = sgd_conv.decision_function(X_train_final)
    val_scores_conv   = sgd_conv.decision_function(X_val_final)

    conv_train_losses.append(hinge_loss(y_train, train_scores_conv))
    conv_val_losses.append(hinge_loss(y_val,   val_scores_conv))

    conv_train_precisions.append(
        precision_score(y_train, sgd_conv.predict(X_train_final))
    )
    conv_val_precisions.append(
        precision_score(y_val, sgd_conv.predict(X_val_final))
    )

# Hinge Loss curve
plt.figure(figsize=(8, 6))
plt.plot(epochs_axis, conv_train_losses, label="Train Hinge Loss")
plt.plot(epochs_axis, conv_val_losses,   label="Validación Hinge Loss")
plt.xlabel("Época")
plt.ylabel("Hinge Loss")
plt.title("Curva de Aprendizaje - Loss - SVM (SGD Hinge)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "graficas/svm/learning_curve_loss.png",
    bbox_inches='tight'
)
plt.close()

print("[OK] learning_curve_loss.png")

# Precision curve
plt.figure(figsize=(8, 6))
plt.plot(epochs_axis, conv_train_precisions, label="Train Precision")
plt.plot(epochs_axis, conv_val_precisions,   label="Validación Precision")
plt.xlabel("Época")
plt.ylabel("Precision")
plt.title("Curva de Aprendizaje - Precisión - SVM (SGD Hinge)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "graficas/svm/learning_curve_precision.png",
    bbox_inches='tight'
)
plt.close()

print("[OK] learning_curve_precision.png")

# =========================================================
# Curvas por tamaño del conjunto de entrenamiento
# =========================================================
# El vectorizer y el scaler ya están ajustados sobre el 100% de train.
# SVM usa decision_function -> hinge_loss para la curva de loss.

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

    lc_svm = LinearSVC(
        C=0.5,
        class_weight=None,
        max_iter=3000
    )
    lc_svm.fit(X_sub, y_sub)

    train_scores_lc = lc_svm.decision_function(X_sub)
    val_scores_lc   = lc_svm.decision_function(X_val_final)

    lc_train_losses.append(hinge_loss(y_sub, train_scores_lc))
    lc_val_losses.append(hinge_loss(y_val,   val_scores_lc))

    train_pred_lc = lc_svm.predict(X_sub)
    val_pred_lc   = lc_svm.predict(X_val_final)

    lc_train_precisions.append(precision_score(y_sub, train_pred_lc))
    lc_val_precisions.append(precision_score(y_val,   val_pred_lc))

    lc_train_f1s.append(f1_score(y_sub, train_pred_lc))
    lc_val_f1s.append(f1_score(y_val,   val_pred_lc))

# Loss vs tamaño
plt.figure(figsize=(8, 6))
plt.plot(lc_sizes, lc_train_losses, marker='o', label="Train Hinge Loss")
plt.plot(lc_sizes, lc_val_losses,   marker='o', label="Validación Hinge Loss")
plt.xlabel("Tamaño del conjunto de entrenamiento")
plt.ylabel("Hinge Loss")
plt.title("Curva por Tamaño - Loss - SVM (LinearSVC)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "graficas/svm/lc_size_loss.png",
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
plt.title("Curva por Tamaño - Precisión - SVM (LinearSVC)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "graficas/svm/lc_size_precision.png",
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
plt.title("Curva por Tamaño - F1 - SVM (LinearSVC)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "graficas/svm/lc_size_f1.png",
    bbox_inches='tight'
)
plt.close()
print("[OK] lc_size_f1.png")

# -------------------------
# 22. Barras: métricas Train / Validación / Test
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
ax.set_title("Métricas Train / Validación / Test - SVM (LinearSVC)")
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis='y')
plt.tight_layout()
plt.savefig(
    "graficas/svm/metrics_comparison.png",
    bbox_inches='tight'
)
plt.close()

print("[OK] metrics_comparison.png")

# -------------------------
# 23. Matriz de Confusión
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
plt.title("Matriz de Confusión - SVM (LinearSVC)")
plt.tight_layout()
plt.savefig(
    "graficas/svm/confusion_matrix.png",
    bbox_inches='tight'
)
plt.close()

print("[OK] confusion_matrix.png")

# -------------------------
# 24. Curva ROC-AUC
# LinearSVC usa decision_function como score
# -------------------------
fpr, tpr, _ = roc_curve(y_test, y_scores)

plt.figure(figsize=(8, 6))

plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], 'k--', label="Clasificador aleatorio")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC-AUC - SVM (LinearSVC)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "graficas/svm/roc_auc_curve.png",
    bbox_inches='tight'
)
plt.close()

print("[OK] roc_auc_curve.png")

print("\nGráficas guardadas en: graficas/svm/")

# =========================================================
# 25. EXPLICABILIDAD (SHAP + LIME)
# =========================================================

import shap

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

sample_size = 200

X_shap_sample = X_test_final[:sample_size]

explainer = shap.LinearExplainer(
    model,
    X_train_final
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
    "xai/svm/shap_summary_plot.png",
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
    "xai/svm/shap_bar_plot.png",
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
# SAMPLE INDEX Y VALORES NUMÉRICOS REALES
# =========================================================

# sample más confiado según decision_function
lime_sample_idx = int(np.argmax(np.abs(y_scores)))

# valores numéricos del sample real ya normalizados (shape 1 x n_numeric)
# LIME solo perturba el texto; las features numéricas se mantienen constantes
lime_sample_num = X_test_num[lime_sample_idx: lime_sample_idx + 1]

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

    # repetir los valores numéricos reales del sample para cada perturbación
    n = len(processed_texts)
    numeric_repeated = np.tile(lime_sample_num, (n, 1))

    # combinar
    final = hstack([tfidf, numeric_repeated]).tocsr()

    # scores SVM -> pseudo-probabilidades con sigmoid
    scores = model.decision_function(final)

    probs_fake = 1 / (1 + np.exp(-scores))
    probs_real = 1 - probs_fake

    probs = np.vstack([probs_real, probs_fake]).T

    return probs

# =========================================================
# LIME EXPLAINER
# =========================================================

lime_explainer = LimeTextExplainer(
    class_names=["REAL", "FAKE"]
)

# =========================================================
# TEXTO ORIGINAL
# =========================================================

sample_text = X_text_full_test.iloc[lime_sample_idx]

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

lime_data = {
    "original_text": X_text_full_test.iloc[lime_sample_idx],
    "processed_text": X_text_test.iloc[lime_sample_idx],
    "features": []
}

for feature, weight in lime_exp.as_list():

    lime_data["features"].append({
        "feature": feature,
        "weight": float(weight)
    })

with open(
    "xai/svm/lime_features.json",
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
