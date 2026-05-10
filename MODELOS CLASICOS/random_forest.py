import warnings
import pandas as pd
import numpy as np
import time
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.exceptions import ConvergenceWarning

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier

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
X_num = df[numeric_features]
X_text_full = df['full_text']
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

# =========================================================
# TF-IDF
# =========================================================

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

# =========================================================
# MODELO 1: SOLO TF-IDF
# =========================================================

# -------------------------
# 6. Modelo TF-IDF
# -------------------------
model_tfidf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    n_jobs=-1,
    random_state=42
)

# -------------------------
# 7. Entrenamiento
# -------------------------
start = time.time()

model_tfidf.fit(X_train_tfidf, y_train)

time_tfidf = time.time() - start

# -------------------------
# 8. Predicción TEST
# -------------------------
y_pred_tfidf = model_tfidf.predict(X_test_tfidf)

# -------------------------
# 9. Probabilidades TEST
# -------------------------
y_probs_tfidf = model_tfidf.predict_proba(X_test_tfidf)[:, 1]

# =========================================================
# MÉTRICAS TF-IDF
# =========================================================

# -------------------------
# 10. Métricas TEST
# -------------------------
acc_tfidf = accuracy_score(y_test, y_pred_tfidf)

prec_tfidf = precision_score(y_test, y_pred_tfidf)

rec_tfidf = recall_score(y_test, y_pred_tfidf)

f1_tfidf = f1_score(y_test, y_pred_tfidf)

macro_f1_tfidf = f1_score(
    y_test,
    y_pred_tfidf,
    average='macro'
)

weighted_f1_tfidf = f1_score(
    y_test,
    y_pred_tfidf,
    average='weighted'
)

# -------------------------
# 11. ROC-AUC
# -------------------------
roc_auc_tfidf = roc_auc_score(
    y_test,
    y_probs_tfidf
)

# -------------------------
# 12. Classification Report
# -------------------------
class_report_tfidf = classification_report(
    y_test,
    y_pred_tfidf
)

# -------------------------
# 13. Confusion Matrix
# -------------------------
conf_matrix_tfidf = confusion_matrix(
    y_test,
    y_pred_tfidf
)

# -------------------------
# 14. Métricas TRAIN
# -------------------------
y_train_pred_tfidf = model_tfidf.predict(X_train_tfidf)

train_acc_tfidf  = accuracy_score(y_train, y_train_pred_tfidf)
train_prec_tfidf = precision_score(y_train, y_train_pred_tfidf)
train_rec_tfidf  = recall_score(y_train, y_train_pred_tfidf)
train_f1_tfidf   = f1_score(y_train, y_train_pred_tfidf)

# -------------------------
# 15. Métricas VALIDACIÓN
# -------------------------
y_val_pred_tfidf = model_tfidf.predict(X_val_tfidf)

val_acc_tfidf  = accuracy_score(y_val, y_val_pred_tfidf)
val_prec_tfidf = precision_score(y_val, y_val_pred_tfidf)
val_rec_tfidf  = recall_score(y_val, y_val_pred_tfidf)
val_f1_tfidf   = f1_score(y_val, y_val_pred_tfidf)

# =========================================================
# MODELO 2: TF-IDF + FEATURES
# =========================================================

# -------------------------
# 16. Normalización
# -------------------------
scaler = StandardScaler()

# FIT SOLO TRAIN
X_train_num = scaler.fit_transform(X_num_train)

# TRANSFORM
X_val_num = scaler.transform(X_num_val)
X_test_num = scaler.transform(X_num_test)

# -------------------------
# 17. Combinar
# -------------------------
X_train_final = hstack([X_train_tfidf, X_train_num]).tocsr()

X_val_final = hstack([X_val_tfidf, X_val_num]).tocsr()

X_test_final = hstack([X_test_tfidf, X_test_num]).tocsr()

# -------------------------
# 18. Modelo Full
# -------------------------
model_full = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)

# -------------------------
# 19. Entrenamiento
# -------------------------
start = time.time()

model_full.fit(X_train_final, y_train)

time_full = time.time() - start

# -------------------------
# 20. Predicción TEST
# -------------------------
y_pred_full = model_full.predict(X_test_final)

# -------------------------
# 21. Probabilidades TEST
# -------------------------
y_probs_full = model_full.predict_proba(X_test_final)[:, 1]

# =========================================================
# MÉTRICAS FULL
# =========================================================

# -------------------------
# 22. Métricas TEST
# -------------------------
acc_full = accuracy_score(y_test, y_pred_full)

prec_full = precision_score(y_test, y_pred_full)

rec_full = recall_score(y_test, y_pred_full)

f1_full = f1_score(y_test, y_pred_full)

macro_f1_full = f1_score(
    y_test,
    y_pred_full,
    average='macro'
)

weighted_f1_full = f1_score(
    y_test,
    y_pred_full,
    average='weighted'
)

# -------------------------
# 23. ROC-AUC
# -------------------------
roc_auc_full = roc_auc_score(
    y_test,
    y_probs_full
)

# -------------------------
# 24. Classification Report
# -------------------------
class_report_full = classification_report(
    y_test,
    y_pred_full
)

# -------------------------
# 25. Confusion Matrix
# -------------------------
conf_matrix_full = confusion_matrix(
    y_test,
    y_pred_full
)

# -------------------------
# 26. Métricas TRAIN
# -------------------------
y_train_pred_full = model_full.predict(X_train_final)

train_acc_full  = accuracy_score(y_train, y_train_pred_full)
train_prec_full = precision_score(y_train, y_train_pred_full)
train_rec_full  = recall_score(y_train, y_train_pred_full)
train_f1_full   = f1_score(y_train, y_train_pred_full)

# -------------------------
# 27. Métricas VALIDACIÓN
# -------------------------
y_val_pred_full = model_full.predict(X_val_final)

val_acc_full  = accuracy_score(y_val, y_val_pred_full)
val_prec_full = precision_score(y_val, y_val_pred_full)
val_rec_full  = recall_score(y_val, y_val_pred_full)
val_f1_full   = f1_score(y_val, y_val_pred_full)

# =========================================================
# RESULTADOS
# =========================================================

# -------------------------
# 28. Resultados
# -------------------------
print("\n===== RANDOM FOREST: COMPARACIÓN =====")

print("\n--- SOLO TF-IDF ---")
print(f"Training Time: {time_tfidf:.4f}")
print(f"Accuracy: {acc_tfidf:.4f}")
print(f"Precision: {prec_tfidf:.4f}")
print(f"Recall: {rec_tfidf:.4f}")
print(f"F1: {f1_tfidf:.4f}")
print(f"Macro F1: {macro_f1_tfidf:.4f}")
print(f"Weighted F1: {weighted_f1_tfidf:.4f}")
print(f"ROC-AUC: {roc_auc_tfidf:.4f}")
print(f"\nTrain  -> Acc: {train_acc_tfidf:.4f}  Prec: {train_prec_tfidf:.4f}  Rec: {train_rec_tfidf:.4f}  F1: {train_f1_tfidf:.4f}")
print(f"Val    -> Acc: {val_acc_tfidf:.4f}  Prec: {val_prec_tfidf:.4f}  Rec: {val_rec_tfidf:.4f}  F1: {val_f1_tfidf:.4f}")

print("\nClassification Report:")
print(class_report_tfidf)

print("\nConfusion Matrix:")
print(conf_matrix_tfidf)

print("\n--- TF-IDF + FEATURES ---")
print(f"Training Time: {time_full:.4f}")
print(f"Accuracy: {acc_full:.4f}")
print(f"Precision: {prec_full:.4f}")
print(f"Recall: {rec_full:.4f}")
print(f"F1: {f1_full:.4f}")
print(f"Macro F1: {macro_f1_full:.4f}")
print(f"Weighted F1: {weighted_f1_full:.4f}")
print(f"ROC-AUC: {roc_auc_full:.4f}")
print(f"\nTrain  -> Acc: {train_acc_full:.4f}  Prec: {train_prec_full:.4f}  Rec: {train_rec_full:.4f}  F1: {train_f1_full:.4f}")
print(f"Val    -> Acc: {val_acc_full:.4f}  Prec: {val_prec_full:.4f}  Rec: {val_rec_full:.4f}  F1: {val_f1_full:.4f}")

print("\nClassification Report:")
print(class_report_full)

print("\nConfusion Matrix:")
print(conf_matrix_full)

# -------------------------
# 29. Diferencias
# -------------------------
print("\n--- DIFERENCIA (Full - TF-IDF) ---")
print(f"Δ F1: {f1_full - f1_tfidf:.4f}")
print(f"Δ Precision: {prec_full - prec_tfidf:.4f}")
print(f"Δ Recall: {rec_full - rec_tfidf:.4f}")

# =========================================================
# JSON
# =========================================================

# -------------------------
# 30. Guardar JSON
# -------------------------
results = {

    "random_forest_tfidf": {
        "training_time_seconds": round(float(time_tfidf), 4),
        "metrics": {
            "accuracy": round(float(acc_tfidf), 4),
            "precision": round(float(prec_tfidf), 4),
            "recall": round(float(rec_tfidf), 4),
            "f1_score": round(float(f1_tfidf), 4),
            "macro_f1_score": round(float(macro_f1_tfidf), 4),
            "weighted_f1_score": round(float(weighted_f1_tfidf), 4),
            "roc_auc": round(float(roc_auc_tfidf), 4)
        },
        "train_metrics": {
            "accuracy": round(float(train_acc_tfidf), 4),
            "precision": round(float(train_prec_tfidf), 4),
            "recall": round(float(train_rec_tfidf), 4),
            "f1_score": round(float(train_f1_tfidf), 4)
        },
        "val_metrics": {
            "accuracy": round(float(val_acc_tfidf), 4),
            "precision": round(float(val_prec_tfidf), 4),
            "recall": round(float(val_rec_tfidf), 4),
            "f1_score": round(float(val_f1_tfidf), 4)
        },
        "confusion_matrix": conf_matrix_tfidf.tolist(),
        "classification_report": class_report_tfidf
    },

    "random_forest_tfidf_features": {
        "training_time_seconds": round(float(time_full), 4),
        "metrics": {
            "accuracy": round(float(acc_full), 4),
            "precision": round(float(prec_full), 4),
            "recall": round(float(rec_full), 4),
            "f1_score": round(float(f1_full), 4),
            "macro_f1_score": round(float(macro_f1_full), 4),
            "weighted_f1_score": round(float(weighted_f1_full), 4),
            "roc_auc": round(float(roc_auc_full), 4)
        },
        "train_metrics": {
            "accuracy": round(float(train_acc_full), 4),
            "precision": round(float(train_prec_full), 4),
            "recall": round(float(train_rec_full), 4),
            "f1_score": round(float(train_f1_full), 4)
        },
        "val_metrics": {
            "accuracy": round(float(val_acc_full), 4),
            "precision": round(float(val_prec_full), 4),
            "recall": round(float(val_rec_full), 4),
            "f1_score": round(float(val_f1_full), 4)
        },
        "confusion_matrix": conf_matrix_full.tolist(),
        "classification_report": class_report_full
    }

}

with open("random_forest_results.json", "w", encoding="utf-8") as f:

    json.dump(
        results,
        f,
        indent=4,
        ensure_ascii=False
    )

print("\nResultados guardados en:")
print("random_forest_results.json")

# =========================================================
# GRÁFICAS DE EVALUACIÓN
# =========================================================

print("\n================================================")
print("GENERANDO GRÁFICAS DE EVALUACIÓN")
print("================================================")

os.makedirs("graficas/random_forest", exist_ok=True)

# =========================================================
# 31. Curvas de convergencia con n_estimators (modelo full)
# =========================================================
# warm_start=True permite añadir árboles incrementalmente sin reentrenar
# n_estimators es el equivalente a "epochs" en Random Forest

tree_checkpoints = [10, 25, 50, 75, 100, 150, 200, 250, 300]

conv_train_losses     = []
conv_val_losses       = []
conv_train_precisions = []
conv_val_precisions   = []

conv_rf = RandomForestClassifier(
    n_estimators=1,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
    warm_start=True
)

for n_trees in tree_checkpoints:

    conv_rf.set_params(n_estimators=n_trees)
    conv_rf.fit(X_train_final, y_train)

    tp = conv_rf.predict_proba(X_train_final)
    vp = conv_rf.predict_proba(X_val_final)

    conv_train_losses.append(log_loss(y_train, tp))
    conv_val_losses.append(log_loss(y_val, vp))

    conv_train_precisions.append(
        precision_score(y_train, conv_rf.predict(X_train_final))
    )
    conv_val_precisions.append(
        precision_score(y_val, conv_rf.predict(X_val_final))
    )

# Loss curve
plt.figure(figsize=(8, 6))
plt.plot(tree_checkpoints, conv_train_losses, marker='o', label="Train Loss")
plt.plot(tree_checkpoints, conv_val_losses,   marker='o', label="Validación Loss")
plt.xlabel("Número de árboles (n_estimators)")
plt.ylabel("Log Loss")
plt.title("Curva de Aprendizaje - Loss - Random Forest")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "graficas/random_forest/learning_curve_loss.png",
    bbox_inches='tight'
)
plt.close()

print("[OK] learning_curve_loss.png")

# Precision curve
plt.figure(figsize=(8, 6))
plt.plot(tree_checkpoints, conv_train_precisions, marker='o', label="Train Precision")
plt.plot(tree_checkpoints, conv_val_precisions,   marker='o', label="Validación Precision")
plt.xlabel("Número de árboles (n_estimators)")
plt.ylabel("Precision")
plt.title("Curva de Aprendizaje - Precisión - Random Forest")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "graficas/random_forest/learning_curve_precision.png",
    bbox_inches='tight'
)
plt.close()

print("[OK] learning_curve_precision.png")

# -------------------------
# 32. Barras: métricas Train / Validación / Test (modelo full)
# -------------------------
metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-score']

train_values = [train_acc_full, train_prec_full, train_rec_full, train_f1_full]
val_values   = [val_acc_full,   val_prec_full,   val_rec_full,   val_f1_full]
test_values  = [acc_full,       prec_full,       rec_full,       f1_full]

x = np.arange(len(metrics_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 6))

ax.bar(x - width, train_values, width, label='Train')
ax.bar(x,         val_values,   width, label='Validación')
ax.bar(x + width, test_values,  width, label='Test')

ax.set_xlabel("Métrica")
ax.set_ylabel("Valor")
ax.set_title("Métricas Train / Validación / Test - RF TF-IDF + Features")
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis='y')
plt.tight_layout()
plt.savefig(
    "graficas/random_forest/metrics_comparison.png",
    bbox_inches='tight'
)
plt.close()

print("[OK] metrics_comparison.png")

# -------------------------
# 33. Matriz de Confusión - Solo TF-IDF
# -------------------------
plt.figure(figsize=(7, 5))

sns.heatmap(
    conf_matrix_tfidf,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['REAL', 'FAKE'],
    yticklabels=['REAL', 'FAKE']
)

plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión - RF Solo TF-IDF")
plt.tight_layout()
plt.savefig(
    "graficas/random_forest/confusion_matrix_tfidf.png",
    bbox_inches='tight'
)
plt.close()

print("[OK] confusion_matrix_tfidf.png")

# -------------------------
# 34. Matriz de Confusión - TF-IDF + Features
# -------------------------
plt.figure(figsize=(7, 5))

sns.heatmap(
    conf_matrix_full,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['REAL', 'FAKE'],
    yticklabels=['REAL', 'FAKE']
)

plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión - RF TF-IDF + Features")
plt.tight_layout()
plt.savefig(
    "graficas/random_forest/confusion_matrix_full.png",
    bbox_inches='tight'
)
plt.close()

print("[OK] confusion_matrix_full.png")

# -------------------------
# 35. Curva ROC-AUC - Solo TF-IDF
# -------------------------
fpr_t, tpr_t, _ = roc_curve(y_test, y_probs_tfidf)

plt.figure(figsize=(8, 6))

plt.plot(fpr_t, tpr_t, label=f"ROC-AUC = {roc_auc_tfidf:.4f}")
plt.plot([0, 1], [0, 1], 'k--', label="Clasificador aleatorio")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC-AUC - RF Solo TF-IDF")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "graficas/random_forest/roc_auc_curve_tfidf.png",
    bbox_inches='tight'
)
plt.close()

print("[OK] roc_auc_curve_tfidf.png")

# -------------------------
# 36. Curva ROC-AUC - TF-IDF + Features
# -------------------------
fpr_f, tpr_f, _ = roc_curve(y_test, y_probs_full)

plt.figure(figsize=(8, 6))

plt.plot(fpr_f, tpr_f, label=f"ROC-AUC = {roc_auc_full:.4f}")
plt.plot([0, 1], [0, 1], 'k--', label="Clasificador aleatorio")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC-AUC - RF TF-IDF + Features")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "graficas/random_forest/roc_auc_curve_full.png",
    bbox_inches='tight'
)
plt.close()

print("[OK] roc_auc_curve_full.png")

print("\nGráficas guardadas en: graficas/random_forest/")

# =========================================================
# 37. EXPLICABILIDAD (SHAP + LIME)
# =========================================================

import shap

from lime.lime_text import LimeTextExplainer

# =========================================================
# CREAR CARPETA
# =========================================================

os.makedirs("xai/random_forest", exist_ok=True)

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

X_shap_sample = X_test_final[:sample_size].toarray()

explainer = shap.TreeExplainer(model_full)

shap_values = explainer.shap_values(X_shap_sample)

# RandomForest binario -> clase FAKE
if isinstance(shap_values, list):

    shap_values_plot = shap_values[1]

else:

    shap_values_plot = shap_values

# -------------------------
# SHAP SUMMARY PLOT
# -------------------------
plt.figure()

shap.summary_plot(
    shap_values_plot,
    X_shap_sample,
    feature_names=all_feature_names,
    max_display=30,
    show=False
)

plt.tight_layout()

plt.savefig(
    "xai/random_forest/shap_summary_plot.png",
    bbox_inches='tight'
)

plt.close()

print("[OK] shap_summary_plot.png")

# -------------------------
# SHAP BAR PLOT
# -------------------------
plt.figure()

shap.summary_plot(
    shap_values_plot,
    X_shap_sample,
    feature_names=all_feature_names,
    plot_type="bar",
    max_display=30,
    show=False
)

plt.tight_layout()

plt.savefig(
    "xai/random_forest/shap_bar_plot.png",
    bbox_inches='tight'
)

plt.close()

print("[OK] shap_bar_plot.png")

# -------------------------
# SAMPLE MÁS CONFIADO
# -------------------------
confidence_scores = np.abs(
    y_probs_full[:sample_size] - 0.5
)

sample_index = np.argmax(confidence_scores)

# -------------------------
# SHAP WATERFALL
# -------------------------
sample_dense = X_shap_sample[sample_index]

sample_shap_values = shap_values_plot[
    sample_index, :, 1
]

explanation = shap.Explanation(
    values=sample_shap_values,
    base_values=explainer.expected_value[1],
    data=sample_dense,
    feature_names=all_feature_names
)

plt.figure()

shap.plots.waterfall(
    explanation,
    show=False
)

plt.savefig(
    "xai/random_forest/shap_waterfall_plot.png",
    bbox_inches='tight'
)

plt.close()

print("[OK] shap_waterfall_plot.png")

with open(
    "xai/random_forest/shap_explained_text.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("===== TEXTO ORIGINAL =====\n\n")
    f.write(X_text_full_test.iloc[sample_index])

    f.write("\n\n===== TEXTO PROCESADO =====\n\n")
    f.write(X_text_test.iloc[sample_index])

print("[OK] shap_explained_text.txt")

# -------------------------
# TOP FEATURES JSON
# -------------------------
mean_abs_shap = np.abs(
    shap_values_plot[:, :, 1]
).mean(axis=0)

top_idx = np.argsort(mean_abs_shap)[::-1][:20]

top_features = []

for idx in top_idx:

    top_features.append({
        "feature": all_feature_names[idx],
        "importance": float(mean_abs_shap[idx])
    })

with open(
    "xai/random_forest/top_features_shap.json",
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
# VALORES NUMÉRICOS REALES DEL SAMPLE
# =========================================================

# lime usa sample_index (el más confiado); sus features numéricas
# se mantienen constantes en cada perturbación de texto
lime_sample_num = X_test_num[sample_index: sample_index + 1]

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
    n = len(processed_texts)
    numeric_repeated = np.tile(lime_sample_num, (n, 1))

    # combinar
    final = hstack([tfidf, numeric_repeated]).tocsr()

    probs = model_full.predict_proba(final)

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
    "xai/random_forest/lime_explanation.html"
)

print("[OK] lime_explanation.html")

# =========================================================
# GUARDAR FEATURES JSON
# =========================================================

lime_data = {
    "original_text": X_text_full_test.iloc[sample_index],
    "processed_text": X_text_test.iloc[sample_index],
    "features": []
}

for feature, weight in lime_exp.as_list():

    lime_data["features"].append({
        "feature": feature,
        "weight": float(weight)
    })

with open(
    "xai/random_forest/lime_features.json",
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
print("- shap_explained_text.txt")
print("- top_features_shap.json")

print("\nLIME:")
print("- lime_explanation.html")
print("- lime_features.json")
