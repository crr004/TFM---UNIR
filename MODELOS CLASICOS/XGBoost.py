import pandas as pd
import numpy as np
import time
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

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

from sklearn.preprocessing import StandardScaler

from scipy.sparse import hstack

import xgboost as xgb
from xgboost import XGBClassifier

from tqdm import tqdm


class TqdmCallback(xgb.callback.TrainingCallback):
    """Barra de progreso tqdm para el entrenamiento de XGBoost."""
    def __init__(self, total, desc="Entrenando XGBoost"):
        self.pbar = tqdm(total=total, desc=desc, unit="ronda")

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False

    def after_training(self, model):
        self.pbar.close()
        return model

# =========================================================
# 1. Cargar datos
# =========================================================
print("[1/6] Cargando datos...")
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

# =========================================================
# 2. Variables
# =========================================================
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

print("[2/6] Vectorizando TF-IDF...")
# FIT SOLO TRAIN
X_train_tfidf = vectorizer.fit_transform(X_text_train)

# TRANSFORM
X_val_tfidf  = vectorizer.transform(X_text_val)
X_test_tfidf = vectorizer.transform(X_text_test)

# =========================================================
# NORMALIZACIÓN
# =========================================================

# -------------------------
# 6. Normalización numérica
# -------------------------
scaler = StandardScaler()

print("[3/6] Normalizando features numéricas...")
# FIT SOLO TRAIN
X_train_num = scaler.fit_transform(X_num_train)

# TRANSFORM
X_val_num  = scaler.transform(X_num_val)
X_test_num = scaler.transform(X_num_test)

# =========================================================
# COMBINACIÓN
# =========================================================

# -------------------------
# 7. Combinar
# -------------------------
print("[4/6] Combinando features TF-IDF + numéricas...")
X_train_final = hstack([X_train_tfidf, X_train_num]).tocsr()
X_val_final   = hstack([X_val_tfidf,   X_val_num  ]).tocsr()
X_test_final  = hstack([X_test_tfidf,  X_test_num ]).tocsr()

# =========================================================
# MODELO
# =========================================================

# -------------------------
# 8. Modelo XGBoost
# -------------------------
model = XGBClassifier(
    n_estimators=1000,          # máximo; early stopping decidirá cuándo parar
    early_stopping_rounds=30,   # para si val_loss no mejora 30 rondas seguidas
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    n_jobs=-1,
    random_state=42,
    callbacks=[TqdmCallback(1000)]
)

# =========================================================
# ENTRENAMIENTO
# =========================================================

# -------------------------
# 9. Entrenamiento con eval_set para capturar logloss por ronda
# -------------------------
print("[5/6] Entrenando XGBoost (máx. 1000 rondas, early stopping=30)...")
start_time = time.time()

model.fit(
    X_train_final,
    y_train,
    eval_set=[
        (X_train_final, y_train),
        (X_val_final,   y_val)
    ],
    verbose=False
)

end_time = time.time()

training_time = end_time - start_time

# =========================================================
# VALIDACIÓN -> THRESHOLD
# =========================================================

# -------------------------
# 10. Probabilidades VALIDACIÓN
# -------------------------
y_val_probs = model.predict_proba(X_val_final)[:, 1]

best_f1 = 0
best_threshold = 0

# -------------------------
# 11. Buscar mejor threshold
# -------------------------
for t in np.arange(0.3, 0.7, 0.01):

    y_val_pred_temp = (y_val_probs >= t).astype(int)

    f1_temp = f1_score(y_val, y_val_pred_temp)

    if f1_temp > best_f1:

        best_f1 = f1_temp
        best_threshold = t

# -------------------------
# 12. Predicción VALIDACIÓN con mejor threshold
# -------------------------
y_val_pred = (y_val_probs >= best_threshold).astype(int)

# =========================================================
# TEST
# =========================================================

# -------------------------
# 13. Probabilidades TEST
# -------------------------
y_probs = model.predict_proba(X_test_final)[:, 1]

# -------------------------
# 14. Predicción TEST
# -------------------------
y_pred = (y_probs >= best_threshold).astype(int)

# =========================================================
# MÉTRICAS
# =========================================================

# -------------------------
# 15. Métricas TEST
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
# 16. ROC-AUC
# -------------------------
roc_auc = roc_auc_score(
    y_test,
    y_probs
)

# -------------------------
# 17. Classification Report
# -------------------------
class_report = classification_report(
    y_test,
    y_pred
)

# -------------------------
# 18. Confusion Matrix
# -------------------------
conf_matrix = confusion_matrix(
    y_test,
    y_pred
)

# =========================================================
# RESULTADOS
# =========================================================

# -------------------------
# 19. Resultados
# -------------------------
print("\n================================================")
print("RESULTADOS - XGBOOST")
print("================================================")

print(f"\nMejor threshold: {best_threshold:.2f}")
print(f"\nTiempo de entrenamiento: {training_time:.4f} segundos")

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
# 20. Métricas TRAIN
# -------------------------
y_train_probs = model.predict_proba(X_train_final)[:, 1]
y_train_pred  = (y_train_probs >= best_threshold).astype(int)

train_accuracy  = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall    = recall_score(y_train, y_train_pred)
train_f1        = f1_score(y_train, y_train_pred)

# -------------------------
# 21. Métricas VALIDACIÓN
# -------------------------
val_accuracy  = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred)
val_recall    = recall_score(y_val, y_val_pred)
val_f1        = f1_score(y_val, y_val_pred)

# -------------------------
# 22. Comparación TRAIN vs VAL vs TEST
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
# GUARDAR JSON
# =========================================================

# -------------------------
# 23. Guardar JSON
# -------------------------
results = {

    "model": "XGBoost",
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

with open("xgboost_results.json", "w", encoding="utf-8") as f:

    json.dump(
        results,
        f,
        indent=4,
        ensure_ascii=False
    )

print("\nResultados guardados en:")
print("xgboost_results.json")

# =========================================================
# GRÁFICAS DE EVALUACIÓN
# =========================================================

print("[6/6] Generando gráficas y explicaciones XAI...")

print("\n================================================")
print("GENERANDO GRÁFICAS DE EVALUACIÓN")
print("================================================")

os.makedirs("graficas/xgboost", exist_ok=True)

# =========================================================
# 24. Curva de Loss por ronda de boosting (eval_set)
# =========================================================
# XGBoost registra la logloss exacta en cada ronda durante el fit
# Eje X = boosting round (equivalente directo a epoch)

evals = model.evals_result()

train_logloss = evals['validation_0']['logloss']
val_logloss   = evals['validation_1']['logloss']
rounds        = list(range(1, len(train_logloss) + 1))

best_iter = model.best_iteration + 1

plt.figure(figsize=(8, 6))
plt.plot(rounds, train_logloss, label="Train Loss")
plt.plot(rounds, val_logloss,   label="Validación Loss")
plt.axvline(x=best_iter, color='red', linestyle='--', label=f"Early stop (ronda {best_iter})")
plt.xlabel("Ronda de boosting (n_estimators)")
plt.ylabel("Log Loss")
plt.title("Curva de Aprendizaje - Loss - XGBoost")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "graficas/xgboost/learning_curve_loss.png",
    bbox_inches='tight'
)
plt.close()

print("[OK] learning_curve_loss.png")

# =========================================================
# 25. Curva de Precision por checkpoints de ronda
# =========================================================
# Los checkpoints van hasta model.best_iteration (donde paró el early stopping)
# iteration_range permite predecir con los primeros N árboles sin reentrenar

best_iter = model.best_iteration + 1   # +1 porque best_iteration es 0-indexed
step = max(10, best_iter // 10)
tree_checkpoints = list(range(step, best_iter + step, step))
tree_checkpoints = [t for t in tree_checkpoints if t <= best_iter]
if best_iter not in tree_checkpoints:
    tree_checkpoints.append(best_iter)

conv_train_precisions = []
conv_val_precisions   = []

for n_trees in tqdm(tree_checkpoints, desc="Checkpoints precisión"):

    tp = model.predict_proba(
        X_train_final,
        iteration_range=(0, n_trees)
    )[:, 1]

    vp = model.predict_proba(
        X_val_final,
        iteration_range=(0, n_trees)
    )[:, 1]

    conv_train_precisions.append(
        precision_score(y_train, (tp >= best_threshold).astype(int))
    )
    conv_val_precisions.append(
        precision_score(y_val, (vp >= best_threshold).astype(int))
    )

plt.figure(figsize=(8, 6))
plt.plot(tree_checkpoints, conv_train_precisions, marker='o', label="Train Precision")
plt.plot(tree_checkpoints, conv_val_precisions,   marker='o', label="Validación Precision")
plt.axvline(x=best_iter, color='red', linestyle='--', label=f"Early stop (ronda {best_iter})")
plt.xlabel("Ronda de boosting (n_estimators)")
plt.ylabel("Precision")
plt.title("Curva de Aprendizaje - Precisión - XGBoost")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "graficas/xgboost/learning_curve_precision.png",
    bbox_inches='tight'
)
plt.close()

print("[OK] learning_curve_precision.png")

# =========================================================
# Curvas por tamaño del conjunto de entrenamiento
# =========================================================
# El vectorizer y el scaler ya están ajustados sobre el 100% de train.
# XGBoost usa log_loss con predict_proba y best_threshold para F1/Precision.

from sklearn.metrics import log_loss as sklearn_log_loss

n_train_total = X_train_final.shape[0]
lc_fractions  = np.linspace(0.1, 1.0, 10)
lc_sizes      = [max(50, int(f * n_train_total)) for f in lc_fractions]

lc_train_losses     = []
lc_val_losses       = []
lc_train_precisions = []
lc_val_precisions   = []
lc_train_f1s        = []
lc_val_f1s          = []

for n in tqdm(lc_sizes, desc="Curvas por tamaño"):

    X_sub = X_train_final[:n]
    y_sub = y_train.iloc[:n]

    lc_xgb = XGBClassifier(
        n_estimators=best_iter,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42
    )
    lc_xgb.fit(X_sub, y_sub, verbose=False)

    tp = lc_xgb.predict_proba(X_sub)
    vp = lc_xgb.predict_proba(X_val_final)

    lc_train_losses.append(sklearn_log_loss(y_sub, tp))
    lc_val_losses.append(sklearn_log_loss(y_val,   vp))

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
plt.title("Curva por Tamaño - Loss - XGBoost")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "graficas/xgboost/lc_size_loss.png",
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
plt.title("Curva por Tamaño - Precisión - XGBoost")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "graficas/xgboost/lc_size_precision.png",
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
plt.title("Curva por Tamaño - F1 - XGBoost")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "graficas/xgboost/lc_size_f1.png",
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
ax.set_title("Métricas Train / Validación / Test - XGBoost")
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis='y')
plt.tight_layout()
plt.savefig(
    "graficas/xgboost/metrics_comparison.png",
    bbox_inches='tight'
)
plt.close()

print("[OK] metrics_comparison.png")

# -------------------------
# 27. Matriz de Confusión
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
plt.title("Matriz de Confusión - XGBoost")
plt.tight_layout()
plt.savefig(
    "graficas/xgboost/confusion_matrix.png",
    bbox_inches='tight'
)
plt.close()

print("[OK] confusion_matrix.png")

# -------------------------
# 28. Curva ROC-AUC
# -------------------------
fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.figure(figsize=(8, 6))

plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], 'k--', label="Clasificador aleatorio")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC-AUC - XGBoost")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "graficas/xgboost/roc_auc_curve.png",
    bbox_inches='tight'
)
plt.close()

print("[OK] roc_auc_curve.png")

print("\nGráficas guardadas en: graficas/xgboost/")

# =========================================================
# 29. EXPLICABILIDAD (SHAP + LIME)
# =========================================================

import shap

from lime.lime_text import LimeTextExplainer

# =========================================================
# CREAR CARPETA
# =========================================================

os.makedirs("xai/xgboost", exist_ok=True)

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

explainer = shap.TreeExplainer(model)

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
    "xai/xgboost/shap_summary_plot.png",
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
    "xai/xgboost/shap_bar_plot.png",
    bbox_inches='tight'
)

plt.close()

print("[OK] shap_bar_plot.png")

# -------------------------
# SAMPLE FAKE CORRECTAMENTE CLASIFICADA
# -------------------------
sample_probs = y_probs[:sample_size]

pred_fake = sample_probs >= best_threshold
real_fake  = y_test.iloc[:sample_size].values == 1

correct_fake_indices = np.where(pred_fake & real_fake)[0]

fake_confidences = np.abs(sample_probs[correct_fake_indices] - 0.5)

sample_index = correct_fake_indices[np.argmax(fake_confidences)]

# -------------------------
# SHAP WATERFALL
# -------------------------
sample_dense      = X_shap_sample[sample_index]
sample_shap_vals  = shap_values[sample_index]

explanation = shap.Explanation(
    values=sample_shap_vals,
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
    "xai/xgboost/shap_waterfall_plot.png",
    bbox_inches='tight'
)

plt.close()

print("[OK] shap_waterfall_plot.png")

with open(
    "xai/xgboost/shap_explained_text.txt",
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
mean_abs_shap = np.abs(shap_values).mean(axis=0)

top_idx = np.argsort(mean_abs_shap)[::-1][:20]

top_features = []

for idx in top_idx:

    top_features.append({
        "feature": all_feature_names[idx],
        "importance": float(mean_abs_shap[idx])
    })

with open(
    "xai/xgboost/top_features_shap.json",
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

# sample_index viene de la sección SHAP (fake más confiada)
# LIME solo perturba el texto; las features numéricas se mantienen constantes
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

    probs = model.predict_proba(final)

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
    "xai/xgboost/lime_explanation.html"
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
    "xai/xgboost/lime_features.json",
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
