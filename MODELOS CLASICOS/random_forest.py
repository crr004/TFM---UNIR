import pandas as pd
import numpy as np
import time
import json

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier

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

df = df[['text_ml', 'label'] + numeric_features].dropna()

# -------------------------
# 2. Variables
# -------------------------
X_text = df['text_ml']
X_num = df[numeric_features]
y = df['label']

# =========================================================
# SPLIT 70 / 10 / 20
# =========================================================

# -------------------------
# 3. TRAIN (70) y TEMP (30)
# -------------------------
X_text_train, X_text_temp, X_num_train, X_num_temp, y_train, y_temp = train_test_split(
    X_text,
    X_num,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# -------------------------
# 4. VALIDATION (10) y TEST (20)
# -------------------------
X_text_val, X_text_test, X_num_val, X_num_test, y_val, y_test = train_test_split(
    X_text_temp,
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
# 10. Métricas principales
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
# 14. Train
# -------------------------
y_train_pred_tfidf = model_tfidf.predict(X_train_tfidf)

train_f1_tfidf = f1_score(
    y_train,
    y_train_pred_tfidf
)

# =========================================================
# MODELO 2: TF-IDF + FEATURES
# =========================================================

# -------------------------
# 15. Normalización
# -------------------------
scaler = StandardScaler()

# FIT SOLO TRAIN
X_train_num = scaler.fit_transform(X_num_train)

# TRANSFORM
X_val_num = scaler.transform(X_num_val)
X_test_num = scaler.transform(X_num_test)

# -------------------------
# 16. Combinar
# -------------------------
X_train_final = hstack([X_train_tfidf, X_train_num])

X_val_final = hstack([X_val_tfidf, X_val_num])

X_test_final = hstack([X_test_tfidf, X_test_num])

# -------------------------
# 17. Modelo Full
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
# 18. Entrenamiento
# -------------------------
start = time.time()

model_full.fit(X_train_final, y_train)

time_full = time.time() - start

# -------------------------
# 19. Predicción TEST
# -------------------------
y_pred_full = model_full.predict(X_test_final)

# -------------------------
# 20. Probabilidades TEST
# -------------------------
y_probs_full = model_full.predict_proba(X_test_final)[:, 1]

# =========================================================
# MÉTRICAS FULL
# =========================================================

# -------------------------
# 21. Métricas principales
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
# 22. ROC-AUC
# -------------------------
roc_auc_full = roc_auc_score(
    y_test,
    y_probs_full
)

# -------------------------
# 23. Classification Report
# -------------------------
class_report_full = classification_report(
    y_test,
    y_pred_full
)

# -------------------------
# 24. Confusion Matrix
# -------------------------
conf_matrix_full = confusion_matrix(
    y_test,
    y_pred_full
)

# -------------------------
# 25. Train
# -------------------------
y_train_pred_full = model_full.predict(X_train_final)

train_f1_full = f1_score(
    y_train,
    y_train_pred_full
)

# =========================================================
# RESULTADOS
# =========================================================

# -------------------------
# 26. Resultados
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
print(f"F1 Train: {train_f1_tfidf:.4f}")

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
print(f"F1 Train: {train_f1_full:.4f}")

print("\nClassification Report:")
print(class_report_full)

print("\nConfusion Matrix:")
print(conf_matrix_full)

# =========================================================
# DIFERENCIAS
# =========================================================

# -------------------------
# 27. Diferencias
# -------------------------
print("\n--- DIFERENCIA ---")
print(f"Δ F1: {f1_full - f1_tfidf:.4f}")
print(f"Δ Precision: {prec_full - prec_tfidf:.4f}")
print(f"Δ Recall: {rec_full - rec_tfidf:.4f}")

# =========================================================
# OVERFITTING
# =========================================================

# -------------------------
# 28. Overfitting
# -------------------------
print("\n===== OVERFITTING CHECK =====")

print("\n--- SOLO TF-IDF ---")
print(f"Train F1: {train_f1_tfidf:.4f}")
print(f"Test F1:  {f1_tfidf:.4f}")

print("\n--- TF-IDF + FEATURES ---")
print(f"Train F1: {train_f1_full:.4f}")
print(f"Test F1:  {f1_full:.4f}")

# -------------------------
# 29. Gap
# -------------------------
gap_tfidf = train_f1_tfidf - f1_tfidf

gap_full = train_f1_full - f1_full

print("\n===== GAP TRAIN vs TEST =====")

print("\n--- SOLO TF-IDF ---")
print(f"Gap F1: {gap_tfidf:.4f}")

print("\n--- TF-IDF + FEATURES ---")
print(f"Gap F1: {gap_full:.4f}")

# =========================================================
# JSON
# =========================================================

# -------------------------
# 30. Diccionario resultados
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

        "train_f1": round(float(train_f1_tfidf), 4),
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

        "train_f1": round(float(train_f1_full), 4),
        "confusion_matrix": conf_matrix_full.tolist(),
        "classification_report": class_report_full

    }

}

# -------------------------
# 31. Guardar JSON
# -------------------------
with open("random_forest_results.json", "w", encoding="utf-8") as f:

    json.dump(
        results,
        f,
        indent=4,
        ensure_ascii=False
    )

print("\nResultados guardados en:")
print("random_forest_results.json")