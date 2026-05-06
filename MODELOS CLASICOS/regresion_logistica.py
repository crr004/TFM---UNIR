import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import json

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
df = df[['text_ml', 'label'] + numeric_features].dropna()

# -------------------------
# 4. Variables
# -------------------------
X_text = df['text_ml']
X_num = df[numeric_features]
y = df['label']

# =========================================================
# SPLIT 70 / 10 / 20
# =========================================================

# -------------------------
# 5. Primer split -> TRAIN (70) y TEMP (30)
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
# 6. Segundo split -> VALIDATION (10) y TEST (20)
# -------------------------
X_text_val, X_text_test, X_num_val, X_num_test, y_val, y_test = train_test_split(
    X_text_temp,
    X_num_temp,
    y_temp,
    test_size=2/3,  # 20% test y 10% val del total
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

# -------------------------
# 26. Plot
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
plt.title("Learning Curve - Regresión Logística")
plt.legend()
plt.grid()
plt.show()

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