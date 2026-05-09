# ==============================================================================
# evaluate.py — Evaluación del Modelo XGBoost
# ==============================================================================
#
# Responsabilidades:
#   - Cargar el modelo entrenado desde disco.
#   - Generar predicciones sobre el conjunto de test.
#   - Imprimir el Classification Report (precision, recall, F1).
#   - Generar y guardar la Matriz de Confusión como heatmap profesional.
#
# ==============================================================================

import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.metrics import classification_report, confusion_matrix


# ==============================================================================
# EVALUACIÓN
# ==============================================================================

def evaluate_model(model_path, X_test, y_test, label_encoder, figures_dir):
    """
    Evalúa el modelo XGBoost sobre el conjunto de test.

    Parámetros:
        model_path:     Ruta al archivo del modelo serializado.
        X_test:         Matriz TF-IDF del conjunto de test.
        y_test:         Etiquetas numéricas del conjunto de test.
        label_encoder:  LabelEncoder ajustado (para decodificar las clases).
        figures_dir:    Carpeta donde guardar la matriz de confusión.
    """
    # --- Cargar modelo ---
    print(f"[Evaluación] Cargando modelo desde: {model_path}")
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # --- Predicciones ---
    print("[Evaluación] Realizando predicciones sobre el conjunto de test...")
    y_pred = model.predict(X_test)

    # --- Classification Report ---
    class_names = label_encoder.classes_
    print("\n" + "=" * 60)
    print("  REPORTE DE CLASIFICACIÓN")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names))

    # --- Matriz de Confusión ---
    _plot_confusion_matrix(y_test, y_pred, class_names, figures_dir)


def _plot_confusion_matrix(y_test, y_pred, class_names, figures_dir):
    """
    Genera y guarda una matriz de confusión con estética profesional
    (misma configuración que el notebook original).
    """
    os.makedirs(figures_dir, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)

    # Configuración estética
    TAMANO_LABELS = 12
    TAMANO_CLASES = 12
    TAMANO_NUMEROS = 10

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    mpl.rcParams['axes.labelsize'] = TAMANO_LABELS
    mpl.rcParams['xtick.labelsize'] = TAMANO_CLASES
    mpl.rcParams['ytick.labelsize'] = TAMANO_CLASES
    sns.set_theme(style="white")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={
            "size": TAMANO_NUMEROS,
            "fontweight": "bold",
            "family": "sans-serif",
        },
        linewidths=0.8,
        linecolor='gray',
        cbar=False,
        square=True,
    )
    ax.set_xlabel('CLASE PREDICHA', labelpad=5, fontweight='bold')
    ax.set_ylabel('CLASE REAL', labelpad=10, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    fig_path = os.path.join(figures_dir, "matriz_confusion_topic_classifier.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[Evaluación] Matriz de confusión guardada en: {fig_path}")
