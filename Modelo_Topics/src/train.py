# ==============================================================================
# train.py — Entrenamiento del Modelo XGBoost
# ==============================================================================
#
# Responsabilidades:
#   - Instanciar y configurar el clasificador XGBoost multiclase.
#   - Entrenar con early stopping y monitorización de log-loss.
#   - Generar la gráfica de curvas de aprendizaje.
#   - Guardar el modelo entrenado en disco.
#
# ==============================================================================

import os
import time

import matplotlib.pyplot as plt
import xgboost as xgb


# ==============================================================================
# DETECCIÓN AUTOMÁTICA DE DISPOSITIVO (GPU / CPU)
# ==============================================================================

def _detect_device():
    """
    Intenta usar CUDA (GPU). Si no está disponible, hace fallback a CPU
    y lo notifica por consola.
    """
    try:
        # Intentamos crear un DMatrix mínimo con device='cuda'
        # para verificar que CUDA está operativo.
        import numpy as np
        from scipy.sparse import csr_matrix

        dummy = csr_matrix(np.array([[1.0, 2.0]], dtype=np.float32))
        test_model = xgb.XGBClassifier(
            n_estimators=1, tree_method='hist', device='cuda'
        )
        test_model.fit(dummy, np.array([0]))
        print("[Train] GPU CUDA detectada y operativa. Se usará aceleración por GPU.")
        return 'cuda'
    except Exception:
        print("[Train] GPU CUDA no disponible. Se usará CPU para el entrenamiento.")
        return 'cpu'


# ==============================================================================
# ENTRENAMIENTO
# ==============================================================================

def train_xgboost(
    X_train, y_train,
    X_test, y_test,
    label_classes,
    output_dir,
    figures_dir,
    n_estimators=30000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=20,
    random_state=42,
):
    """
    Entrena un clasificador XGBoost multiclase con la configuración dada.

    Parámetros:
        X_train, y_train: Datos de entrenamiento (matriz TF-IDF + etiquetas).
        X_test, y_test:   Datos de evaluación para early stopping.
        label_classes:    Array con los nombres de las clases (del LabelEncoder).
        output_dir:       Carpeta donde guardar el modelo.
        figures_dir:      Carpeta donde guardar la gráfica de aprendizaje.
        n_estimators:     Número máximo de árboles.
        max_depth:        Profundidad máxima de los árboles.
        learning_rate:    Tasa de aprendizaje.
        subsample:        Fracción de muestras por árbol.
        colsample_bytree: Fracción de features por árbol.
        early_stopping_rounds: Rondas sin mejora antes de detener.
        random_state:     Semilla para reproducibilidad.

    Retorna:
        El modelo XGBoost entrenado.
    """
    device = _detect_device()
    num_classes = len(label_classes)

    print(f"\n[Train] Configuración del modelo:")
    print(f"  - n_estimators:          {n_estimators}")
    print(f"  - max_depth:             {max_depth}")
    print(f"  - learning_rate:         {learning_rate}")
    print(f"  - subsample:             {subsample}")
    print(f"  - colsample_bytree:      {colsample_bytree}")
    print(f"  - early_stopping_rounds: {early_stopping_rounds}")
    print(f"  - device:                {device}")
    print(f"  - num_class:             {num_classes}")

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective='multi:softprob',
        num_class=num_classes,
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=-1,
        tree_method='hist',
        device=device,
        eval_metric='mlogloss',
    )

    evals = [(X_train, y_train), (X_test, y_test)]

    print("\n[Train] Iniciando entrenamiento...")
    start_time = time.time()

    model.fit(
        X_train, y_train,
        eval_set=evals,
        verbose=50,
    )

    elapsed = time.time() - start_time
    print(f"\n[Train] ¡Entrenamiento completado en {elapsed:.2f} segundos!")

    # --- Guardar modelo ---
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "xgboost_topic_classifier.pkl")
    model.save_model(model_path)
    print(f"[Train] Modelo guardado en: {model_path}")

    # --- Curvas de aprendizaje ---
    _plot_learning_curves(model, figures_dir)

    return model


def _plot_learning_curves(model, figures_dir):
    """Genera y guarda la gráfica de curvas de aprendizaje (log-loss)."""
    os.makedirs(figures_dir, exist_ok=True)

    results = model.evals_result()
    epochs = len(results['validation_0']['mlogloss'])
    x_axis = range(0, epochs)

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, results['validation_0']['mlogloss'], label='Entrenamiento')
    plt.plot(x_axis, results['validation_1']['mlogloss'], label='Prueba')
    plt.legend()
    plt.ylabel('Pérdida (Log Loss)')
    plt.xlabel('Número de Árboles (Iteraciones)')
    plt.title('Curvas de Aprendizaje de XGBoost')
    plt.tight_layout()

    fig_path = os.path.join(figures_dir, "curvas_aprendizaje_xgboost.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Train] Gráfica de aprendizaje guardada en: {fig_path}")
