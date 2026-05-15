# ==============================================================================
# main.py — Orquestador Principal (CLI)
# ==============================================================================
#
# Punto de entrada del paquete encapsulado.
# Expone 4 comandos a través de argparse:
#
#   python -m src.main preprocess   → Carga, limpieza, TF-IDF, split, guardado.
#   python -m src.main train        → Entrena el modelo XGBoost.
#   python -m src.main evaluate     → Evalúa el modelo con el set de test.
#   python -m src.main predict      → Clasifica una noticia nueva.
#
# ==============================================================================

import argparse
import os
import sys
import pickle

from .preprocessing import load_and_prepare_data, build_tfidf_and_split, save_artifacts
from .train import train_xgboost
from .evaluate import evaluate_model
from .predict import load_inference_pipeline, classify_news


# ==============================================================================
# RUTAS POR DEFECTO
# ==============================================================================

# Se resuelven relativamente al directorio de este archivo (src/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

DEFAULT_DATA_PATH = os.path.join(_PROJECT_DIR, "..", "Data", "Topic", "Dataset_Topic_Classifier.csv")
DEFAULT_ARTIFACTS_DIR = os.path.join(_PROJECT_DIR, "artefactos")
DEFAULT_FIGURES_DIR = os.path.join(_PROJECT_DIR, "figuras")
DEFAULT_CACHE_DIR = os.path.join(_PROJECT_DIR, "artefactos", "cache")


# ==============================================================================
# FUNCIONES DE COMANDO
# ==============================================================================

def cmd_preprocess(args):
    """Ejecuta el pipeline completo de preprocesamiento."""
    data_path = args.data_path
    artifacts_dir = args.artifacts_dir
    cache_dir = os.path.join(artifacts_dir, "cache")

    # 1. Cargar y limpiar datos
    df = load_and_prepare_data(data_path)

    # 2. TF-IDF + Split
    X_train, X_test, y_train, y_test, vectorizer, label_encoder = build_tfidf_and_split(df)

    # 3. Guardar artefactos (vectorizer + label_encoder)
    save_artifacts(vectorizer, label_encoder, artifacts_dir)

    # 4. Cachear las matrices para train/evaluate
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "preprocessed_data.pkl")
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'label_classes': label_encoder.classes_,
        }, f)
    print(f"\n[Main] Datos preprocesados cacheados en: {cache_path}")
    print("[Main] ¡Preprocesamiento completado con éxito!")


def cmd_train(args):
    """Entrena el modelo XGBoost usando los datos preprocesados."""
    artifacts_dir = args.artifacts_dir
    figures_dir = args.figures_dir
    cache_dir = os.path.join(artifacts_dir, "cache")
    cache_path = os.path.join(cache_dir, "preprocessed_data.pkl")

    if not os.path.isfile(cache_path):
        print("[Main] ERROR: No se encontraron datos preprocesados.")
        print("       Ejecuta primero: python -m src.main preprocess")
        sys.exit(1)

    # Cargar datos cacheados
    print(f"[Main] Cargando datos preprocesados desde: {cache_path}")
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)

    # Entrenar
    train_xgboost(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_test=data['X_test'],
        y_test=data['y_test'],
        label_classes=data['label_classes'],
        output_dir=artifacts_dir,
        figures_dir=figures_dir,
    )

    print("\n[Main] ¡Entrenamiento completado con éxito!")


def cmd_evaluate(args):
    """Evalúa el modelo entrenado sobre el conjunto de test."""
    artifacts_dir = args.artifacts_dir
    figures_dir = args.figures_dir
    cache_dir = os.path.join(artifacts_dir, "cache")
    cache_path = os.path.join(cache_dir, "preprocessed_data.pkl")

    model_path_ubj = os.path.join(artifacts_dir, "xgboost_topic_classifier.ubj")
    model_path_pkl = os.path.join(artifacts_dir, "xgboost_topic_classifier.pkl")

    if os.path.isfile(model_path_ubj):
        model_path = model_path_ubj
    elif os.path.isfile(model_path_pkl):
        model_path = model_path_pkl
    else:
        model_path = None

    if not os.path.isfile(cache_path):
        print("[Main] ERROR: No se encontraron datos preprocesados.")
        print("       Ejecuta primero: python -m src.main preprocess")
        sys.exit(1)

    if not model_path or not os.path.isfile(model_path):
        print("[Main] ERROR: No se encontró el modelo entrenado.")
        print("       Ejecuta primero: python -m src.main train")
        sys.exit(1)

    # Cargar datos y label encoder
    print(f"[Main] Cargando datos preprocesados desde: {cache_path}")
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)

    import joblib
    enc_path = os.path.join(artifacts_dir, "label_encoder.pkl")
    label_encoder = joblib.load(enc_path)

    # Evaluar
    evaluate_model(
        model_path=model_path,
        X_test=data['X_test'],
        y_test=data['y_test'],
        label_encoder=label_encoder,
        figures_dir=figures_dir,
    )

    print("\n[Main] ¡Evaluación completada con éxito!")


def cmd_predict(args):
    """Clasifica una noticia nueva usando los artefactos existentes."""
    artifacts_dir = args.artifacts_dir
    title = args.title
    content = args.content

    if not title or not content:
        print("[Main] ERROR: Debes proporcionar --title y --content para la predicción.")
        sys.exit(1)

    # Cargar pipeline de inferencia
    pipeline = load_inference_pipeline(artifacts_dir)

    # Clasificar
    print(f"\n[Predict] Procesando noticia...")
    categoria = classify_news(title, content, pipeline)

    print("\n" + "=" * 60)
    print("  RESULTADO DE LA PREDICCIÓN")
    print("=" * 60)
    print(f"  TÍTULO:     {title}")
    print(f"  CONTENIDO:  {content[:120]}...")
    print(f"  PREDICCIÓN: {categoria.upper()}")
    print("=" * 60)


# ==============================================================================
# PARSER CLI
# ==============================================================================

def build_parser():
    """Construye el parser de argumentos del CLI."""
    parser = argparse.ArgumentParser(
        description="Clasificador de Tópicos de Noticias — XGBoost Encapsulado",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Argumentos globales
    parser.add_argument(
        '--data_path',
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Ruta al CSV del dataset.\n(por defecto: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        '--artifacts_dir',
        type=str,
        default=DEFAULT_ARTIFACTS_DIR,
        help=f"Carpeta de artefactos (modelo, vectorizador, encoder).\n(por defecto: {DEFAULT_ARTIFACTS_DIR})",
    )
    parser.add_argument(
        '--figures_dir',
        type=str,
        default=DEFAULT_FIGURES_DIR,
        help=f"Carpeta de salida para figuras.\n(por defecto: {DEFAULT_FIGURES_DIR})",
    )

    # Subcomandos
    subparsers = parser.add_subparsers(dest='command', help="Comando a ejecutar")

    # preprocess
    subparsers.add_parser(
        'preprocess',
        help="Carga, limpia, vectoriza (TF-IDF) y divide el dataset.",
    )

    # train
    subparsers.add_parser(
        'train',
        help="Entrena el modelo XGBoost con los datos preprocesados.",
    )

    # evaluate
    subparsers.add_parser(
        'evaluate',
        help="Evalúa el modelo con el conjunto de test (report + matriz de confusión).",
    )

    # predict
    predict_parser = subparsers.add_parser(
        'predict',
        help="Clasifica una noticia nueva (requiere --title y --content).",
    )
    predict_parser.add_argument(
        '--title',
        type=str,
        required=True,
        help="Título de la noticia a clasificar.",
    )
    predict_parser.add_argument(
        '--content',
        type=str,
        required=True,
        help="Contenido/cuerpo de la noticia a clasificar.",
    )

    return parser


# ==============================================================================
# PUNTO DE ENTRADA
# ==============================================================================

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        'preprocess': cmd_preprocess,
        'train': cmd_train,
        'evaluate': cmd_evaluate,
        'predict': cmd_predict,
    }

    commands[args.command](args)


if __name__ == '__main__':
    main()
