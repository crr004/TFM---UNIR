import argparse
from pathlib import Path

from .evaluate import evaluate_model
from .explain import run_explanations
from .train import train_beto


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fine-tuning BETO + Explainability (LIME, SHAP, Integrated Gradients)",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="../Data/Bin/Data_Bin_Classifier.csv",
        help="Ruta al dataset binario CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./artifacts",
        help="Carpeta de artefactos de entrenamiento/explicabilidad",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./artifacts/model",
        help="Carpeta del modelo entrenado",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Fine-tuning BETO")
    subparsers.add_parser("evaluate", help="Evaluación en test")

    explain_parser = subparsers.add_parser("explain", help="Generar explicaciones")
    explain_parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Texto libre para explicar (si no se pasa, usa muestras del test split)",
    )
    explain_parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="Número de muestras del test para explicar",
    )
    explain_parser.add_argument(
        "--faithfulness_scope",
        type=str,
        default="sample",
        choices=["sample", "full_test"],
        help="sample: usa n_samples. full_test: agrega métricas sobre todo test (o un muestreo).",
    )
    explain_parser.add_argument(
        "--faithfulness_max_samples",
        type=int,
        default=None,
        help="Límite opcional de muestras cuando faithfulness_scope=full_test.",
    )
    explain_parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Semilla para muestreo reproducible en full_test.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)

    if args.command == "train":
        train_beto(dataset_path=dataset_path, output_dir=output_dir)
    elif args.command == "evaluate":
        evaluate_model(
            model_dir=model_dir, dataset_path=dataset_path, output_dir=output_dir
        )
    elif args.command == "explain":
        run_explanations(
            model_dir=model_dir,
            output_dir=output_dir,
            dataset_path=dataset_path,
            text=args.text,
            n_samples=args.n_samples,
            faithfulness_scope=args.faithfulness_scope,
            faithfulness_max_samples=args.faithfulness_max_samples,
            random_state=args.random_state,
        )


if __name__ == "__main__":
    main()
