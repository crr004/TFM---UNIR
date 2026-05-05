from pathlib import Path
import re

import pandas as pd

DATASET_PATH = Path("./Data/dataset_preprocesado_binario.csv")
TEXT_COLUMN = "text_beto"  # cambia esto si tu columna de texto tiene otro nombre


def main():
    df = pd.read_csv(DATASET_PATH)

    print("Columnas del dataset:")
    print(df.columns.tolist())
    print()

    if TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"No existe la columna {TEXT_COLUMN!r}. Columnas disponibles: {df.columns.tolist()}"
        )

    text = df[TEXT_COLUMN].fillna("").astype(str)

    patterns = {
        "Prediction": r"Prediction\s*:",
        "Prediction True": r"Prediction\s*:\s*True",
        "Prediction False": r"Prediction\s*:\s*False",
        "Source": r"Source\s*:",
        "Content": r"Content\s*:",
        "Label": r"Label\s*:",
    }

    print("Conteo de patrones sospechosos:")
    for name, pattern in patterns.items():
        count = text.str.contains(pattern, case=False, regex=True).sum()
        print(f"{name:20s}: {count}")

    print()

    mask_prediction = text.str.contains(r"Prediction\s*:", case=False, regex=True)
    print(f"Total de filas con 'Prediction:': {mask_prediction.sum()}")

    if mask_prediction.sum() > 0:
        print("\nEjemplos de textos con 'Prediction:':")
        examples = df.loc[mask_prediction, [TEXT_COLUMN]].head(10)

        for i, row in examples.iterrows():
            snippet = row[TEXT_COLUMN]
            snippet = re.sub(r"\s+", " ", str(snippet)).strip()
            print(f"\n--- Ejemplo índice {i} ---")
            print(snippet[:800])

    label_col = None
    for candidate in ["label", "State", "state"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is not None:
        tmp = df.copy()
        tmp["_metadata_prediction"] = "none"
        tmp.loc[
            text.str.contains(r"Prediction\s*:\s*True", case=False, regex=True),
            "_metadata_prediction",
        ] = "Prediction True"
        tmp.loc[
            text.str.contains(r"Prediction\s*:\s*False", case=False, regex=True),
            "_metadata_prediction",
        ] = "Prediction False"

        print(f"\nCruce entre metadato Prediction y columna {label_col!r}:")
        print(pd.crosstab(tmp["_metadata_prediction"], tmp[label_col], margins=True))
    else:
        print("\nNo encontré columna label/State para hacer tabla cruzada.")


if __name__ == "__main__":
    main()
