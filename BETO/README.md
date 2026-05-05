# BETO Fine-Tuning + Explainability

Pipeline para fine-tunear BETO (clasificación binaria fake/real) y generar explicaciones con:

- LIME
- SHAP
- Integrated Gradients (Captum)

## 1) Objetivo

Entrenar BETO sobre el dataset binario y disponer de explicaciones locales por predicción para auditoría del modelo.

## 2) Dataset esperado

Por defecto usa:

- ../Data/Bin/Data_Bin_Classifier.csv

Columnas compatibles:

- Texto: text_beto, full_text o Content (en ese orden de prioridad)
- Etiqueta: label (0/1) o State (fake/real, false/true, 1/0)

## 3) Instalación

Desde la carpeta BETO:

```bash
pip install -r requirements.txt
```

## 4) Entrenamiento

```bash
python -m src.main train --dataset_path ../Data/Bin/Data_Bin_Classifier.csv --output_dir ./artifacts
```

Salida principal:

- artifacts/model
- artifacts/train_split.csv
- artifacts/val_split.csv
- artifacts/test_split.csv
- artifacts/metrics/val_metrics.json

## 5) Evaluación

```bash
python -m src.main evaluate --dataset_path ../Data/Bin/Data_Bin_Classifier.csv --model_dir ./artifacts/model --output_dir ./artifacts
```

Salida:

- artifacts/metrics/test_metrics.json
- artifacts/metrics/test_predictions.csv

## 6) Explicabilidad

### Explicar texto libre

```bash
python -m src.main explain --model_dir ./artifacts/model --output_dir ./artifacts --text "El Gobierno confirma nuevas ayudas económicas para 2026"
```

### Explicar muestras del dataset de test

```bash
python -m src.main explain --dataset_path ../Data/Bin/Data_Bin_Classifier.csv --model_dir ./artifacts/model --output_dir ./artifacts --n_samples 3
```

### Resumen global de faithfulness sobre test

```bash
python -m src.main explain --dataset_path ../Data/Bin/Data_Bin_Classifier.csv --model_dir ./artifacts/model --output_dir ./artifacts --faithfulness_scope full_test
```

Para limitar coste computacional en test grande:

```bash
python -m src.main explain --dataset_path ../Data/Bin/Data_Bin_Classifier.csv --model_dir ./artifacts/model --output_dir ./artifacts --faithfulness_scope full_test --faithfulness_max_samples 200 --random_state 42
```

Salida:

- artifacts/explanations/lime_sample_X.html
- artifacts/explanations/ig_sample_X.json
- artifacts/explanations/faithfulness_sample_X.json
- artifacts/explanations/shap_summary.json
- artifacts/explanations/explanations_index.csv
- artifacts/explanations/faithfulness_global_summary.json

## 7) Qué aporta cada técnica

- LIME: explicación local interpretable por palabras (perturbación local).
- SHAP: contribución de tokens a la clase predicha (aprox. aditiva).
- Integrated Gradients: atribución basada en gradiente integrado sobre embeddings.
- Faithfulness tests:
	- Comprehensiveness: cuánto cae la probabilidad al quitar palabras top.
	- Sufficiency: cuánto se conserva la predicción usando solo palabras top.
	- Deletion/Insertion curves: sensibilidad del modelo al retirar/incluir evidencia.

## 8) Lectura rápida de faithfulness

- Comprehensiveness alto (delta medio alto): las palabras destacadas son realmente importantes para la predicción.
- Sufficiency cercano a 0: las palabras top por sí solas conservan casi toda la evidencia.
- Deletion AUC bajo: al eliminar top words, la confianza cae rápido (buena señal de faithfulness).
- Insertion AUC alto: al añadir top words, la confianza sube rápido (buena señal de faithfulness).

## 9) Recomendaciones para estado del arte

Para avanzar hacia un stack más SOTA en explicabilidad de LLMs/Transformers, la siguiente fase recomendable es:

- Attention rollout y análisis de capas/cabezas.
- Faithfulness tests (deletion/insertion curves, comprehensiveness/sufficiency).
- Rationale supervision (si consigues anotaciones humanas).
- Contrafactuales automáticos y robustez semántica.
- Calibración de incertidumbre (ECE, temperature scaling) antes de interpretar.

## 10) Nota

El pipeline está preparado como base reproducible y extensible para tu TFM: puedes entrenar, evaluar y generar explicaciones sin notebooks.
