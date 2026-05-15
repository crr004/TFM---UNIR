# Clasificador de Tópicos de Noticias — XGBoost

Este módulo contiene un pipeline encapsulado y reproducible para **clasificar automáticamente noticias en español** en una de 9 categorías temáticas, utilizando un modelo **XGBoost** entrenado con representación **TF-IDF**.

El sistema está diseñado para que, una vez entrenado, la inferencia sea rápida y sencilla sin necesidad de re-ejecutar el entrenamiento.

---

## Categorías de Clasificación

El modelo agrupa los tópicos originales del dataset [MLSUM](https://huggingface.co/datasets/reciTAL/mlsum) (212 etiquetas) en **9 categorías principales**:

| # | Categoría |
|---|-----------|
| 0 | Ciencia y Tecnología |
| 1 | Cultura y Entretenimiento |
| 2 | Deportes |
| 3 | Economía |
| 4 | España / Local |
| 5 | Internacional |
| 6 | Otros |
| 7 | Política |
| 8 | Sociedad y Estilo de Vida |

---

## Estructura de carpetas

```text
Modelo_Clasificador_Topics/
├── src/                                 # Código fuente encapsulado
│   ├── __init__.py                      # Inicializador del paquete
│   ├── main.py                          # Orquestador CLI (punto de entrada)
│   ├── preprocessing.py                 # Carga, limpieza, TF-IDF y split
│   ├── train.py                         # Entrenamiento del modelo XGBoost
│   ├── evaluate.py                      # Evaluación (report + matriz de confusión)
│   └── predict.py                       # Pipeline de inferencia end-to-end
├── artefactos/                          # Artefactos serializados del modelo
│   ├── xgboost_topic_classifier.ubj     # Modelo XGBoost entrenado
│   ├── tfidf_vectorizer.pkl             # Vectorizador TF-IDF ajustado
│   ├── label_encoder.pkl                # Codificador de etiquetas
│   └── cache/                           # Datos preprocesados cacheados
│       └── preprocessed_data.pkl        # Matrices TF-IDF + labels (train/test)
├── figuras/                             # Visualizaciones generadas
│   ├── curvas_aprendizaje_xgboost.png   # Curvas de aprendizaje (log-loss)
│   └── matriz_confusion_topic_classifier.png  # Matriz de confusión
├── Modelo_XGBoost_Topic.ipynb           # Notebook original de desarrollo
└── README.md                            # Este archivo
```

---

## Descripción de cada archivo

### `src/preprocessing.py`
**Qué hace:** Se encarga de toda la preparación de los datos antes de entrenar el modelo.

1. **Carga** el dataset CSV (`Dataset_Topic_Classifier.csv`).
2. **Agrupa** los 212 tópicos originales en 9 categorías usando la función `map_topic()`.
3. **Limpia** el texto con expresiones regulares (elimina URLs, HTML, caracteres especiales).
4. **Procesa** el texto con spaCy (`es_core_news_sm`): lematización y eliminación de stopwords.
5. **Genera** la representación numérica TF-IDF (máximo 30.000 features, unigramas + bigramas).
6. **Divide** el dataset en 80% entrenamiento / 20% test (estratificado).
7. **Guarda** el vectorizador TF-IDF y el LabelEncoder como artefactos `.pkl`.

**Artefactos que genera:**
- `artefactos/tfidf_vectorizer.pkl`
- `artefactos/label_encoder.pkl`
- `artefactos/cache/preprocessed_data.pkl` (datos cacheados para train/evaluate)

---

### `src/train.py`
**Qué hace:** Entrena el modelo XGBoost multiclase.

1. **Detecta automáticamente** si hay GPU CUDA disponible. Si no la hay, usa CPU (fallback automático).
2. **Configura** el clasificador XGBoost con los hiperparámetros óptimos:
   - 30.000 estimadores máximos con early stopping (20 rondas sin mejora).
   - `tree_method='hist'` para entrenamiento rápido.
   - `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`.
3. **Entrena** el modelo monitorizando la log-loss en train y test.
4. **Guarda** el modelo entrenado en formato UBJ (Universal Binary JSON de XGBoost).
5. **Genera** la gráfica de curvas de aprendizaje.

**Artefactos que genera:**
- `artefactos/xgboost_topic_classifier.ubj`
- `figuras/curvas_aprendizaje_xgboost.png`

---

### `src/evaluate.py`
**Qué hace:** Evalúa el rendimiento del modelo sobre el conjunto de test.

1. **Carga** el modelo entrenado desde disco.
2. **Genera predicciones** sobre el conjunto de test.
3. **Imprime** por consola el **Classification Report** completo (precision, recall, F1-score por clase).
4. **Genera y guarda** una **Matriz de Confusión** profesional como heatmap PNG (DPI 300).

**Artefactos que genera:**
- `figuras/matriz_confusion_topic_classifier.png`

**Output por consola:** Classification Report con métricas por clase y globales.

---

### `src/predict.py`
**Qué hace:** Pipeline de inferencia para clasificar noticias nuevas.

1. **Carga** los 3 artefactos: modelo XGBoost, vectorizador TF-IDF y LabelEncoder.
2. **Carga** el modelo spaCy para procesamiento lingüístico.
3. **Clasifica** una noticia en crudo (título + contenido) siguiendo el pipeline completo:
   - Unir título + contenido → limpieza regex → lematización spaCy → vectorización TF-IDF → predicción → decodificación.

> **Nota:** Este módulo es completamente **independiente del dataset original y del entrenamiento**. Solo necesita los artefactos serializados y spaCy instalado.

---

### `src/main.py`
**Qué hace:** Orquestador CLI que conecta todos los módulos anteriores.

Expone 4 comandos ejecutables desde terminal mediante `argparse`:

| Comando | Descripción |
|---------|-------------|
| `preprocess` | Ejecuta la carga, limpieza, TF-IDF y split. Guarda artefactos. |
| `train` | Entrena el modelo XGBoost con los datos preprocesados. |
| `evaluate` | Evalúa el modelo y genera el reporte + matriz de confusión. |
| `predict` | Clasifica una noticia nueva a partir de su título y contenido. |

---

## Guía de Ejecución Paso a Paso

> **Requisito previo:** Situarse en la carpeta `Modelo_Clasificador_Topics/` antes de ejecutar los comandos.

### Paso 1 — Preprocesamiento

Carga el dataset, limpia los textos, genera el TF-IDF y divide en train/test:

```bash
python -m src.main preprocess
```

**Opcionalmente**, puedes especificar una ruta diferente al dataset:

```bash
python -m src.main preprocess --data_path "../Data/Topic/Dataset_Topic_Classifier.csv"
```

Este paso genera:
- `artefactos/tfidf_vectorizer.pkl`
- `artefactos/label_encoder.pkl`
- `artefactos/cache/preprocessed_data.pkl`

---

### Paso 2 — Entrenamiento

Entrena el modelo XGBoost usando los datos preprocesados del paso anterior:

```bash
python -m src.main train
```

> Si dispones de GPU CUDA, el entrenamiento se acelerará automáticamente. Si no, se usará CPU sin intervención manual.

Este paso genera:
- `artefactos/xgboost_topic_classifier.ubj`
- `figuras/curvas_aprendizaje_xgboost.png`

---

### Paso 3 — Evaluación

Evalúa el modelo sobre el conjunto de test reservado:

```bash
python -m src.main evaluate
```

Este paso genera:
- `figuras/matriz_confusion_topic_classifier.png`
- Classification Report impreso por consola.

---

### Paso 4 — Inferencia (Predicción)

Clasifica una noticia nueva proporcionando su título y contenido:

```bash
python -m src.main predict --title "El Real Madrid gana la Champions" --content "El equipo blanco venció al rival en la final de la Champions League celebrada en París."
```

**Ejemplo de salida:**
```
============================================================
  RESULTADO DE LA PREDICCIÓN
============================================================
  TÍTULO:     El Real Madrid gana la Champions
  CONTENIDO:  El equipo blanco venció al rival en la final de la Champions League celebrada ...
  PREDICCIÓN: DEPORTES
============================================================
```

> **Importante:** Este comando solo necesita los artefactos en `artefactos/` y spaCy instalado. No requiere el dataset original ni haber ejecutado el preprocesamiento.

---

## Dependencias

Las librerías necesarias para ejecutar este paquete son:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `spacy` (+ modelo `es_core_news_sm`)
- `matplotlib`
- `seaborn`
- `joblib`
- `tqdm`

Para instalar el modelo de spaCy:
```bash
python -m spacy download es_core_news_sm
```

> Si el modelo de spaCy no está instalado, el sistema lo descargará automáticamente la primera vez que lo necesite.
