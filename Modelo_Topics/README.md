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

## 🏗️ Estructura del Proyecto
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
│   ├── xgboost_topic_classifier.pkl     # Modelo XGBoost entrenado
│   ├── tfidf_vectorizer.pkl             # Vectorizador TF-IDF ajustado
│   └── label_encoder.pkl                # Codificador de etiquetas

├── figuras/                             # Visualizaciones generadas
│   ├── curvas_aprendizaje_xgboost.png   # Curvas de aprendizaje (log-loss)
│   └── matriz_confusion_topic_classifier.png  # Matriz de confusión
├── Modelo_XGBoost_Topic.ipynb           # Notebook original de desarrollo
└── README.md                            # Este archivo
```

---

## Descripción de cada archivo

### `src/preprocessing.py`
**Qué hace:** Se encarga de toda la preparación de los datos antes de entrenar el modelo mediante una estrategia de **limpieza progresiva**.

1. **Carga** el dataset CSV (`Dataset_Topic_Classifier.csv`).
2. **Agrupa** los 212 tópicos originales en 9 categorías y limpia estructuralmente el dataset (elimina nulos, duplicados y textos muy cortos).
3. **Filtra el idioma** utilizando `langdetect` para descartar los textos cuya lengua dominante no sea el español.
4. **Limpia el texto con Regex:** Normalización Unicode y eliminación de URLs, emails, HTML, repeticiones de caracteres y secuencias numéricas.
5. **Procesa NLP con spaCy (`es_core_news_lg`):**
   - **Reemplazo NER:** Sustituye entidades nombradas por *placeholders* genéricos (`__PERSONA__`, `__ORGANIZACION__`, `__LUGAR__`) para generalizar el vocabulario.
   - **Lematización** y eliminación de *stopwords* (manteniendo una lista blanca semántica: `no`, `sí`, `ni`).
6. **Genera** la representación numérica TF-IDF (máximo 30.000 features, unigramas + bigramas).
7. **Divide** el dataset en 80% entrenamiento / 20% test (estratificado).
8. **Guarda** el vectorizador TF-IDF, el LabelEncoder y las matrices en caché.

**Artefactos que genera:**
- `artefactos/tfidf_vectorizer.pkl`
- `artefactos/label_encoder.pkl`
- `artefactos/cache/preprocessed_data.pkl` (datos cacheados para train/evaluate)

---

### `src/train.py`
**Qué hace:** Entrena el modelo XGBoost multiclase.

1. **Detecta automáticamente** si hay GPU CUDA disponible (`device='cuda'`). Si no la hay, hace un *fallback* a CPU.
2. **Configura** el clasificador XGBoost con los hiperparámetros óptimos:
   - 30.000 estimadores máximos con *early stopping* (20 rondas sin mejora).
   - `tree_method='hist'` para aceleración del entrenamiento.
   - `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`.
3. **Entrena** el modelo monitorizando la métrica `mlogloss` en train y test.
4. **Guarda** el modelo entrenado en formato pkl.
5. **Genera** la gráfica de las curvas de aprendizaje.

**Artefactos que genera:**
- `artefactos/xgboost_topic_classifier.pkl`
- `figuras/curvas_aprendizaje_xgboost.png`

---

### `src/evaluate.py`
**Qué hace:** Evalúa el rendimiento del modelo sobre el conjunto de test.

1. **Carga** el modelo entrenado desde disco.
2. **Genera predicciones** sobre el conjunto de test cacheado.
3. **Imprime** por consola el **Classification Report** completo (precision, recall, F1-score por clase).
4. **Genera y guarda** una **Matriz de Confusión** profesional como *heatmap* PNG (DPI 300).

**Artefactos que genera:**
- `figuras/matriz_confusion_topic_classifier.png`

**Output por consola:** Classification Report con métricas por clase y globales.

---

### `src/predict.py`
**Qué hace:** Pipeline de inferencia estandarizado para clasificar noticias nuevas.

1. **Carga** los 3 artefactos: modelo XGBoost, vectorizador TF-IDF y LabelEncoder.
2. **Carga** el modelo spaCy para el procesamiento lingüístico.
3. **Clasifica** una noticia en crudo (título + contenido) aplicando **exactamente** las mismas fases de limpieza que en el entrenamiento (limpieza Regex → spaCy con reemplazo NER y lematización → TF-IDF → predicción → decodificación).

> **Nota:** Este módulo es completamente **independiente del dataset original y del entrenamiento**. Solo necesita los artefactos serializados y spaCy instalado.

---

### `src/main.py`
**Qué hace:** Orquestador CLI que conecta todos los módulos anteriores.

Expone 4 comandos ejecutables desde terminal mediante `argparse`:

| Comando | Descripción |
|---------|-------------|
| `preprocess` | Ejecuta la carga, limpieza, TF-IDF, split y cacheo. Guarda artefactos. |
| `train` | Entrena el modelo XGBoost con los datos preprocesados. |
| `evaluate` | Evalúa el modelo y genera el reporte + matriz de confusión. |
| `predict` | Clasifica una noticia nueva a partir de su título y contenido. |

---

## Guía de Ejecución Paso a Paso

> **Requisito previo:** Situarse en la raíz del proyecto (`Modelo_Clasificador_Topics/`) antes de ejecutar los comandos.

### Paso 1 — Preprocesamiento

Carga el dataset, limpia los textos, genera el TF-IDF y divide en train/test:
```bash
python -m src.main preprocess
```

**Opcionalmente**, puedes especificar una ruta diferente al dataset:

```bash
python -m src.main preprocess --data_path "../Data/Topic/Dataset_Topic_Classifier.csv"
```

---

### Paso 2 — Entrenamiento

Entrena el modelo XGBoost usando los datos cacheados del paso anterior:
```bash
python -m src.main train
```

> Si dispones de GPU CUDA, el entrenamiento se acelerará automáticamente.

---

### Paso 3 — Evaluación

Evalúa el modelo sobre el conjunto de test reservado:
```bash
python -m src.main evaluate
```

---

### Paso 4 — Inferencia (Predicción)

Clasifica una noticia nueva proporcionando su título y contenido:

```bash
python -m src.main predict --title "El Real Madrid gana la Champions" --content "El equipo blanco venció al rival en la final de la Champions League celebrada en París."
```

**Ejemplo de salida:**
```text
============================================================
  RESULTADO DE LA PREDICCIÓN
============================================================
  TÍTULO:     El Real Madrid gana la Champions
  CONTENIDO:  El equipo blanco venció al rival en la final de la Champions League celebrada ...
  PREDICCIÓN: DEPORTES
============================================================
```

---

## Dependencias

Las librerías necesarias para ejecutar este paquete son:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `spacy` (+ modelo `es_core_news_lg`)
- `langdetect`
- `matplotlib`
- `seaborn`
- `joblib`
- `tqdm`

### Cómo instalar las dependencias

1. Abre tu terminal, asegúrate de estar en el directorio de tu proyecto y ejecuta este comando para instalar las librerías:

```bash
pip install -r requirements.txt
```

2. Una vez finalizada la instalación de las librerías, descarga el modelo de idioma específico de spaCy ejecutando:

```bash
python -m spacy download es_core_news_lg
```

