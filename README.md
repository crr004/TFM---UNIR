# 🎓 Trabajo de Fin de Máster (TFM): Detección y Clasificación de Noticias (Fake News & Topics)

## 🎯 1. Objetivo del Proyecto

El objetivo principal de este Trabajo de Fin de Máster (TFM) es diseñar, desarrollar y evaluar de forma exhaustiva sistemas de Inteligencia Artificial aplicados al Procesamiento de Lenguaje Natural (NLP). El sistema está diseñado para analizar textos periodísticos y noticias para abordar dos retos fundamentales en la era de la información:

1. **Clasificación Binaria (Detección de Fake News):** Diseñar y entrenar modelos predictivos avanzados, abarcando desde Machine Learning tradicional hasta Deep Learning y arquitecturas basadas en Transformers (BETO), capaces de discriminar con alta precisión si una noticia es verídica (`True`) o falsa (`False`). Este análisis se basa en características léxicas, semánticas y estructurales del texto.
2. **Clasificación Multiclase (Topic Modeling):** Desarrollar un sistema de Machine Learning robusto, escalable y orientado a producción para categorizar automáticamente artículos de noticias dentro de diversas temáticas predefinidas (política, economía, deportes, tecnología, etc.).

El proyecto cubre integralmente el ciclo de vida de los datos:
- **Ingesta y Preparación:** Unificación de fuentes dispares y normalización.
- **Preprocesamiento Avanzado de NLP:** Limpieza de texto, tokenización, lematización usando `spaCy` y adaptación del formato a diferentes arquitecturas de modelado.
- **Análisis Exploratorio de Datos (EDA):** Detección de patrones, análisis de distribuciones y balanceo de clases.
- **Modelado:** Entrenamiento de una amplia batería de algoritmos:
  - Modelos Clásicos: Regresión Logística, SVM, Random Forest, XGBoost.
  - Modelos de Deep Learning: CNN 1D, LSTM, BiLSTM, BiGRU.
  - Transformers: BETO (BERT en español).
- **Explicabilidad (XAI):** Uso de técnicas para interpretar las decisiones de los modelos y garantizar la transparencia y confiabilidad del sistema.

---

## 📂 2. Estructura de Carpetas

El repositorio sigue un patrón de diseño modular, separando las fases de tratamiento de datos, análisis exploratorio y modelado (clásico, profundo y de producción) para mantener un código limpio y reproducible.

```text
📦 TFM---UNIR
 ┣ 📂 BETO/                                 # 🤖 Implementación del Transformer BETO para Detección de Fake News
 ┣ 📂 Data/                                 # 📥 Ingesta, preparación, limpieza y preprocesado avanzado (NLP)
 ┣ 📂 EDA/                                  # 📊 Análisis Exploratorio de Datos (Fake News y Temáticas)
 ┣ 📂 MODELOS CLASICOS/                     # ⚙️ Modelos de Machine Learning Tradicional (Fake News)
 ┣ 📂 MODELOS DEEP LEARNING/                # 🧠 Modelos de Redes Neuronales Profundas (Fake News)
 ┣ 📂 Modelo_Topics/                        # 🚀 Modelo en Producción de Clasificación Multiclase (Temáticas)
 ┣ 📜 .gitignore                            # 🚫 Reglas de exclusión para control de versiones
 ┗ 📜 README.md                             # 📖 Este archivo de documentación técnica
```

---

## 🔍 3. Información Detallada del Proyecto

A continuación, se detalla rigurosamente el contenido y la finalidad de cada uno de los módulos que componen este repositorio.

### 📥 `Data/` (Preparación y Preprocesamiento)
Este módulo es el núcleo de la ingeniería de datos. Gestiona la extracción, transformación y carga (ETL), así como el procesamiento de lenguaje natural profundo.
* **`Bin/` y `Topic/`**: Directorios destinados a almacenar los conjuntos de datos en bruto (CSVs, Parquets) provenientes de múltiples fuentes, separados por su problema objetivo (binario o multiclase).
* **`prepare_data_bin.py` / `prepare_data_topic.py`**: Scripts de unificación. Su función es extraer, normalizar nombres de columnas, resolver conflictos y consolidar múltiples datasets en un único archivo limpio por cada problema (`Data_Bin_Classifier.csv` y `Dataset_Topic_Classifier.csv`).
* **`preprocesado.py`**: El motor principal de NLP del proyecto. Aplica técnicas avanzadas de normalización Unicode, tokenización, eliminación de stopwords y lematización (mediante el modelo `es_core_news_lg` de `spaCy`). Destaca por generar tres representaciones distintas del mismo texto optimizadas para diferentes arquitecturas:
  * `text_ml`: Limpieza severa orientada a modelos basados en frecuencias y TF-IDF.
  * `text_dl`: Mantiene estructura y conectores para redes recurrentes y convolucionales.
  * `text_beto`: Limpieza mínima que conserva la integridad gramatical y semántica para modelos Transformer.
  * Adicionalmente, extrae *features* estructurales clave (cantidad de URLs, mayúsculas, exclamaciones).
* **`dataset_preprocesado_binario.csv`**: El artefacto final listo para ser consumido en las fases de entrenamiento.

### 📊 `EDA/` (Análisis Exploratorio de Datos)
Carpeta dedicada a la investigación estadística y visual del comportamiento de los datos.
* **`Eda_Dataset_Binario.ipynb`**: Estudio exhaustivo para detectar patrones textuales que diferencian noticias verdaderas de falsas (análisis de n-gramas, longitud de texto, uso de signos de puntuación).
* **`Eda_Dataset_Topic.ipynb`**: Análisis de las distribuciones temáticas, evaluación de desbalanceo de clases y estrategias de consolidación y agrupación de categorías para optimizar el rendimiento de la clasificación.

### ⚙️ `MODELOS CLASICOS/` (Machine Learning Tradicional para Fake News)
Implementación y evaluación de algoritmos clásicos de Machine Learning que establecen el *baseline* o línea base de rendimiento en la clasificación binaria.
* **`regresion_logistica.py`, `svm.py`, `random_forest.py`, `XGBoost.py`**: Scripts independientes de entrenamiento y evaluación. Utilizan vectorización TF-IDF sobre la columna `text_ml`.
* **`graficas/` y `xai/`**: Subdirectorios que almacenan las matrices de confusión, curvas ROC, precision-recall y los análisis de explicabilidad generados, vitales para evaluar la interpretabilidad de cada algoritmo.
* **Resultados JSON (`*_results.json`)**: Archivos que persisten las métricas de evaluación (Accuracy, Precision, Recall, F1-Score) de cada modelo para facilitar su comparación sistemática.

### 🧠 `MODELOS DEEP LEARNING/` (Redes Neuronales para Fake News)
Agrupa las implementaciones de Deep Learning, explotando la secuencia del texto (`text_dl`) mediante arquitecturas avanzadas desarrolladas en TensorFlow/Keras.
* **Subcarpetas (`Modelo_CNN_Bin/`, `Modelo_LSTM_Bin/`, `Modelo_BiLSTM_Bin/`, `Modelo BiGRU_Bin/`)**:
  * **CNN 1D**: Extrae características locales (n-gramas de texto).
  * **LSTM / BiLSTM**: Capta dependencias temporales y contexto secuencial (unidireccional y bidireccional).
  * **BiGRU**: Alternativa eficiente a LSTM con rendimiento computacional optimizado.
* En su interior, cada modelo contiene su respectivo Notebook de entrenamiento, y subcarpetas para persistir sus `artefactos/` (pesos del modelo en `.h5`/`.keras`, tokenizadores), así como generar sus propias métricas y explicabilidad local (XAI).

### 🤖 `BETO/` (Modelos Transformer)
Módulo dedicado a la implementación de BETO (Bidirectional Encoder Representations from Transformers en español) para alcanzar el estado del arte en la detección de Fake News.
* **`src/`**: Código modular con el pipeline de fine-tuning usando la librería Transformers de Hugging Face.
* **Scripts de Verificación (`check_leakage.py`, `check_nvidia.py`)**: Herramientas críticas para asegurar la calidad de la partición de datos (evitar *data leakage* entre conjuntos de entrenamiento y prueba) y verificar la correcta configuración y disponibilidad de recursos GPU (CUDA) para la aceleración del entrenamiento.

### 🚀 `Modelo_Topics/` (Clasificación Multiclase en Producción)
A diferencia de los enfoques experimentales previos, este directorio encapsula la solución para la clasificación de temáticas estructurada bajo un paradigma estricto de *Software Engineering*, lista para ser desplegada en producción o ser servida mediante APIs.
* **`src/`**: Carpeta fuente modular:
  * `main.py`: Interfaz de Línea de Comandos (CLI) para orquestar todos los procesos: limpieza, entrenamiento, predicción y evaluación.
  * `preprocessing.py`: Módulo autónomo de NLP que estandariza los inputs crudos mediante `spaCy`.
  * `train.py` & `predict.py`: Lógica de entrenamiento y flujo de inferencia. Utiliza **XGBoost** como modelo *core* debido a su robustez, balance entre explicabilidad y altísimo rendimiento en la predicción sobre la matriz de características.
* **`artefactos/`**: Directorio donde residen los modelos preentrenados (`.pkl`), vectorizadores (`TfidfVectorizer`) y codificadores de etiquetas (`LabelEncoder`), todos debidamente serializados.
* **`Modelo_XGBoost_Topic.ipynb`**: Registro histórico de la experimentación visual y la búsqueda de hiperparámetros óptimos que derivaron en el modelo de producción final.

---

Este Trabajo de Fin de Máster representa una solución End-to-End que demuestra no solo la aplicación de tecnologías punteras en Inteligencia Artificial (desde árboles de decisión hasta Transformers), sino también el dominio de las mejores prácticas en ingeniería de datos, arquitectura de software orientado a Machine Learning (MLOps) y el rigor analítico fundamental para garantizar la explicabilidad y confianza en los modelos desarrollados.
