# 🎓 Proyecto TFM: Detección y Clasificación de Noticias (Fake News & Topics)

Este repositorio alberga el código fuente, los conjuntos de datos, los análisis exploratorios y los modelos desarrollados para mi **Trabajo de Fin de Máster (TFM)**. 

El proyecto se divide en dos grandes objetivos de Inteligencia Artificial aplicados al Procesamiento de Lenguaje Natural (NLP):
1. **Clasificación Binaria (Fake News)**: Identificar si una noticia es verídica (`True`) o falsa (`False`).
2. **Clasificación Multiclase (Topic Modeling)**: Categorizar una noticia en una de las 9 temáticas principales (Política, Economía, Deportes, etc.).

---

## 📂 Esquema del Proyecto

A continuación se detalla la estructura completa del repositorio y la función de cada directorio y archivo clave:

```text
📦 TFM
 ┣ 📂 Data/                                 # 📥 INGESTA Y PREPARACIÓN DE DATOS
 ┃ ┣ 📂 Bin/                                # Datasets originales (crudos) para Fake News
 ┃ ┣ 📂 Topic/                              # Datasets originales (crudos) para Topics
 ┃ ┣ 📜 prepare_data_bin.py                 # Script: Unificación inicial (Binario)
 ┃ ┣ 📜 prepare_data_topic.py               # Script: Unificación y limpieza (Topics)
 ┃ ┣ 📜 preprocesado.py                     # Script: Limpieza profunda y NLP para modelos
 ┃ ┣ 📜 dataset_preprocesado_binario.csv    # Dataset final tras el NLP (features ML, DL, BETO)
 ┃ ┣ 📜 README_preprocesado.md              # Documentación técnica del pipeline de limpieza
 ┃ ┗ 📜 README.md                           # Documentación de los orígenes de datos
 ┃
 ┣ 📂 EDA/                                  # 📊 ANÁLISIS EXPLORATORIO DE DATOS (EDA)
 ┃ ┣ 📓 Eda_Dataset_Binario.ipynb           # Análisis de noticias falsas vs reales
 ┃ ┗ 📓 Eda_Dataset_Topic.ipynb             # Análisis lingüístico y de temáticas
 ┃
 ┣ 📂 MODELOS DEEP LEARNING/                # 🧠 MODELOS DE DEEP LEARNING (FAKE NEWS)
 ┃ ┣ 📂 Modelo_CNN_Bin/                     # CNN
 ┃ ┃ ┣ 📂 artefactos/, metricas/, xai/
 ┃ ┃ ┣ 📓 Modelo_CNN_Bin.ipynb
 ┃ ┃ ┗ 📓 Modelo_CNN_SHAP_LLM copy.ipynb    # CNN + SHAP + IA Generativa
 ┃ ┣ 📂 Modelo_LSTM_Bin/                    # LSTM
 ┃ ┃ ┣ 📂 artefactos/, metricas/, xai/
 ┃ ┃ ┗ 📓 Modelo_LSTM.ipynb
 ┃ ┣ 📂 Modelo_BiLSTM_Bin/                  # BiLSTM
 ┃ ┃ ┣ 📂 artefactos/, metricas/, xai/
 ┃ ┃ ┗ 📓 Modelo_BiLSTM.ipynb
 ┃ ┗ 📂 Modelo BiGRU_Bin/                   # BiGRU
 ┃   ┣ 📂 artefactos/, metricas/, xai/
 ┃   ┗ 📓 Modelo_BiGRU.ipynb
 ┃
 ┣ 📂 Modelo_Topics/                        # 🧠 MODELO DE PRODUCCIÓN (TOPICS)
 ┃ ┣ 📂 src/                                # Código fuente encapsulado y modularizado
 ┃ ┃ ┣ 📜 main.py, train.py, predict.py...  # Orquestador CLI y pipelines
 ┃ ┣ 📂 artefactos/, figuras/               # Modelos, codificadores y gráficos
 ┃ ┣ 📓 Modelo_XGBoost_Topic.ipynb          # Notebook original de desarrollo del modelo
 ┃ ┗ 📜 README.md                           # Guía detallada de uso y ejecución del modelo
 ┃
 ┣ 📜 requirements.txt                      # Dependencias globales del proyecto
 ┣ 📜 .gitignore                            # Archivos y carpetas ignorados por Git
 ┗ 📜 README.md                             # Este archivo (Guía General)
```

---

## 🧩 Descripción de los Módulos Principales

### 1. Preparación de Datos Avanzada (`/Data`)
El pipeline de datos cuenta con extracciones desde múltiples fuentes, pero recientemente se ha añadido un procesamiento de PLN avanzado:
- **`prepare_data_bin.py` / `prepare_data_topic.py`**: Unifican y estandarizan datos en bruto.
- **`preprocesado.py`**: Limpieza profunda del texto (normalización, tokenización, lematización con spaCy) generando variables de texto específicas:
  - `text_ml`: Lematizado y limpio de stopwords para modelos de ML tradicionales (TF-IDF).
  - `text_dl`: Conserva el orden secuencial y más contexto para Deep Learning.
  - `text_beto`: Limpieza mínima, ideal para modelos Transformers (BETO).
- También extrae múltiples características numéricas estructurales (número de urls, signos de puntuación, mayúsculas) y elimina conflictos o duplicados. Genera el output `dataset_preprocesado_binario.csv`.

### 2. Análisis Exploratorio (`/EDA`)
Análisis exhaustivo antes de entrenar utilizando visualizaciones avanzadas.
- **`Eda_Dataset_Binario.ipynb`**: Compara la estructura y semántica entre noticias Fake/True.
- **`Eda_Dataset_Topic.ipynb`**: Balanceo de clases y agrupación por temáticas.

### 3. Modelos Deep Learning (Fake News) (`/MODELOS DEEP LEARNING`)
Para el problema binario, se han implementado múltiples arquitecturas de Deep Learning agrupadas en la carpeta `MODELOS DEEP LEARNING`. Cada subcarpeta contiene el Notebook de desarrollo, así como sus modelos guardados (`artefactos`), informes de resultados (`metricas`) y análisis de interpretabilidad o XAI (`xai`) usando técnicas avanzadas como **LIME**, **SHAP** y **Modelos de Lenguaje (LLMs)**:
- **`Modelo_CNN_Bin`**: Redes Neuronales Convolucionales adaptadas para NLP, buscando patrones locales en el texto. Incluye un notebook especializado (`Modelo_CNN_SHAP_LLM copy.ipynb`) que integra explicabilidad matemática con SHAP y generación de informes en lenguaje natural mediante IA Generativa (API de Gemini).
- **`Modelo_LSTM_Bin`**: Memoria a Corto y Largo Plazo, ideal para detectar relaciones secuenciales en la redacción de la noticia.
- **`Modelo_BiLSTM_Bin`**: Versión bidireccional de LSTM para capturar contexto tanto a futuro como a pasado.
- **`Modelo BiGRU_Bin`**: GRU bidireccional, una arquitectura de compuertas eficiente con alto rendimiento en análisis de secuencias complejas.

### 4. Modelo Clasificador de Tópicos (`/Modelo_Topics`)
Representa el producto final del pipeline de categorización. Desarrollado con **XGBoost** alimentado por matrices **TF-IDF**, integrando un preprocesamiento lingüístico avanzado que incluye Lematización y **Reemplazo de Entidades Nombradas (NER)**. Dispone de una CLI para realizar todo el ciclo de vida del dato de manera modular.

---

## 🛠️ Stack Tecnológico

- **Lenguaje Base**: Python 3.10+
- **Procesamiento de Lenguaje Natural (NLP)**: `spaCy` (modelo `es_core_news_sm`), `langdetect`, `wordcloud`
- **Machine Learning & Deep Learning**: `scikit-learn`, `XGBoost`, `tensorflow`, `keras`
- **Manipulación de Datos**: `pandas`, `numpy`
- **Explicabilidad (XAI) e IA Generativa**: `lime`, `shap`, `google-generativeai`
- **Visualización Analítica**: `matplotlib`, `seaborn`
- **Utilidades**: `tqdm`, `jupyter`

Para poder ejecutar todos y cada uno de los notebooks y scripts de este proyecto (incluyendo los modelos de Deep Learning y el de Tópicos), es necesario instalar las dependencias unificadas de manera global. Ejecuta el siguiente comando en la raíz del proyecto:

```bash
pip install -r requirements.txt
```

> **Nota importante sobre spaCy:**  
> Además de instalar los requisitos de la lista anterior, para que el preprocesamiento de NLP (lematización, NER, etc.) funcione correctamente, es obligatorio descargar el modelo de lenguaje en español de spaCy ejecutando el siguiente comando en tu entorno virtual o terminal:
> ```bash
> python -m spacy download es_core_news_sm
> ```
