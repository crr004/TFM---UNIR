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
 ┣ 📂 Modelo_CNN_Bin/                       # 🧠 MODELO DEEP LEARNING: CNN (FAKE NEWS)
 ┃ ┣ 📂 artefactos/                         # Modelos y tokenizadores serializados
 ┃ ┣ 📂 metricas/                           # Curvas de entrenamiento, matrices de confusión
 ┃ ┣ 📂 xai/                                # Análisis de Explicabilidad (LIME)
 ┃ ┗ 📓 Modelo_CNN_Bin.ipynb                # Desarrollo del modelo CNN
 ┃
 ┣ 📂 Modelo_LSTM_Bin/                      # 🧠 MODELO DEEP LEARNING: LSTM (FAKE NEWS)
 ┃ ┣ 📂 artefactos/, metricas/, xai/
 ┃ ┗ 📓 Modelo_LSTM.ipynb                   # Desarrollo del modelo LSTM
 ┃
 ┣ 📂 Modelo_BiLSTM_Bin/                    # 🧠 MODELO DEEP LEARNING: BiLSTM (FAKE NEWS)
 ┃ ┣ 📂 artefactos/, metricas/, xai/
 ┃ ┗ 📓 Modelo_BiLSTM.ipynb                 # Desarrollo del modelo BiLSTM
 ┃
 ┣ 📂 Modelo BiGRU_Bin/                     # 🧠 MODELO DEEP LEARNING: BiGRU (FAKE NEWS)
 ┃ ┣ 📂 artefactos/, metricas/, xai/
 ┃ ┗ 📓 Modelo_BiGRU.ipynb                  # Desarrollo del modelo BiGRU
 ┃
 ┣ 📂 Modelo_Clasificador_Topics/           # 🧠 MODELO DE PRODUCCIÓN (TOPICS)
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

### 3. Modelos Deep Learning (Fake News)
Para el problema binario, se han implementado múltiples arquitecturas de Deep Learning. Cada carpeta contiene el Notebook de desarrollo, así como sus modelos guardados (`artefactos`), informes de resultados (`metricas`) y análisis de interpretabilidad o XAI (`xai`) usando técnicas como LIME:
- **`Modelo_CNN_Bin`**: Redes Neuronales Convolucionales adaptadas para NLP, buscando patrones locales en el texto.
- **`Modelo_LSTM_Bin`**: Memoria a Corto y Largo Plazo, ideal para detectar relaciones secuenciales en la redacción de la noticia.
- **`Modelo_BiLSTM_Bin`**: Versión bidireccional de LSTM para capturar contexto tanto a futuro como a pasado.
- **`Modelo BiGRU_Bin`**: GRU bidireccional, una arquitectura de compuertas eficiente con alto rendimiento en análisis de secuencias complejas.

### 4. Modelo Clasificador de Tópicos (`/Modelo_Clasificador_Topics`)
Representa el producto final del pipeline de categorización. Desarrollado con **XGBoost** alimentado por matrices **TF-IDF**, integrando un preprocesamiento lingüístico avanzado que incluye Lematización y **Reemplazo de Entidades Nombradas (NER)**. Dispone de una CLI para realizar todo el ciclo de vida del dato de manera modular.

---

## 🛠️ Stack Tecnológico

- **Lenguaje Base**: Python 3.10+
- **Procesamiento de Lenguaje Natural (NLP)**: `spaCy` (modelo `es_core_news_sm`), `langdetect`, `wordcloud`
- **Machine Learning & Deep Learning**: `scikit-learn`, `XGBoost`, `tensorflow`, `keras`
- **Manipulación de Datos**: `pandas`, `numpy`
- **Explicabilidad (XAI)**: `lime`
- **Visualización Analítica**: `matplotlib`, `seaborn`
- **Utilidades**: `tqdm`, `jupyter`

Para instalar las dependencias necesarias de manera global, ejecuta:
```bash
pip install -r requirements.txt
```
