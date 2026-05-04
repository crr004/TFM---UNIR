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
 ┃ ┣ 📜 prepare_data_bin.py                 # Script: Unificación y limpieza (Binario)
 ┃ ┣ 📜 prepare_data_topic.py               # Script: Unificación y limpieza (Topics)
 ┃ ┗ 📜 README.md                           # Documentación técnica de los orígenes de datos
 ┃
 ┣ 📂 EDA/                                  # 📊 ANÁLISIS EXPLORATORIO DE DATOS (EDA)
 ┃ ┣ 📓 Eda_Dataset_Binario.ipynb           # Análisis profundo de noticias falsas vs reales
 ┃ ┗ 📓 Eda_Dataset_Topic.ipynb             # Análisis lingüístico y distribución de tópicos
 ┃
 ┣ 📂 Modelo_Clasificador_Topics/           # 🧠 MODELO DE PRODUCCIÓN (TOPICS)
 ┃ ┣ 📂 src/                                # Código fuente encapsulado y modularizado
 ┃ ┃ ┣ 📜 __init__.py                       # Inicializador del paquete
 ┃ ┃ ┣ 📜 main.py                           # Orquestador CLI
 ┃ ┃ ┣ 📜 preprocessing.py                  # Limpieza, spaCy NER, filtrado y TF-IDF
 ┃ ┃ ┣ 📜 train.py                          # Entrenamiento XGBoost (con soporte GPU)
 ┃ ┃ ┣ 📜 evaluate.py                       # Generación de métricas y reporte
 ┃ ┃ ┗ 📜 predict.py                        # Pipeline de inferencia end-to-end
 ┃ ┣ 📂 artefactos/                         # Modelos y codificadores serializados (.pkl)
 ┃ ┣ 📂 figuras/                            # Matrices de confusión y curvas de aprendizaje
 ┃ ┣ 📓 Modelo_XGBoost_Topic.ipynb          # Notebook original de desarrollo del modelo
 ┃ ┣ 📜 requirements.txt                    # Dependencias específicas de este módulo
 ┃ ┗ 📜 README.md                           # Guía detallada de uso y ejecución del modelo
 ┃
 ┣ 📜 .gitignore                            # Archivos y carpetas ignorados por Git
 ┗ 📜 README.md                             # Este archivo (Guía General)
```

---

## 🧩 Descripción de los Módulos Principales

### 1. Preparación de Datos (`/Data`)
Esta carpeta contiene la materia prima del proyecto. Los scripts de Python se encargan de automatizar la extracción desde múltiples fuentes (CSVs, Parquets, Excel), normalizar los nombres de columnas, limpiar textos vacíos, y unificar todo en dos datasets maestros listos para el modelado:
- **`prepare_data_bin.py`**: Genera el dataset consolidado. Agrupa 4 fuentes distintas y estandariza la variable objetivo: `True` (Noticia Real) o `False` (Fake News).
- **`prepare_data_topic.py`**: Genera el dataset consolidado de temáticas extraído de fuentes de noticias generalistas.

### 2. Análisis Exploratorio (`/EDA`)
Antes de construir los modelos, se realiza un exhaustivo análisis lingüístico utilizando bibliotecas de vanguardia como **spaCy**.
- **`Eda_Dataset_Binario.ipynb`**: Estudia las diferencias estructurales entre noticias reales y falsas (longitud de textos, uso de mayúsculas, signos de puntuación, N-gramas y palabras más distintivas).
- **`Eda_Dataset_Topic.ipynb`**: Analiza el desbalanceo de clases y justifica las decisiones de agrupamiento necesarias para mejorar la calidad y rendimiento del futuro modelo predictivo.

### 3. Modelo Clasificador de Tópicos (`/Modelo_Clasificador_Topics`)
Representa el **producto final** del pipeline de categorización. Ha sido reescrito y encapsulado siguiendo los mejores estándares de la industria del software:
- Utiliza **XGBoost** alimentado por matrices **TF-IDF**.
- Integra un preprocesamiento lingüístico avanzado que incluye Lematización y **Reemplazo de Entidades Nombradas (NER)**.
- Dispone de una Interfaz de Línea de Comandos (CLI) que permite ejecutar los ciclos del proyecto de forma independiente (`preprocess`, `train`, `evaluate`, `predict`).
- > 💡 *Dirígete al `README.md` de esa carpeta para ver las instrucciones de uso paso a paso.*

---

## 🛠️ Stack Tecnológico

- **Lenguaje Base**: Python 3.10+
- **Procesamiento de Lenguaje Natural (NLP)**: `spaCy` (modelo `es_core_news_lg`), `langdetect`
- **Machine Learning**: `scikit-learn`, `XGBoost`
- **Manipulación de Datos**: `pandas`, `numpy`
- **Visualización Analítica**: `matplotlib`, `seaborn`
