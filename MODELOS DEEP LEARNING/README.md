# 🧠 Modelos de Deep Learning para Detección de Fake News

Este directorio contiene todas las experimentaciones e implementaciones de modelos basados en **Deep Learning** orientados a la clasificación binaria de noticias (Fake News vs. True News). 

Para abordar la complejidad semántica y estructural inherente al lenguaje natural, se han desarrollado cuatro arquitecturas distintas. Cada una aborda el problema de la extracción de características y comprensión del texto desde una perspectiva topológica diferente, permitiendo comparar el rendimiento empírico de enfoques espaciales (CNN) frente a enfoques puramente secuenciales o temporales (RNNs, LSTM, GRU).

A continuación, se detalla rigurosamente la naturaleza y la motivación detrás de cada uno de los modelos implementados.

---

## 🏗️ Arquitecturas Implementadas

### 1. Modelo CNN (Redes Neuronales Convolucionales)
**Directorio:** `/Modelo_CNN_Bin`

Aunque tradicionalmente asociadas a la Visión por Computadora, las CNNs (Convolutional Neural Networks) son extremadamente eficaces en NLP (Procesamiento de Lenguaje Natural) para la extracción de características **locales**.
- **Fundamento Teórico:** Las capas convolucionales aplican filtros (kernels) de distintos tamaños (p. ej., bi-gramas, tri-gramas) sobre las secuencias de texto embebidas. 
- **Objetivo en el Proyecto:** Detectar patrones, expresiones específicas o *n-gramas* clave que frecuentemente delatan el lenguaje sensacionalista o carente de rigor típico de una Fake News.
- **Ventaja Principal:** Alta velocidad de entrenamiento y excelente capacidad para aislar "palabras gatillo" independientes del contexto a largo plazo.

### 2. Modelo LSTM (Long Short-Term Memory)
**Directorio:** `/Modelo_LSTM_Bin`

Las LSTM son un tipo avanzado de Redes Neuronales Recurrentes (RNN) diseñadas específicamente para resolver el problema del desvanecimiento del gradiente en el aprendizaje de secuencias largas.
- **Fundamento Teórico:** Utilizan una arquitectura celular compleja compuesta por tres compuertas (entrada, olvido y salida) que deciden activamente qué información del contexto histórico del texto debe mantenerse en la memoria y cuál debe descartarse.
- **Objetivo en el Proyecto:** Analizar la narrativa de la noticia de principio a fin, manteniendo la coherencia semántica global. Es capaz de recordar el contexto de una afirmación hecha en el primer párrafo al evaluar la conclusión de la noticia.
- **Ventaja Principal:** Comprensión profunda de secuencias largas y modelado de dependencias a largo plazo, crucial para identificar desinformación que se construye narrativamente.

### 3. Modelo BiLSTM (LSTM Bidireccional)
**Directorio:** `/Modelo_BiLSTM_Bin`

Una evolución directa del modelo LSTM estándar que procesa la entrada en dos direcciones simultáneas.
- **Fundamento Teórico:** La secuencia de texto se alimenta a dos capas LSTM ocultas: una procesa el texto hacia adelante (de principio a fin) y la otra hacia atrás (de fin a principio). Los estados ocultos de ambas se concatenan.
- **Objetivo en el Proyecto:** Entender el contexto completo de una palabra. El significado de un término en una noticia suele depender no solo de las palabras que lo preceden, sino también de las que le siguen. 
- **Ventaja Principal:** Representación semántica mucho más rica y robusta. Suele ofrecer un rendimiento superior (accuracy y F1-Score) frente a las LSTM unidireccionales en tareas de clasificación de textos complejos.

### 4. Modelo BiGRU (Gated Recurrent Unit Bidireccional)
**Directorio:** `/Modelo BiGRU_Bin`

Las GRU son una variante optimizada de las LSTM que simplifican el flujo de información reduciendo el número de compuertas (fusionando la compuerta de olvido y entrada).
- **Fundamento Teórico:** Al igual que la BiLSTM, procesa la información en ambas direcciones (pasado y futuro), pero su celda de memoria requiere menos parámetros computacionales, haciendo que el modelo sea más rápido y fácil de converger.
- **Objetivo en el Proyecto:** Conseguir un rendimiento predictivo equivalente (o incluso superior en datasets con menos volumen) al de una BiLSTM, pero con una mayor eficiencia computacional y menor riesgo de sobreajuste (*overfitting*).
- **Ventaja Principal:** Eficiencia arquitectónica. Suele ser el modelo secuencial óptimo cuando se busca un equilibrio entre la captura del contexto bidireccional y el coste computacional.

---

## 📂 Estructura Interna de Cada Modelo

Dentro de cada una de las subcarpetas mencionadas, se mantiene una estructura estandarizada para asegurar la reproducibilidad y el correcto despliegue de los resultados:

- `📓 Modelo_*.ipynb`: Notebook principal donde se detalla el pipeline completo de carga de datos preprocesados, definición de la topología de la red (capas de Embedding, Convolución/Recurrencia, Dropout, Dense), compilación, entrenamiento y validación cruzada/hold-out.
- `📂 artefactos/`: Directorio destinado a almacenar los modelos entrenados serializados (p. ej., en formato `.h5` o `.keras`), así como los tokenizadores utilizados para mapear el vocabulario, fundamentales para la etapa de inferencia.
- `📂 metricas/`: Almacena las evidencias del rendimiento del modelo. Incluye gráficas de evolución del *Loss* y *Accuracy* por época, así como Matrices de Confusión y reportes de clasificación detallados.
- `📂 xai/` *(Explainable AI)*: Contiene análisis de explicabilidad (comúnmente generados mediante librerías como **LIME** o **SHAP**) que permiten interpretar localmente qué subconjuntos de palabras o frases empujaron al modelo a clasificar una noticia concreta como Verdadera o Falsa, garantizando la transparencia de la IA.

---

## 🚀 Requisitos para la Ejecución

Debido a la naturaleza de estos modelos, se recomienda encarecidamente contar con soporte de aceleración por hardware (GPU) para reducir los tiempos de entrenamiento.
El entorno está gestionado principalmente a través de `TensorFlow` / `Keras`. Las dependencias exactas se encuentran unificadas en el `requirements.txt` en la raíz del repositorio.
