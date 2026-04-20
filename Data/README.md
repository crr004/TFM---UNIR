# DOCUMENTACIÓN DE LOS CONJUNTOS DE DATOS:

En la Carpeta `Data` se encuentran los conjuntos de datos que se utilizan para el proyecto. La documentación de estos es la siguiente:

## `Data_Bin_Classifier.csv` (generado por `prepare_data_bin.py`)

Este dataset es el resultado de concatenar los siguientes datasets:

* `Fake News Detector Spanish News Edition (Kaggle)`: https://www.kaggle.com/code/msantrod/fake-news-detector-spanish-news-edition/input?select=train.csv

* `The Spanish Fake News Corpus (Github)`: https://github.com/jpposadas/FakeNewsCorpusSpanish

* `Fake-news-latam-omdena (Hugging Face)`: https://huggingface.co/datasets/IsaacRodgz/Fake-news-latam-omdena

* `Spanish Political Fake News (Kaggle)`: https://www.kaggle.com/datasets/javieroterovizoso/spanish-political-fake-news

## `Data_Topic_Classifier.csv` (generado por `prepare_data_topic.py`)

* `MLSUM (Multilingual Summarization Corpus) - Subconjunto en Español`: https://huggingface.co/datasets/reciTAL/mlsum/tree/refs%2Fconvert%2Fparquet/es

    * Más de 260.000 noticias reales extraídas de medios en castellano (solo en su partición de entrenamiento), lo que lo hace mucho más grande que los anteriores.

    * Se deben descargar los archivos de train y validation.
