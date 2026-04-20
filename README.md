# INFORMACIÓN DEL REPOSITORIO (TFM)

Este repositorio contiene los scripts y notebooks necesarios para el desarrollo del Trabajo de Fin de Máster.


## `Data`

Contiene los datasets utilizados en el proyecto.

### `Data/Bin`

Contiene los scripts necesarios para preparar los datos para el clasificador binario.

### `Data/Topic`

Contiene los scripts necesarios para preparar los datos para el clasificador de tópicos.


### `Data/prepare_data_bin.py`

Este script se encarga de preparar los datos para el clasificador binario. En concreto, se encarga de: 
    * Cargar los datasets de diferentes fuentes.
    * Preprocesar los datos.
    * Concatenar los datasets.
    * Guardar los datasets concatenados.

### `Data/prepare_data_topic.py`

Este script se encarga de preparar los datos para el clasificador de tópicos. En concreto, se encarga de: 
    * Cargar los datasets de diferentes fuentes.
    * Preprocesar los datos.
    * Concatenar los datasets.
    * Guardar los datasets concatenados.

---

## `EDA/Eda_Dataset_Binario.ipynb`

Este notebook se encarga de realizar un análisis exploratorio de Data_Bin_Classifier.csv, que será el conjunto de datos dedicado a predecir si un documento es verdadero o falso.

## `EDA/Eda_Dataset_Multiclase.ipynb`

Este notebook se encarga de realizar un análisis exploratorio de Data_Multi_Classifier.csv, que será el conjunto de datos dedicado a predecir la categoría de un documento (político, social, económico, etc.). 




