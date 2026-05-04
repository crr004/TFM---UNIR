# README del preprocesado de datos

## TL;DR

Este fichero toma el dataset binario de noticias, une `Title` + `Content`, limpia y normaliza el texto, elimina duplicados y conflictos, y genera varias versiones del mismo contenido para cada tipo de modelo: `text_ml` para ML clásico con TF-IDF, `text_dl` para LSTM/CNN y `text_beto` para BETO. Además añade features estructurales como número de URLs, mayúsculas, signos y dígitos.

Este script prepara el dataset binario de noticias falsas para distintos tipos de modelos de PLN y ML. Su objetivo no es solo limpiar texto, sino generar varias vistas del mismo contenido para que cada familia de modelos reciba la representación más adecuada.

## Archivo principal

- `Data/preprocesado.py`

## Entradas

El script lee el archivo:

- `Data/Bin/Data_Bin_Classifier.csv`

Este dataset debe contener, como mínimo, estas columnas:

- `Title`
- `Content`
- `State`

También se conservan, si existen:

- `ID`
- `Dataset`

## Salida

El script genera:

- `Data/dataset_preprocesado_binario.csv`

Ese archivo queda listo para entrenar modelos clásicos, modelos de deep learning y modelos tipo BETO.

## Qué hace el preprocesado

### 1. Construye `full_text`

Se crea la columna:

- `full_text = Title + Content`

Después se normalizan espacios y se eliminan vacíos.

**Uso:** representa el texto original unido, útil como referencia general y como base para el resto de transformaciones.

### 2. Extrae características estructurales

Sobre `full_text` se generan variables numéricas que capturan señales típicas de noticias dudosas o sensacionalistas:

- `num_chars`
- `num_words`
- `num_exclamaciones`
- `num_interrogaciones`
- `num_urls`
- `num_emails`
- `num_mayusculas`
- `num_digitos`
- `num_ellipsis`
- `num_quotes`

**Uso:** estas variables pueden añadirse a modelos clásicos o combinarse con modelos neuronales como features auxiliares.

### 3. Deduplica y elimina conflictos

Antes de limpiar definitivamente, el script crea una versión normalizada del texto para detectar duplicados.

Además:

- elimina textos repetidos
- detecta casos donde el mismo texto aparece con etiquetas distintas
- descarta esos conflictos para evitar fuga de información o ruido en el entrenamiento

Esto es importante porque el duplicado de noticias entre clases puede inflar artificialmente las métricas.

### 4. Crea `text_ml`

Esta es la representación pensada para modelos clásicos como:

- SVM
- Random Forest
- Logistic Regression
- Naive Bayes

Procesado aplicado:

- normalización Unicode
- minúsculas
- reemplazo de URLs por `url`
- reemplazo de correos por `email`
- reemplazo de números por `numero`
- corrección de repeticiones largas
- eliminación de boilerplate típico de medios
- lematización con spaCy
- eliminación de stopwords, excepto negaciones clave
- eliminación de tokens muy cortos o no alfabéticos

**Uso recomendado:** `TF-IDF + text_ml`

Esta versión compacta y lematizada suele funcionar mejor con modelos lineales y árboles porque reduce ruido y dimensionalidad.

### 5. Crea `text_dl`

Esta es la representación pensada para deep learning generalista, por ejemplo:

- LSTM
- GRU
- CNN para texto
- BiLSTM

Procesado aplicado:

- normalización Unicode
- minúsculas
- reemplazo de URLs, correos y números por tokens
- corrección de repeticiones largas
- eliminación de boilerplate
- sin lematización
- sin eliminación agresiva de stopwords

**Uso recomendado:** tokenizador propio + secuencias para redes neuronales.

La idea es conservar mejor el orden y la forma original de las palabras, porque estos modelos aprenden patrones secuenciales y se benefician de texto menos transformado que el de `text_ml`.

### 6. Crea `text_beto`

Esta es la versión mínima para BETO o transformers similares.

Procesado aplicado:

- normalización Unicode
- limpieza de espacios múltiples
- conservación casi total del texto original
- no se pasa a minúsculas de forma obligatoria
- no se lematiza
- no se eliminan stopwords

**Uso recomendado:** BETO / Transformers

Aquí la prioridad es no destruir información contextual, porque el propio modelo se encarga de la tokenización subword.

## Columnas finales generadas

El CSV final contiene, entre otras, estas columnas:

- `full_text`
- `text_ml`
- `text_dl`
- `text_beto`
- `label`
- las variables estructurales

## Codificación de etiquetas

El script transforma `State` a `label` con este criterio:

- `fake`, `false`, `1` -> `1`
- `real`, `true`, `no fake`, `0` -> `0`

Si aparece algún valor de `State` no reconocido, el script lo detecta y detiene la ejecución para evitar etiquetas incorrectas.

## Criterios de limpieza usados

### Se conservan como tokens especiales

- `url`
- `email`
- `numero`

### Se preservan como señales relevantes

- negaciones como `no`, `nunca`, `sin`, `tampoco`
- repeticiones de caracteres reducidas, no eliminadas por completo
- ciertos caracteres de puntuación útiles para DL y BETO

### Se eliminan o normalizan

- HTML o entidades Unicode raras
- espacios repetidos
- boilerplate textual frecuente en portales de noticias
- caracteres claramente no informativos para el modelo clásico

## Recomendación de uso por modelo

- **Modelos clásicos con TF-IDF:** `text_ml`
- **Modelos de deep learning secuencial:** `text_dl`
- **BETO / Transformers:** `text_beto`
- **Features extra:** columnas numéricas estructurales

## Flujo recomendado de trabajo

1. Ejecutar `preprocesado.py`
2. Cargar `dataset_preprocesado_binario.csv`
3. Elegir la columna de texto según el modelo
4. Si hace falta, concatenar las features estructurales al vector final

## Nota práctica

Este preprocesado está diseñado para no quedarse en una única representación. La intención es que el mismo dataset sirva bien a varios enfoques de modelado sin forzar una limpieza excesiva que perjudique a modelos neuronales o transformers.
