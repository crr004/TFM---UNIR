# Visualizaciones del notebook BETO

Este README resume las gráficas añadidas al notebook `notebook_beto_graficas_con_curvas.ipynb`.

## Requisitos previos

Para que todas las gráficas tengan datos, ejecuta primero:

1. la sección de entrenamiento con `RUN_TRAIN = True`, y
2. la sección de evaluación con `RUN_EVALUATE = True`.

El entrenamiento guarda el historial del `Trainer` en:

```text
OUTPUT_DIR/metrics/trainer_log_history.json
```

La evaluación guarda las predicciones de test en:

```text
OUTPUT_DIR/metrics/test_predictions.csv
```

La sección de visualización lee esos dos archivos. Si alguno falta, el notebook mostrará un mensaje indicando qué paso debe ejecutarse antes.

## Gráficas incluidas

### 1. Curva combinada de loss de entrenamiento y validación

Muestra en una misma figura la pérdida de entrenamiento (`loss`) y la pérdida de validación (`eval_loss`) registradas durante el fine-tuning.

Esta es una de las gráficas principales para analizar la evolución del modelo. La curva de entrenamiento suele tener más puntos porque se registra cada cierto número de steps (`logging_steps`), mientras que la curva de validación normalmente tiene un punto por época si la evaluación está configurada con `eval_strategy="epoch"`.

Sirve para comprobar si el modelo aprende y para detectar posibles señales de sobreajuste. Si la loss de entrenamiento sigue bajando pero la loss de validación se estanca o empieza a subir, podría indicar que el modelo está memorizando el conjunto de entrenamiento.

> Nota: el conjunto de test no se usa para esta curva. El test debe reservarse para la evaluación final independiente, no para monitorizar el entrenamiento por época.

### 2. Loss de entrenamiento

Muestra la pérdida (`loss`) registrada durante el fine-tuning en función del step de entrenamiento.

Sirve para comprobar si el modelo está aprendiendo. Lo normal es que la curva tienda a bajar, aunque puede tener oscilaciones por el tamaño de batch, el learning rate o la dificultad de los ejemplos.

### 3. Loss de validación

Muestra la pérdida de validación (`eval_loss`) por época o por step, según lo que haya registrado el `Trainer`.

Es útil para detectar sobreajuste. Si la loss de entrenamiento baja pero la loss de validación empieza a subir, el modelo puede estar memorizando el conjunto de entrenamiento.

### 4. Métricas de validación

Muestra conjuntamente:

- accuracy,
- F1,
- precision,
- recall.

Estas métricas se calculan sobre el conjunto de validación durante el entrenamiento. En clasificación binaria REAL/FAKE, el F1 es especialmente útil porque resume el equilibrio entre precision y recall.

### 5. Learning rate

Muestra el learning rate registrado durante el entrenamiento.

Es útil para verificar el efecto de la configuración del scheduler y del `warmup_ratio`. En este notebook ayuda a comprobar que el learning rate evoluciona como se espera durante el fine-tuning.

### 6. Matriz de confusión en test

Muestra los aciertos y errores del modelo sobre el conjunto de test.

Las filas representan la clase real y las columnas la clase predicha. Permite ver rápidamente si el modelo confunde más noticias REAL como FAKE o noticias FAKE como REAL.

### 7. Curva ROC y curva precision-recall

La curva ROC muestra la relación entre true positive rate y false positive rate para distintos umbrales de decisión usando `p(FAKE)`.

La curva precision-recall muestra el intercambio entre precision y recall para distintos umbrales. Esta curva suele ser especialmente informativa cuando existe desbalance entre clases.

### 8. Histograma de probabilidades `p(FAKE)` por clase real

Muestra la distribución de la probabilidad asignada a la clase FAKE, separando los ejemplos cuya clase real es REAL y FAKE.

Idealmente, los ejemplos REAL deberían concentrarse cerca de 0 y los ejemplos FAKE cerca de 1. Los ejemplos cerca del umbral 0.5 son los más ambiguos para el modelo.

## Función principal

La celda final de la sección de visualización ejecuta:

```python
plot_all_training_and_test_charts(OUTPUT_DIR)
```

Esta función genera todas las gráficas anteriores en orden, incluida la curva combinada de `Train loss` frente a `Validation loss`. También puedes llamar cada función por separado si solo quieres inspeccionar una gráfica concreta.

La función añadida para la curva combinada es:

```python
plot_train_validation_loss(OUTPUT_DIR)
```

Esta función lee `OUTPUT_DIR/metrics/trainer_log_history.json`, separa las entradas de entrenamiento y validación del historial del `Trainer`, y representa ambas pérdidas en una misma figura.
