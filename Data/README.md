# Preprocesado de datos

1. Carga y corrección de datos  
   * Lectura del dataset en formato CSV  
   * Corrección de problemas de codificación (encoding) en los textos  

2. Construcción del texto  
   * Unión del título y el contenido en un único texto completo  

3. Eliminación de duplicados  
   * Eliminación de noticias repetidas para evitar sesgos en el modelo  

4. Limpieza básica  
   * Pasar todo el texto a minúsculas  
   * Eliminar URLs  
   * Eliminar correos electrónicos  
   * Eliminar caracteres especiales innecesarios  

5. Normalización  
   * Reducir múltiples espacios a uno solo  
   * Normalizar repeticiones de caracteres (ej: "holaaaa" → "hola")  

6. Tratamiento de números  
   * Sustituir números por un token genérico ("numero")  

7. Filtrado de texto  
   * Eliminación de tokens no alfabéticos  
   * Eliminación de palabras muy cortas  

8. Lematización  
   * Reducción de las palabras a su forma base mediante spaCy  
   * Eliminación de stopwords (manteniendo negaciones como "no")  

9. Variable objetivo  
   * Conversión de la etiqueta booleana (True/False) a formato numérico (1/0)  

10. Generación del dataset final  
   * Creación del texto limpio listo para modelado  
   * Exportación del dataset en formato CSV para su uso posterior



