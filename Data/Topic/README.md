## 📊 Conjunto de Datos: MLSUM

Para el correcto funcionamiento de este proyecto, es necesario descargar y organizar manualmente el dataset **MLSUM**.

### 🔗 Enlace de descarga
Puedes obtener los archivos desde el repositorio de Hugging Face:
[MLSUM - Parquet Files (Spanish)](https://huggingface.co/datasets/reciTAL/mlsum/tree/refs%2Fconvert%2Fparquet/es)

### 📝 Descripción
* **Volumen:** Contiene más de **260,000 noticias reales** extraídas de medios en castellano (solo en su partición de entrenamiento), superando en tamaño a los datasets anteriores.
* **Requisito:** Se deben descargar obligatoriamente los archivos correspondientes a las particiones de **train**, **validation** y **test**.

### 📂 Estructura de directorios
Los archivos descargados deben ubicarse en la siguiente ruta siguiendo este esquema:

```text
Topic/
└── MLSUM/
    ├── train/          # Archivos .parquet de entrenamiento
    ├── validation/     # Archivos .parquet de validación
    └── test/           # Archivos .parquet de prueba
