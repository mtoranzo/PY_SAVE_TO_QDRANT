# PY_SAVE_TO_QDRANT

## Descripción
Esta aplicación permite guardar datos en Qdrant, un motor de búsqueda vectorial de alto rendimiento. Es útil para almacenar y buscar datos vectoriales, como embeddings generados por modelos de machine learning.

## Propósito
El propósito principal de esta aplicación es facilitar el almacenamiento y la recuperación de datos vectoriales en Qdrant, lo que es ideal para aplicaciones de búsqueda semántica, sistemas de recomendación y análisis de datos.

## Instalación
Sigue los pasos a continuación para instalar y configurar la aplicación:

```bash
# Crear un entorno virtual
python -m venv venv

# Activar el entorno virtual
./venv/Scripts/Activate.ps1  # En Windows
source ./venv/bin/activate   # En Linux/MacOS

# Instalar las dependencias
pip install -r requirements.txt
```

## Ejecución de la aplicación
Para ejecutar la aplicación, utiliza los siguientes comandos:

```bash
# Ejecutar el script principal
python save_to_qdrant.py

# Desactivar el entorno virtual al finalizar
deactivate
```

## Uso
1. Asegúrate de que Qdrant esté configurado y en ejecución en tu entorno.
2. Configura los parámetros necesarios en el archivo `save_to_qdrant.py` (como la dirección del servidor Qdrant y los datos a guardar).
3. Ejecuta el script como se indica en la sección de ejecución.
4. Verifica que los datos se hayan almacenado correctamente en Qdrant.