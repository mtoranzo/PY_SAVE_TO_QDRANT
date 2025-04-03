import os
import glob
import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import AutoTokenizer, AutoModel
import torch

# Configuración de Qdrant
QDRANT_URL = "http://localhost:6333"  # Cambia esto si tu Qdrant está en otro host
QDRANT_API_KEY = "KEY"  # Reemplaza con tu clave API
COLLECTION_NAME = "md_files_v3"

# Configuración del modelo de embeddings (ejemplo con Hugging Face)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  
EMBEDDING_MODEL_VECTOR_SIZE = 384

# Función para inicializar el cliente de Qdrant
def init_qdrant_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Función para verificar si la colección existe
def collection_exists(client, collection_name):
    collections = client.get_collections().collections
    return any(collection.name == collection_name for collection in collections)

# Función para eliminar una colección en Qdrant
def delete_collection(client, collection_name):
    if collection_exists(client, collection_name):
        client.delete_collection(collection_name=collection_name)
        print(f"La colección '{collection_name}' ha sido eliminada.")
    else:
        print(f"La colección '{collection_name}' no existe.")


# Función para crear una colección en Qdrant si no existe
def create_collection_if_not_exists(client, collection_name, vector_size):
    if not collection_exists(client, collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE, on_disk=True)
        )
        print(f"Colección '{collection_name}' creada en Qdrant.")
    else:
        print(f"La colección '{collection_name}' ya existe. No se creó nuevamente.")

# Función para generar embeddings (ejemplo con Hugging Face)
def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Función para generar un ID numérico único basado en el nombre del archivo
def generate_numeric_id(file_name):
    return int(hashlib.sha256(file_name.encode('utf-8')).hexdigest(), 16) % 10**16

# Función principal para procesar los archivos .md y cargarlos en Qdrant
def process_md_files(base_dir, collection_name, client, tokenizer, model):
    # Obtener todos los archivos .md en el directorio y subdirectorios
    md_files = glob.glob(os.path.join(base_dir, "**/*.md"), recursive=True)
    print(f"Encontrados {len(md_files)} archivos .md en '{base_dir}'.")

    # Procesar cada archivo .md
    for file_path in md_files:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        
        # Generar el embedding del contenido del archivo
        embedding = get_embedding(content, tokenizer, model)
        
        # Crear un ID único para cada archivo (usando un hash numérico)
        file_id = generate_numeric_id(os.path.basename(file_path))
        
        # Crear el payload con metadatos
        payload = {
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
            "content": content
        }

        # "content": content[:100]  # Solo una parte del contenido para el payload
        
        # Verificar si el punto ya existe en Qdrant
        existing_point = client.retrieve(
            collection_name=collection_name,
            ids=[file_id]  # Asegúrate de usar el ID numérico aquí
        )
        
        if existing_point:
            # Si el punto existe, actualizarlo
            client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=file_id,  # Asegúrate de usar el ID numérico aquí
                        vector=embedding.tolist(),
                        payload=payload
                    )
                ],
                wait=True
            )
            print(f"Archivo '{file_path}' actualizado en Qdrant.")
        else:
            # Si el punto no existe, insertarlo
            client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=file_id,  # Asegúrate de usar el ID numérico aquí
                        vector=embedding.tolist(),
                        payload=payload
                    )
                ]
            )
            print(f"Archivo '{file_path}' cargado en Qdrant.")

    print("Todos los archivos .md han sido procesados y cargados en Qdrant.")

# Función para verificar los datos en Qdrant
def verify_data(client, collection_name, tokenizer, model):
    # Ejemplo de consulta
    query_text = "Que es un lead"
    query_vector = get_embedding(query_text, tokenizer, model).tolist()

    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=5
    )

    print("\nResultados de la búsqueda:")
    for result in search_result:
        print(f"ID: {result.id}, Score: {result.score}, Payload: {result.payload}")

# Punto de entrada del script
if __name__ == "__main__":
    # Inicializar el cliente de Qdrant
    qdrant_client = init_qdrant_client()

    # Eliminar la colección en Qdrant si existe
    delete_collection(qdrant_client, COLLECTION_NAME) 

    # Crear la colección en Qdrant si no existe
    create_collection_if_not_exists(qdrant_client, COLLECTION_NAME, vector_size=EMBEDDING_MODEL_VECTOR_SIZE)  # Tamaño del modelo all-MiniLM-L6-v2

    # Cargar el modelo de embeddings
    print("Cargando el modelo de embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL)

    # Procesar los archivos .md y cargarlos en Qdrant
    base_directory = "D:\\REPOSITORIOS LOCALES\\DEALER_SERVICES\\07_PY_FILE_TO_MD\\app_data"  # Cambia esto por la ruta de tu directorio
    process_md_files(base_directory, COLLECTION_NAME, qdrant_client, tokenizer, model)

    # Verificar los datos en Qdrant
    verify_data(qdrant_client, COLLECTION_NAME, tokenizer, model)