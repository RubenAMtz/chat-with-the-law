# download blob from azure storage

from azure.storage.blob import BlobServiceClient
import os
from dotenv import load_dotenv

dotenv_path = f".env.prod"
load_dotenv(dotenv_path=dotenv_path)

connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
key = os.getenv("AZURE_STORAGE_ACCESS_KEY")
container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")

print(f"Connection string: {connection_string}")
print(f"Key: {key}")
print(f"Container name: {container_name}")

blob_service_client = BlobServiceClient.from_connection_string(connection_string, credential=key)
container_client = blob_service_client.get_container_client(container_name)

# download blobs recursively from from azure storage folder
embeddings_folder = 'leyes_chromadb_512_40_ada_002_recursive'

blobs = container_client.list_blobs(name_starts_with=embeddings_folder)
for blob in blobs:
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob.name)
    downloaded_blob = blob_client.download_blob()
    if not os.path.exists(blob.name):
        os.makedirs(os.path.dirname(blob.name), exist_ok=True)
    with open(blob.name, "wb") as my_blob:
        my_blob.write(downloaded_blob.readall())

