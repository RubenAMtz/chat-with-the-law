# download blob from azure storage

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os
from dotenv import load_dotenv
import json
import numpy as np

dotenv_path = f".env.prod"
load_dotenv(dotenv_path=dotenv_path)

connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

# download blob from azure storage
blob_client = blob_service_client.get_blob_client(container=container_name, blob="embeddings/leyes_chromadb_512_40_ada_002_recursive/index/index_7f34d7a0-1444-407e-ac98-63e657161537.bin")
downloaded_blob = blob_client.download_blob()
with open("index_7f34d7a0-1444-407e-ac98-63e657161537.bin", "wb") as my_blob:
    my_blob.write(downloaded_blob.readall())
