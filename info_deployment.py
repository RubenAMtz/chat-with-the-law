from azure.ai.ml import MLClient
import os
from azure.ai.ml.entities._deployment.deployment_settings import OnlineRequestSettings
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

dotenv_path = f".env.prod"
load_dotenv(dotenv_path=dotenv_path)

subscription_id = os.getenv("AZUREML_SUBSCRIPTION_ID")
resource_group = os.getenv("AZUREML_RESOURCE_GROUP")
workspace = os.getenv("AZUREML_WORSKPACE_NAME")

ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)

# Define an endpoint name
endpoint_name = "hybrid-search-chat"

deployment_info = ml_client.online_endpoints.get(name=endpoint_name)
print(deployment_info)