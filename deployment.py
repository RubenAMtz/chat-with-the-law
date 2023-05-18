from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
import os
from azure.ai.ml.entities._deployment.deployment_settings import OnlineRequestSettings
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
# from keys import subscription_id, resource_group, workspace_name

dotenv_path = f".env.prod"
load_dotenv(dotenv_path=dotenv_path)

subscription_id = os.getenv("AZUREML_SUBSCRIPTION_ID")
resource_group = os.getenv("AZUREML_RESOURCE_GROUP")
workspace = os.getenv("AZUREML_WORSKPACE_NAME")

# get a handle to the workspace
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)

# Define an endpoint name
endpoint_name = "hybrid-search-chat"

# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name = endpoint_name, 
    description="this is an endpoint to chat with an agent",
    auth_mode="key"
)

model = Model(path="embeddings/leyes_chromadb_512_40_ada_002_recursive/index/index_7f34d7a0-1444-407e-ac98-63e657161537.bin")
env = Environment(
    conda_file="hybrid-search.yaml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
)

blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=model,
    environment=env,
    code_configuration=CodeConfiguration(
        code=".", scoring_script="scoring/chat.py"
    ),
    instance_type="Standard_F4s_v2",
    instance_count=1,
    request_settings=OnlineRequestSettings(request_timeout_ms=70000)
)

try: 
    ml_client.online_endpoints.get(name=endpoint_name)
except Exception as e:
    print(e)    
    ml_client.online_endpoints.begin_create_or_update(endpoint)
    
        
if ml_client.online_endpoints.get(name=endpoint_name):
    state = ml_client.online_endpoints.get(name=endpoint_name).provisioning_state
    print(state)
    if state == 'Succeeded':
        print(ml_client.online_endpoints.get(name=endpoint_name))
        try:
            ml_client.online_deployments.begin_create_or_update(blue_deployment)
        except Exception as e:
            print(e)

