import json
import boto3
import mlflow
import sagemaker
import pandas as pd
import mlflow.sagemaker
from mlflow.deployments import get_deploy_client
import logging
from mlflow.tracking import MlflowClient

print("Tracking URI:", mlflow.get_tracking_uri()) 

client = MlflowClient()
models = client.search_registered_models()

for model in models:
    print(f"Model Name: {model.name}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set AWS region
try:
    session = boto3.Session()
    region = session.region_name or "eu-north-1"  # Default to eu-north-1 if region is not set
    logger.info(f"Using AWS region: {region}")
except Exception as e:
    logger.error(f"Error retrieving AWS region: {e}")
    raise

# Specify IAM Role for SageMaker
role = "<ADD YOUR ARN>"  # Replace with your actual SageMaker execution role ARN
logger.info(f"Using IAM Role: {role}")

# MLflow Tracking URI
tracking_uri = "file:USER/Desktop/Projects/MLOps/Deploy_Model/mlruns"  # Ensure this is reachable
mlflow.set_tracking_uri(tracking_uri)

# Model details
endpoint_name = "Kinematic-prediction"
model_name = "xgb" 
model_version = 13
model_uri = f"models:/{model_name}/{model_version}"
image_uri = "<ADD YOUR ARN>"

# ✅ Check if model exists in MLflow
try:
    client = MlflowClient()
    model_versions = client.get_latest_versions(model_name)
    print("Starting-------")
    print(model_versions)
    if any(m.version == model_version for m in model_versions):
        logger.info(f"Model {model_uri} exists. Proceeding with deployment.")
    else:
        raise ValueError(f"Model {model_uri} not found in MLflow Model Registry.")
except Exception as e:
    logger.error(f"Error retrieving model {model_uri}: {e}")
    raise

# Deployment configuration
config = {
    "execution_role_arn": role,
    "image_url": image_uri,
    "instance_type": "ml.m5.xlarge",
    "instance_count": 1,
    "region_name": region
}

# Initialize SageMaker deployment client
try:
    deploy_client = get_deploy_client("sagemaker")
    logger.info("Successfully connected to SageMaker deployment client.")
except Exception as e:
    logger.error(f"Error initializing SageMaker deployment client: {e}")
    raise

# Deploy the model
try:
    response = deploy_client.create_deployment(
        name=endpoint_name,
        model_uri=model_uri,
        flavor="python_function",
        config=config
    )
    logger.info(f"Model deployed successfully! Endpoint Name: {endpoint_name}")
except Exception as e:
    logger.error(f"Error deploying model to SageMaker: {e}")
    raise
