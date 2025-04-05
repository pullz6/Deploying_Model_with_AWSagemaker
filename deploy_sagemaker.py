import mlflow.sagemaker as mfs
from mlflow.deployments import get_deploy_client
import boto3
import logging
from time import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG = {
    "DEPLOYMENT_NAME": "my-mlflow-model",
    "MODEL_URI": "s3://mlflow-sagemaker-eu-west-2-905418206632/Kinematic-prediction-model-5b36c82f44864e06a710/",
    "IMAGE_URI": "905418206632.dkr.ecr.eu-west-2.amazonaws.com/deploy/mflow_1:latest",
    "REGION": "eu-west-2",
    "INSTANCE_TYPE": "ml.t2.medium",
    "TIMEOUT": 1800,
    "ENABLE_CLEANUP": True
}

class SageMakerDeployer:
    def __init__(self, config):
        self.config = config
        self.sm_client = boto3.client('sagemaker', region_name=config["REGION"])
        self.deployment_client = get_deploy_client("sagemaker")

    def deploy_model(self):
        try:
            logger.info("🚀 Starting deployment with MLflow deployments API")
            
            # New deployment API
            self.deployment_client.create_deployment(
                name=self.config["DEPLOYMENT_NAME"],
                model_uri=self.config["MODEL_URI"],
                config={
                    "instance_type": self.config["INSTANCE_TYPE"],
                    "archive": False,
                    "image_url": self.config["IMAGE_URI"],
                    "synchronous": True,
                    "timeout_seconds": self.config["TIMEOUT"],
                    "region_name": self.config["REGION"]
                }
            )
            logger.info("✅ Deployment successful!")
            return True
        except Exception as e:
            logger.error(f"❌ Deployment failed: {str(e)}")
            return False

    # ... [keep the rest of your existing methods unchanged] ...

if __name__ == "__main__":
    deployer = SageMakerDeployer(CONFIG)
    deployer.run()