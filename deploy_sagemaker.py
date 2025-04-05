import mlflow.sagemaker as mfs
from mlflow.deployments import get_deploy_client
import boto3
import logging
import time
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG = {
    "DEPLOYMENT_NAME": "my-mlflow-model",
    "MODEL_URI": "s3://mlflow-sagemaker-eu-west-2-905418206632/Kinematic-prediction-model-5b36c82f44864e06a710/",
    "IMAGE_URI": "905418206632.dkr.ecr.eu-west-2.amazonaws.com/deploy/mflow_1:latest",
    "REGION": "eu-west-2",
    "INSTANCE_TYPE": "ml.t2.medium",
    "TIMEOUT": 3600,  # 1 hour timeout
    "DEPLOY_MODE": "replace"  # Critical for updates
}

class SageMakerDeployer:
    def __init__(self, config):
        self.config = config
        self.sm_client = boto3.client('sagemaker', region_name=config["REGION"])
        self.deployment_client = get_deploy_client("sagemaker")

    def _endpoint_exists(self):
        """Check if endpoint already exists"""
        try:
            self.sm_client.describe_endpoint(EndpointName=self.config["DEPLOYMENT_NAME"])
            return True
        except ClientError as e:
            if "Could not find endpoint" in str(e):
                return False
            raise

    def _cleanup_existing(self):
        """Delete existing endpoint resources"""
        try:
            # Get associated resources first
            endpoint_info = self.sm_client.describe_endpoint(
                EndpointName=self.config["DEPLOYMENT_NAME"]
            )
            config_name = endpoint_info["EndpointConfigName"]
            
            # Delete in proper order
            self.sm_client.delete_endpoint(EndpointName=self.config["DEPLOYMENT_NAME"])
            self.sm_client.delete_endpoint_config(EndpointConfigName=config_name)
            
            # Model may already be deleted by SageMaker
            try:
                self.sm_client.delete_model(ModelName=config_name)
            except ClientError as e:
                if "Could not find model" not in str(e):
                    raise
            
            logger.info("♻️ Existing resources cleaned up")
            return True
        except Exception as e:
            logger.error(f"⚠️ Cleanup failed: {str(e)}")
            return False

    def deploy(self):
        """Handle deployment with replace mode"""
        try:
            # Check existing endpoint if in replace mode
            if self.config["DEPLOY_MODE"] == "replace" and self._endpoint_exists():
                if not self._cleanup_existing():
                    raise RuntimeError("Failed to clean existing endpoint")
                time.sleep(10)  # Wait for deletions to complete

            # Deployment configuration
            deploy_config = {
                "instance_type": self.config["INSTANCE_TYPE"],
                "image_url": self.config["IMAGE_URI"],
                "region_name": self.config["REGION"],
                "mode": self.config["DEPLOY_MODE"],
                "timeout_seconds": self.config["TIMEOUT"],
                "synchronous": True,
                "archive": False
            }

            logger.info(f"🚀 Deploying in {self.config['DEPLOY_MODE']} mode...")
            
            # Start deployment
            self.deployment_client.create_deployment(
                name=self.config["DEPLOYMENT_NAME"],
                model_uri=self.config["MODEL_URI"],
                flavor="python_function",
                config=deploy_config
            )
            
            logger.info("✅ Deployment successful!")
            return True

        except Exception as e:
            logger.error(f"❌ Deployment failed: {str(e)}")
            return False

if __name__ == "__main__":
    deployer = SageMakerDeployer(CONFIG)
    deployer.deploy()