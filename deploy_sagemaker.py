import mlflow.sagemaker as mfs
from mlflow.deployments import get_deploy_client
import boto3
import logging
import time

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
    "TEST_DURATION": 300,
    "ENABLE_CLEANUP": True,
    "DEPLOY_MODE": "replace"
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
        except self.sm_client.exceptions.ClientError as e:
            if "Could not find endpoint" in str(e):
                return False
            raise

    def cleanup(self):
        """Clean up existing endpoint resources"""
        endpoint_name = self.config["DEPLOYMENT_NAME"]
        try:
            # Get endpoint config first
            endpoint_config = self.sm_client.describe_endpoint(
                EndpointName=endpoint_name
            )["EndpointConfigName"]
            
            # Get model name (usually same as endpoint config name)
            model_name = endpoint_config
            
            logger.info(f"🧹 Cleaning up existing resources for {endpoint_name}...")
            
            # Delete endpoint
            self.sm_client.delete_endpoint(EndpointName=endpoint_name)
            logger.info(f"🗑️ Deleted endpoint: {endpoint_name}")
            
            # Delete endpoint config
            self.sm_client.delete_endpoint_config(EndpointConfigName=endpoint_config)
            logger.info(f"🗑️ Deleted endpoint config: {endpoint_config}")
            
            # Delete model
            self.sm_client.delete_model(ModelName=model_name)
            logger.info(f"🗑️ Deleted model: {model_name}")
            
            return True
        except Exception as e:
            logger.error(f"⚠️ Cleanup failed: {str(e)}")
            return False

    def deploy_model(self):
        """Handle deployment with replace mode"""
        try:
            if self._endpoint_exists() and self.config["DEPLOY_MODE"] != "replace":
                raise ValueError(
                    f"Endpoint {self.config['DEPLOYMENT_NAME']} exists. "
                    "Set DEPLOY_MODE='replace' to update it."
                )

            logger.info(f"🚀 Deploying in {self.config['DEPLOY_MODE']} mode...")

            # Force clean existing resources if in replace mode
            if self.config["DEPLOY_MODE"] == "replace" and self._endpoint_exists():
                if not self.cleanup():
                    raise RuntimeError("Failed to clean up existing endpoint")
                time.sleep(10)  # Wait for resources to delete

            self.deployment_client.create_deployment(
                name=self.config["DEPLOYMENT_NAME"],
                model_uri=self.config["MODEL_URI"],
                config={
                    "instance_type": self.config["INSTANCE_TYPE"],
                    "image_url": self.config["IMAGE_URI"],
                    "region_name": self.config["REGION"],
                    "timeout_seconds": self.config["TIMEOUT"],
                    "mode": self.config["DEPLOY_MODE"],
                    "archive": False,
                    "synchronous": True
                }
            )
            logger.info("✅ Deployment successful!")
            return True

        except Exception as e:
            logger.error(f"❌ Deployment failed: {str(e)}")
            return False

    def test_endpoint(self):
        """Test the deployed endpoint"""
        if self.config["TEST_DURATION"] <= 0:
            return

        logger.info(f"🧪 Testing endpoint for {self.config['TEST_DURATION']} seconds...")
        start_time = time.time()
        
        try:
            runtime = boto3.client('sagemaker-runtime', region_name=self.config["REGION"])
            
            while time.time() - start_time < self.config["TEST_DURATION"]:
                # Add your actual test logic here
                time.sleep(30)
                logger.info("Testing endpoint...")
                
        except Exception as e:
            logger.error(f"⚠️ Test error: {str(e)}")

    def run(self):
        """Execute full workflow"""
        try:
            if self.deploy_model():
                self.test_endpoint()
                if self.config["ENABLE_CLEANUP"]:
                    self.cleanup()
        except Exception as e:
            logger.error(f"🔥 Workflow failed: {str(e)}")
            raise

if __name__ == "__main__":
    deployer = SageMakerDeployer(CONFIG)
    deployer.run()