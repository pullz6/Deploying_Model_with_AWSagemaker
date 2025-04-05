import mlflow.sagemaker as mfs
from mlflow.deployments import get_deploy_client
import boto3
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG = {
    "DEPLOYMENT_NAME": <ENTER>,
    "MODEL_URI": <ENTER>,
    "IMAGE_URI": <ENTER>,
    "REGION": <ENTER>,
    "INSTANCE_TYPE":<ENTER>,
    "TIMEOUT": 1800,  # 30 minutes
    "TEST_DURATION": 300,  # 5 minutes
    "ENABLE_CLEANUP": True
}

class SageMakerDeployer:
    def __init__(self, config):
        self.config = config
        self.sm_client = boto3.client('sagemaker', region_name=config["REGION"])
        self.deployment_client = get_deploy_client("sagemaker")

    def deploy_model(self):
        """Deploy model using MLflow's deployment client"""
        try:
            logger.info(f"🚀 Starting deployment to {self.config['DEPLOYMENT_NAME']}")
            
            self.deployment_client.create_deployment(
                name=self.config["DEPLOYMENT_NAME"],
                model_uri=self.config["MODEL_URI"],
                config={
                    "instance_type": self.config["INSTANCE_TYPE"],
                    "image_url": self.config["IMAGE_URI"],
                    "region_name": self.config["REGION"],
                    "timeout_seconds": self.config["TIMEOUT"],
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
                time.sleep(30)  # Check every 30 seconds
                logger.info("Endpoint test in progress...")
                
        except Exception as e:
            logger.error(f"⚠️ Test error: {str(e)}")

    def cleanup(self):
        """Clean up AWS resources"""
        if not self.config["ENABLE_CLEANUP"]:
            return

        endpoint_name = self.config["DEPLOYMENT_NAME"]
        try:
            # Get associated resources
            endpoint_config = self.sm_client.describe_endpoint(
                EndpointName=endpoint_name)["EndpointConfigName"]
            model_name = endpoint_config  # Typically matches endpoint config name

            # Delete resources
            self.sm_client.delete_endpoint(EndpointName=endpoint_name)
            logger.info(f"🗑️ Deleted endpoint: {endpoint_name}")
            
            self.sm_client.delete_endpoint_config(EndpointConfigName=endpoint_config)
            logger.info(f"🗑️ Deleted endpoint config: {endpoint_config}")
            
            self.sm_client.delete_model(ModelName=model_name)
            logger.info(f"🗑️ Deleted model: {model_name}")
            
        except Exception as e:
            logger.error(f"⚠️ Cleanup error: {str(e)}")

    def run(self):
        """Execute full deployment workflow"""
        if self.deploy_model():
            self.test_endpoint()
            self.cleanup()

if __name__ == "__main__":
    deployer = SageMakerDeployer(CONFIG)
    deployer.run()
