import mlflow.sagemaker as mfs
import boto3
from datetime import datetime, timedelta
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "DEPLOYMENT_NAME": "my-mlflow-model",
    "MODEL_URI": "s3://mlflow-sagemaker-eu-west-2-905418206632/Kinematic-prediction-model-5b36c82f44864e06a710/",
    "IMAGE_URI": "905418206632.dkr.ecr.eu-west-2.amazonaws.com/deploy/mflow_1:latest",
    "REGION": "eu-west-2",
    "INSTANCE_TYPE": "ml.t2.medium",
    "DEPLOY_TIMEOUT": 1800,  # 30 minutes
    "TEST_DURATION": 600,    # 10 minutes (set to 0 for immediate cleanup)
    "ENABLE_CLEANUP": True   # Set False to skip cleanup
}

class SageMakerDeployer:
    def __init__(self, config):
        self.config = config
        self.sm_client = boto3.client('sagemaker', region_name=config["REGION"])

    def deploy_model(self):
        """Deploy model to SageMaker endpoint"""
        try:
            logger.info(f"🚀 Deploying model to SageMaker (timeout: {self.config['DEPLOY_TIMEOUT']}s)")
            
            mfs.deploy(
                app_name=self.config["DEPLOYMENT_NAME"],
                model_uri=self.config["MODEL_URI"],
                image_url=self.config["IMAGE_URI"],
                region_name=self.config["REGION"],
                instance_type=self.config["INSTANCE_TYPE"],
                timeout_seconds=self.config["DEPLOY_TIMEOUT"],
                mode=mfs.DEPLOYMENT_MODE_REPLACE,
                synchronous=True
            )
            logger.info("✅ Deployment successful!")
            return True
        except Exception as e:
            logger.error(f"❌ Deployment failed: {str(e)}")
            return False

    def test_endpoint(self):
        """Placeholder for your test logic"""
        if self.config["TEST_DURATION"] <= 0:
            return
        
        logger.info(f"⏳ Running tests for {self.config['TEST_DURATION']} seconds...")
        
        # Example: Ping the endpoint (replace with your actual test logic)
        try:
            runtime = boto3.client('sagemaker-runtime', region_name=self.config["REGION"])
            start_time = time.time()
            
            while time.time() - start_time < self.config["TEST_DURATION"]:
                # Replace with your actual inference test
                time.sleep(60)  # Check every minute
                logger.info("🧪 Testing endpoint... (replace with your test logic)")
                
        except Exception as e:
            logger.error(f"⚠️ Test error: {str(e)}")

    def cleanup(self):
        """Delete SageMaker resources to stop charges"""
        if not self.config["ENABLE_CLEANUP"]:
            return
            
        logger.info("🧹 Cleaning up AWS resources...")
        
        endpoint_name = self.config["DEPLOYMENT_NAME"]
        try:
            # Delete endpoint
            self.sm_client.delete_endpoint(EndpointName=endpoint_name)
            logger.info(f"🗑️ Deleted endpoint: {endpoint_name}")
            
            # Delete endpoint config
            endpoint_config_name = self.sm_client.describe_endpoint(
                EndpointName=endpoint_name)["EndpointConfigName"]
            self.sm_client.delete_endpoint_config(
                EndpointConfigName=endpoint_config_name)
            logger.info(f"🗑️ Deleted endpoint config: {endpoint_config_name}")
            
            # Delete model
            model_name = endpoint_config_name  # SageMaker creates model with same name as config
            self.sm_client.delete_model(ModelName=model_name)
            logger.info(f"🗑️ Deleted model: {model_name}")
            
        except Exception as e:
            logger.error(f"⚠️ Cleanup error: {str(e)}")

    def run(self):
        """Execute full workflow"""
        if self.deploy_model():
            self.test_endpoint()
            if self.config["ENABLE_CLEANUP"]:
                self.cleanup()

if __name__ == "__main__":
    deployer = SageMakerDeployer(CONFIG)
    deployer.run()