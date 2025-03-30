FROM continuumio/miniconda3

# Install MLflow & Dependencies
RUN pip install mlflow boto3

# Set the entrypoint
ENTRYPOINT ["mlflow", "models", "serve", "-m", "/opt/ml/model", "-p", "8080"]
