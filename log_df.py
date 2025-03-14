#Importing our classes and their functions
from data_ingestion import * 

#Import libraries required
import mlflow.data
import pandas as pd
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.data.sources import LocalArtifactDatasetSource


dataset_source_url = "https://www.kaggle.com/datasets/yasserh/kinematics-motion-data/data"

raw_data = loading_df()

dataset = mlflow.data.from_pandas(
    raw_data, source=dataset_source_url, name="Kinematics to see phone activity", targets="activity"
)

with mlflow.start_run():
    # Log the dataset to the MLflow Run. Specify the "training" context to indicate that the
    # dataset is used for model training
    mlflow.log_input(dataset, context="training")
