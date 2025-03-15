#Importing our classes and their functions 
from data_cleaning import * 

#Import libraries required
import mlflow.data
import pandas as pd
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.data.sources import LocalArtifactDatasetSource

#/Users/pulsaragunawardhana/.cache/kagglehub/datasets/yasserh/kinematics-motion-data/versions/1

dataset_source_url = "https://www.kaggle.com/datasets/yasserh/kinematics-motion-data/data"

raw_data = cleaning_df()

dataset = mlflow.data.from_pandas(
    raw_data, source=dataset_source_url, name="Kinematics - dataset", targets="activity"
)

experiment = mlflow.get_experiment_by_name("Kinematics to see phone activity")
with mlflow.start_run(experiment_id=experiment.experiment_id):
    # Log the dataset to the MLflow Run. Specify the "training" context to indicate that the
    # dataset is used for model training
    mlflow.log_input(dataset, context="training")
