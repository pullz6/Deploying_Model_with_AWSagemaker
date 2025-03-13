#Importing our classes and their functions
from data_ingestion import * 

#Import libraries required
import mlflow.data
import pandas as pd
from mlflow.data.pandas_dataset import PandasDataset


dataset_source_url = "https://www.kaggle.com/datasets/yasserh/kinematics-motion-data/data"
raw_data = loading_df()

dataset = mlflow.data.from_pandas(
    raw_data, source=dataset_source_url, name="Kinematics to see phone activity", targets="activity"
)
