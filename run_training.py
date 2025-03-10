#Importing our classes and their functions
from data_cleaning import * 
from model import * 
import mlflow
from mlflow.models import infer_signature

#Importing libraries
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from data_cleaning import * 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error 

print('starting the run')
    #mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    # Create a new MLflow Experiment
mlflow.set_experiment("Kinematics to see phone activity")

df = cleaning_df()
X = df.drop(['wrist','time','date','gyro_x','gyro_y','gyro_z'],axis=1)
y = df['activity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) 
model = model()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
rmse = mean_squared_error(y_test, predictions)
mse = mean_absolute_error(y_test, predictions)
print('mean_squared_error : ', mean_squared_error(y_test, predictions)) 
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))

with mlflow.start_run():
    # Log the hyperparameters
        #mlflow.log_params(params)

        # Log the loss metric
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MSE", mse)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Training a linear regression model to predict kinematic movement through phones")

        # Infer the model signature
        signature = infer_signature(X_train, model.predict(X_train))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="Kinematic model",
            signature=signature,
            input_example=X_train,
            registered_model_name="Model.v.0.1",
        ) 