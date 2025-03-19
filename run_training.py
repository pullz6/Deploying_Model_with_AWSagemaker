#Importing our classes and their functions
from data_cleaning import * 
from model_stack import * 
from model_tuning import * 


#Importing libraries
import numpy as np
import pandas as pd
import mlflow
from mlflow.models import infer_signature

#Importing sklearn utilities 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error 

def split_data(): 
    """This function is used to split the data"""
    df = cleaning_df()
    X = df.drop(['wrist','time','date','gyro_x','gyro_y','gyro_z'],axis=1)
    y = df['activity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) 
    return  X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data()
print('starting the run')
    #mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    # Create a new MLflow Experiment
mlflow.set_experiment("Kinematics to see phone activity")
  
# get the models to evaluate
models = get_models()
    
# evaluate the models and store results
results, names = list(), list()
    
for name, model in models.items():
    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Training a linear regression model to predict kinematic movement through phones")
    with mlflow.start_run(nested=True):
        scores = evaluate_model(model, X_train, y_train)
        model.fit(X_train,y_train)
        signature = infer_signature(X_train, model.predict(X_train))
            
        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="Kinematic model "+str(name),
            signature=signature,
            input_example=X_train,
            registered_model_name=name,
        )
            
        print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
        mlflow.log_metric(str(name)+" Mean", mean(scores))
        mlflow.log_metric(str(name)+" STD", std(scores))

        
        

        

        

        