#Importing Libraries
import numpy as np
import pandas as pd 

#Utility for sklearn 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

#Importing the models from Sklearn
import xgboost as xgb
from sklearn.linear_model import LinearRegression

def model(): 
    lr_model = LinearRegression()
    xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=42)
    model = make_pipeline(lr_model, xgb_model)
    return model