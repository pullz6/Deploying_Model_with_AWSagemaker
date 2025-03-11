#Importing Libraries
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn.pipeline import make_pipeline

def model(): 
    model = LinearRegression()
    return model