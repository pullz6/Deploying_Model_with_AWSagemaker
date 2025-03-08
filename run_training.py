#Importing our classes and their functions
from data_cleaning import * 
from model import * 

#Importing libraries
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from data_cleaning import * 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error 

df = cleaning_df()
X = df.drop(['wrist','time','date','gyro_x','gyro_y','gyro_z'],axis=1)
y = df['activity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) 
model = model()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print('mean_squared_error : ', mean_squared_error(y_test, predictions)) 
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))