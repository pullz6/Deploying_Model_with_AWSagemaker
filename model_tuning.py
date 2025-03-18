#Importing our classes and their functions 
from data_cleaning import * 

#Utility for sklearn 
from sklearn.model_selection import RandomizedSearchCV

#Importing the models from Sklearn
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split


df = cleaning_df()
X = df.drop(['wrist','time','date','gyro_x','gyro_y','gyro_z'],axis=1)
y = df['activity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) 


# Define the parameter grid for XGBoost
param_dist = {
    'n_estimators': np.arange(50, 500, 50),
    'max_depth': np.arange(3, 10),
    'learning_rate': np.linspace(0.01, 0.3, 10),
    'subsample': np.linspace(0.5, 1, 5),
    'colsample_bytree': np.linspace(0.3, 1, 5),
}

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', seed=123)
random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, 
                                   n_iter=20, scoring='r2', cv=5, n_jobs=-1)
random_search.fit(X_train, y_train)

print("Best params for XGBoost:", random_search.best_params_)