#Importing Libraries
import numpy as np
import pandas as pd 
from numpy import mean, std

#Utility for sklearn 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold, cross_val_score

#Importing the models from Sklearn
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB


# get a stacking ensemble of models
def get_stacking():
	# define the base models
	level0 = list()
	level0.append(('lr', LinearRegression()))
	level0.append(('bayes', GaussianNB()))
    level0.append(('xgb',xg.XGBRegressor(objective='reg:squarederror', n_estimators=10, seed=123)))
	# define meta learner model
	level1 = LinearRegression()
	# define the stacking ensemble
	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
	return model
 
# get a list of models to evaluate
def get_models():
	models = dict()
	models['lr'] = LinearRegression()
	models['bayes'] = GaussianNB()
	models['stacking'] = get_stacking()
    models['xgb'] = xg.XGBRegressor(objective='reg:squarederror', n_estimators=10, seed=123)
	return models
 
def evaluate_model(model, X, y):
    # Define cross-validation method (Repeated K-Fold for regression)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # Evaluate model using cross-validation with negative RMSE scoring
    scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')
    return scores
 