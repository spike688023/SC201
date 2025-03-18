"""
File: boston_housing_competition.py
Name: 
--------------------------------
This file demonstrates how to analyze boston
housing dataset. Students will upload their 
results to kaggle.com and compete with people
in class!

You are allowed to use pandas, sklearn, or build the
model from scratch! Go data scientists!
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
import lightgbm as lgb
from xgboost import XGBRegressor
import matplotlib.pylab as plt
from sklearn import preprocessing, linear_model, metrics, ensemble, svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

TRAIN_FILE = 'boston_housing/train.csv'
TEST_FILE = 'boston_housing/test.csv'

param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(10, 30),
    'min_samples_split': randint(5, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 0.5]
}


def main():
	train_data = pd.read_csv(TRAIN_FILE)
	test_data = pd.read_csv(TEST_FILE)
	y = train_data.pop("medv")

	X_train, X_val, y_train, y_val = train_test_split(train_data, y, test_size=0.8, random_state=42)

	print( "                                     train data,   validation data,      test data predictions" )
	#############################
	# Degree 1 Polynomial Model #
	#############################
	h = linear_model.LinearRegression()

	h.fit(X_train, y_train)

	predictions_train = h.predict(X_train)
	predictions_val = h.predict(X_val)
	predictions_test = h.predict(test_data)
	out_file_name = "linear_regression_degree_1.csv"

	# RMS error
	print_prediction_status( "Linear regression of poly degree 1", predictions_train, y_train, predictions_val, y_val, out_file_name)
	out_file(test_data.ID, predictions_test, out_file_name)

	#############################
	# Degree 2 Polynomial Model #
	#############################
	poly = preprocessing.PolynomialFeatures(degree=2)  # 轉換成二次多項式特徵
	X_train_poly = poly.fit_transform(X_train)
	X_val_poly = poly.fit_transform(X_val)
	X_test_poly = poly.fit_transform(test_data)

	h.fit(X_train_poly, y_train)

	predictions_train = h.predict(X_train_poly)
	predictions_val = h.predict(X_val_poly)
	predictions_test = h.predict(X_test_poly)
	out_file_name = "linear_regression_degree_2.csv"

	# RMS error
	print_prediction_status( "Linear regression of poly degree 2", predictions_train, y_train, predictions_val, y_val, out_file_name)
	out_file(test_data.ID, predictions_test, out_file_name)

	#############################
	# Random Forest Regressor  #
	#############################
	h = ensemble.RandomForestRegressor(n_estimators=100, random_state=42)

	model_status(h, X_train, y_train, X_val, y_val, test_data, "random_forest_regressor.csv", "Random Forest Regressor")

####h.fit(X_train, y_train)

####predictions_train = h.predict(X_train)
####predictions_val = h.predict(X_val)
####predictions_test = h.predict(test_data)
####out_file_name = "random_forest_regressor.csv"

##### RMS error
####print_prediction_status( "Random Forest Regressor", predictions_train, y_train, predictions_val, y_val, out_file_name)
####out_file(test_data.ID, predictions_test, out_file_name)



	# find best parameter
	random_search = RandomizedSearchCV(ensemble.RandomForestRegressor(random_state=42),param_dist, n_iter=20, cv=5, scoring='neg_root_mean_squared_error')
	random_search.fit(X_train, y_train)
	print("Best Parameters:", random_search.best_params_)

	# training again
	# 使用 RandomizedSearchCV 找到的最佳參數
	best_params = random_search.best_params_

	# 直接餵給 RandomForestRegressor
	best_model = ensemble.RandomForestRegressor(**best_params, random_state=42)

	model_status(best_model, X_train, y_train, X_val, y_val, test_data, "random_forest_regressor_best.csv", "Random Forest Regressor")

####best_model.fit(X_train, y_train)

##### 訓練最佳模型
####predictions_train = best_model.predict(X_train)
####predictions_val = best_model.predict(X_val)
####predictions_test = best_model.predict(test_data)
####out_file_name = "random_forest_regressor_best.csv"

##### RMS error
####print_prediction_status( "Random Forest Regressor", predictions_train, y_train, predictions_val, y_val, out_file_name)
####out_file(test_data.ID, predictions_test, out_file_name)

	#############################
	# Gradient Boosting Decision Tree, GBDT #
	#############################
	h = ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)

	h.fit(X_train, y_train)

	predictions_train = h.predict(X_train)
	predictions_val = h.predict(X_val)
	predictions_test = h.predict(test_data)
	out_file_name = "gradient_boosting_decision_tree.csv"

	# RMS error
	print_prediction_status( "Gradient Boosting Decision Tree ", predictions_train, y_train, predictions_val, y_val, out_file_name)
	out_file(test_data.ID, predictions_test, out_file_name)

	#############################
	# Support Vector Regression, SVR #
	#############################
	scaler_X = preprocessing.StandardScaler()
	scaler_y = preprocessing.StandardScaler()
	X_train_scaled = scaler_X.fit_transform(X_train)
	y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1, 1)).ravel()
	X_val_scaled = scaler_X.transform(X_val)
	X_test_scaled = scaler_X.transform(test_data)

	#  訓練 Support Vector Regression（SVR）
	h = svm.SVR(kernel='rbf', C=100, epsilon=0.1)
	h.fit(X_train_scaled, y_train_scaled)

	#  預測房價（需還原縮放）
	y_train_pred_scaled = h.predict(X_train_scaled)
	y_val_pred_scaled = h.predict(X_val_scaled)
	y_test_pred_scaled = h.predict(X_test_scaled)
	predictions_train = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
	predictions_val = scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).ravel()
	predictions_test = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()
	out_file_name = "support_vector_regression.csv"

	# RMS error
	print_prediction_status( "Support Vector Regression ", predictions_train, y_train, predictions_val, y_val, out_file_name)
	out_file(test_data.ID, predictions_test, out_file_name)

	#############################
	# Extreme Gradient Boosting #
	#############################
	h = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)

	h.fit(X_train, y_train)

	predictions_train = h.predict(X_train)
	predictions_val = h.predict(X_val)
	predictions_test = h.predict(test_data)
	out_file_name = "extreme_gradient_boosting.csv"

	# RMS error
	print_prediction_status( "Extreme Gradient Boosting ", predictions_train, y_train, predictions_val, y_val, out_file_name)
	out_file(test_data.ID, predictions_test, out_file_name)

	#############################
	# Light Gradient Boosting Machine #
	#############################
####h = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6,force_col_wise=True)

####h.fit(X_train, y_train)

####predictions_train = h.predict(X_train)
####predictions_val = h.predict(X_val)
####predictions_test = h.predict(test_data)
####out_file_name = "light_gradient_boosting_machine.csv"

####print_prediction_status( "Light Gradient Boosting Machine ", predictions_train, y_train, predictions_val, y_val, out_file_name)
####out_file(test_data.ID, predictions_test, out_file_name)

	#############################
	# Elastic Net Regularization #
	#############################
	h = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)

	h.fit(X_train, y_train)

	predictions_train = h.predict(X_train)
	predictions_val = h.predict(X_val)
	predictions_test = h.predict(test_data)
	out_file_name = "elastic_net_regularization.csv"

	# RMS error
	print_prediction_status( "Elastic Net Regularization ", predictions_train, y_train, predictions_val, y_val, out_file_name)
	out_file(test_data.ID, predictions_test, out_file_name)

def model_status( model, X_train, y_train, X_val, y_val, test_data, out_file_name, model_name ):
	model.fit(X_train, y_train)

	predictions_train = model.predict(X_train)
	predictions_val = model.predict(X_val)
	predictions_test = model.predict(test_data)

	# RMS error
	print_prediction_status( model_name, predictions_train, y_train, predictions_val, y_val, out_file_name)
	out_file(test_data.ID, predictions_test, out_file_name)


def print_prediction_status( model_str, predictions_train, y_train, predictions_val, y_val, out_file_name):
	print( "{0:<34} : {1:<10},   {2:<8},           {3:<8}".format(model_str, round( metrics.mean_squared_error(predictions_train, y_train)**0.5 ,8), round( metrics.mean_squared_error(predictions_val, y_val)**0.5,8) , out_file_name ))

def out_file(ID, predictions, filename):
	"""
	: param predictions: numpy.array, a list-like data structure that stores 0's and 1's
	: param filename: str, the filename you would like to write the results to
	"""

	with open(filename, 'w') as out:
		out.write('ID,medv\n')
		for index in range(len(predictions)):
			out.write(str(ID[index])+','+str(predictions[index])+'\n')


def sign(x):
	if x > 0:
		return 1
	elif x < 0:
		return -1
	else:
		return 0



if __name__ == '__main__':
	main()
