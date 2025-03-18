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
import lightgbm as lgb
from xgboost import XGBRegressor
import matplotlib.pylab as plt
from sklearn import preprocessing, linear_model, metrics, ensemble, svm
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint

TRAIN_FILE = 'boston_housing/train.csv'
TEST_FILE = 'boston_housing/test.csv'

random_forest_regressor_param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(10, 30),
    'min_samples_split': randint(5, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 0.5]
}

GBDT_param_dist = {
    'n_estimators': [50, 100, 200, 500],  # 樹的數量
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # 學習率
    'max_depth': [3, 4, 5, 6, 7],  # 樹的最大深度
    'subsample': [0.6, 0.8, 1.0],  # 每棵樹的樣本比例
    'min_samples_split': [2, 5, 10],  # 內部節點的最小樣本數
    'min_samples_leaf': [1, 3, 5, 10],  # 葉節點的最小樣本數
    'max_features': ['sqrt', 'log2', None],  # 每棵樹的特徵選擇
    'loss': ['squared_error', 'huber'],  # 損失函數（MSE 或 Huber）
}

EGB_param_dist = {
    'n_estimators': [50, 100, 200, 500],  # 樹的數量
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # 學習率
    'max_depth': [3, 4, 5, 6, 7, 8],  # 樹的最大深度
    'subsample': [0.6, 0.8, 1.0],  # 每棵樹使用的樣本比例
    'colsample_bytree': [0.6, 0.8, 1.0],  # 每棵樹使用的特徵比例
    'gamma': [0, 0.1, 0.2, 0.3],  # 節點分裂所需的最小損失減少
    'min_child_weight': [1, 3, 5, 10],  # 決定葉子節點中最小樣本權重和
    'reg_alpha': [0, 0.01, 0.1, 1],  # L1 正則化
    'reg_lambda': [0, 0.01, 0.1, 1]  # L2 正則化
}

LGBM_eparam_distributions = {
    "n_estimators": [50, 100, 200, 300, 500],
    "learning_rate": np.linspace(0.01, 0.3, 10),
    "max_depth": [3, 4, 5, 6, 7, -1],  # -1 代表無限制
    "num_leaves": [20, 31, 40, 50, 60],
    "subsample": np.linspace(0.6, 1.0, 5),
    "colsample_bytree": np.linspace(0.6, 1.0, 5),
    "reg_alpha": np.linspace(0, 1, 5),
    "reg_lambda": np.linspace(0, 1, 5)
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
	print("\n")
	h = ensemble.RandomForestRegressor(n_estimators=100, random_state=42)

	model_status(h, X_train, y_train, X_val, y_val, test_data, "random_forest_regressor.csv", "Random Forest Regressor")

	# find best parameter
	random_search = RandomizedSearchCV(ensemble.RandomForestRegressor(random_state=42),random_forest_regressor_param_dist, n_iter=20, cv=5, scoring='neg_root_mean_squared_error')
	random_search.fit(X_train, y_train)
	print("Best Parameters:", random_search.best_params_)

	# 使用 RandomizedSearchCV 找到的最佳參數
	best_params = random_search.best_params_

	# 直接餵給 RandomForestRegressor
	best_model = ensemble.RandomForestRegressor(**best_params, random_state=42)
	model_status(best_model, X_train, y_train, X_val, y_val, test_data, "random_forest_regressor_best.csv", "Random Forest Regressor")

	print("\n")

	#############################
	# Gradient Boosting Decision Tree, GBDT #
	#############################
	h = ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)

	model_status(h, X_train, y_train, X_val, y_val, test_data, "gradient_boosting_decision_tree.csv", "Gradient Boosting Decision Tree")

	# find best parameter
	random_search = RandomizedSearchCV(ensemble.GradientBoostingRegressor(random_state=42),GBDT_param_dist, n_iter=20, cv=5, scoring='neg_root_mean_squared_error')
	random_search.fit(X_train, y_train)
	print("Best Parameters:", random_search.best_params_)

	# 使用 RandomizedSearchCV 找到的最佳參數
	best_params = random_search.best_params_

	# 直接餵給 RandomForestRegressor
	best_model = ensemble.GradientBoostingRegressor(**best_params, random_state=42)
	model_status(best_model, X_train, y_train, X_val, y_val, test_data, "gradient_boosting_decision_tree_best.csv", "Gradient Boosting Decision Tree")

	print("\n")

	#############################
	# Extreme Gradient Boosting #
	#############################
	h = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)

	model_status(h, X_train, y_train, X_val, y_val, test_data, "extreme_gradient_boosting.csv", "Extreme Gradient Boosting")

	# find best parameter
	random_search = RandomizedSearchCV(XGBRegressor(random_state=42),EGB_param_dist, n_iter=20, cv=5, scoring='neg_root_mean_squared_error')
	random_search.fit(X_train, y_train)
	print("Best Parameters:", random_search.best_params_)

	# 使用 RandomizedSearchCV 找到的最佳參數
	best_params = random_search.best_params_

	best_model = XGBRegressor(**best_params, random_state=42)

	model_status(best_model, X_train, y_train, X_val, y_val, test_data, "extreme_gradient_boosting_best.csv", "Extreme Gradient Boosting")
	print("\n")


	#############################
	# Light Gradient Boosting Machine #
	#############################
	h = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, verbose=-1)

	model_status(h, X_train, y_train, X_val, y_val, test_data, "light_gradient_boosting_machine.csv", "Light Gradient Boosting Machine")

	# find best parameter
	random_search = RandomizedSearchCV(lgb.LGBMRegressor(random_state=42),LGBM_eparam_distributions, n_iter=20, cv=5, scoring='neg_root_mean_squared_error')
	random_search.fit(X_train, y_train)
	print("Best Parameters:", random_search.best_params_)

	# 使用 RandomizedSearchCV 找到的最佳參數
	best_params = random_search.best_params_

	best_model = lgb.LGBMRegressor(**best_params, random_state=42)
	model_status(best_model, X_train, y_train, X_val, y_val, test_data, "light_gradient_boosting_machine_best.csv", "Light Gradient Boosting Machine")

	print("\n")

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
