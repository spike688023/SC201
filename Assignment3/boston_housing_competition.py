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
from sklearn import preprocessing, linear_model
import numpy as np
import matplotlib.pylab as plt
from sklearn import metrics

TRAIN_FILE = 'boston_housing/train.csv'
TEST_FILE = 'boston_housing/test.csv'

def main():
	train_data = pd.read_csv(TRAIN_FILE)
	y = train_data.pop("medv")

	#############################
	# Degree 1 Polynomial Model #
	#############################
	h = linear_model.LinearRegression()
	regressor_degree_1 = h.fit(train_data, y)
	predictions = h.predict(train_data)

	# RMS error
	print( "Linear regression of poly degree 1 : ", metrics.mean_squared_error(predictions, y)**0.5 )

	#############################
	# Degree 2 Polynomial Model #
	#############################
	poly = preprocessing.PolynomialFeatures(degree=2)  # 轉換成二次多項式特徵
	X_train_poly = poly.fit_transform(train_data)
	regressor_degree_2 = h.fit(X_train_poly, y)
	predictions = h.predict(X_train_poly)

	# RMS error
	print( "Linear regression of poly degree 2 : ", metrics.mean_squared_error(predictions, y)**0.5 )

def sign(x):
	if x > 0:
		return 1
	elif x < 0:
		return -1
	else:
		return 0


if __name__ == '__main__':
	main()
