"""
File: titanic_level2.py
Name: 
----------------------------------
This file builds a machine learning algorithm by pandas and sklearn libraries.
We'll be using pandas to read in dataset, store data into a DataFrame,
standardize the data by sklearn, and finally train the model and
test it on kaggle website. Hyper-parameters tuning are not required due to its
high level of abstraction, which makes it easier to use but less flexible.
You should find a good model that surpasses 77% test accuracy on kaggle.
"""

import math
import pandas as pd
from sklearn import preprocessing, linear_model

TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be read into pandas
	:param mode: str, indicating the mode we are using (either Train or Test)
	:param training_data: DataFrame, a 2D data structure that looks like an excel worksheet
						  (You will only use this when mode == 'Test')
	:return: Tuple(data, labels), if the mode is 'Train'; or return data, if the mode is 'Test'
	"""
	data = pd.read_csv(filename)
	labels = None

	# drop unused data
	data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin' ])

	# transform str to int/float
	data[["Pclass", "Age", "SibSp", "Parch"]] = data[["Pclass", "Age", "SibSp", "Parch"]].apply(pd.to_numeric, errors="coerce")
	data["Sex"] = data["Sex"].replace({"male": 1, "female": 0})
	data["Fare"] = pd.to_numeric(data["Fare"])
	data["Embarked"] = data["Embarked"].replace({"S": 0, "C": 1, "Q": 2})

	if mode == 'Train':
		data = data.dropna(subset=['Age','Embarked'])
		labels = data.pop('Survived')
		return data, labels
	elif mode == 'Test':
		data["Age"] = data["Age"].fillna(training_data["Age"].mean().round(3))
		data["Fare"] = data["Fare"].fillna(training_data["Fare"].mean().round(3))
		return data


def one_hot_encoding(data, feature):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: DataFrame, remove the feature column and add its one-hot encoding features
	"""
	data[feature] -= 0 if (data[feature] == 0).any() else 1
	data = pd.get_dummies(data, columns=[feature] )
	return data


def standardization(data, mode='Train'):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param mode: str, indicating the mode we are using (either Train or Test)
	:return data: DataFrame, standardized features
	"""
	scaler = preprocessing.StandardScaler()
	data = scaler.fit_transform(data)
	return data


def main():
	"""
	You should call data_preprocess(), one_hot_encoding(), and
	standardization() on your training data. You should see ~80% accuracy on degree1;
	~83% on degree2; ~87% on degree3.
	Please write down the accuracy for degree1, 2, and 3 respectively below
	(rounding accuracies to 8 decimal places)
	TODO: real accuracy on degree1 -> ______________________
	TODO: real accuracy on degree2 -> ______________________
	TODO: real accuracy on degree3 -> ______________________
	"""
	train_data, Y = data_preprocess(TRAIN_FILE)

	train_data = one_hot_encoding(train_data, 'Sex')
	train_data = one_hot_encoding(train_data, 'Pclass')
	train_data = one_hot_encoding(train_data, 'Embarked')

	# Normalization / Standardization
	normalizer = preprocessing.StandardScaler()
	X_train = normalizer.fit_transform(train_data)

	#############################
	# Degree 1 Polynomial Model #
	#############################
	h = linear_model.LogisticRegression()
	classifier = h.fit(X_train, Y)
	train_acc = classifier.score(X_train, Y)
	print(train_acc)

	#############################
	# Degree 2 Polynomial Model #
	#############################
	poly_phi_extractor = preprocessing.PolynomialFeatures(degree=2)
	X_train_poly = poly_phi_extractor.fit_transform(X_train)
	print(X_train)
	print(X_train_poly)
	classifier_poly = h.fit(X_train_poly, Y)
	train_acc = classifier_poly.score(X_train_poly, Y)
	print(train_acc)

	# Test dataset
########test_data = data_preprocess(TEST_FILE, mode='Test')
########X_test = normalizer.transform(test_data)
########X_test_poly = poly_phi_extractor.transform(X_test)
########predictions_poly = classifier_poly.predict(X_test_poly)
	#out_file(predictions_poly, "pandas_sklearn_degree2.csv")


if __name__ == '__main__':
	main()
