"""
File: titanic_level1.py
Name: 
----------------------------------
This file builds a machine learning algorithm from scratch 
by Python. We'll be using 'with open' to read in dataset,
store data into a Python dict, and finally train the model and 
test it on kaggle website. This model is the most flexible among all
levels. You should do hyper-parameter tuning to find the best model.
"""

import math
from util import *
TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename: str, data: dict, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be processed
	:param data: an empty Python dictionary
	:param mode: str, indicating if it is training mode or testing mode
	:param training_data: dict[str: list], key is the column name, value is its data
						  (You will only use this when mode == 'Test')
	:return data: dict[str: list], key is the column name, value is its data
	"""
		
	with open(filename, 'r') as file:
		# first row
		# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
		next(file)
		#first_line, lines = file.readlines()[0], file.readlines()[1:]
		#lines =  file.readlines()[1:]

		for num, line in enumerate(file):
			line = line.strip()
			person_data = line.split(',')

			# start from column Pclass
			if mode == 'Train':
				start = 2
				# skip NaN case in index start+4 (Age) and index start+10 (Embarked)
				if person_data[start+4] == "" or person_data[start+10] == "":
						continue
            	# add Survived info for train 
				data.setdefault("Survived", []).append( int(person_data[1]) )
			else:
				Age_avg_in_Train_data = round( sum(training_data['Age'])/len(training_data['Age']), 3 )
				Fare_avg_in_Train_data = round( sum(training_data['Fare'])/len(training_data['Fare']), 3 )
				start = 1

			# Train data
            # 0          ,1       ,2     ,3 4 ,5  ,6  ,7    ,8    ,9     ,10  ,11   ,12
            # PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
            #
			# Test data
            # 0          ,1     ,2 3 ,4  ,5  ,6    ,7    ,8     ,9   ,10   ,11
            # PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
			for i in range(len(person_data)):
				if i == start:
					# Pclass
				    data.setdefault("Pclass", []).append( int(person_data[i]) )
				elif i == start+3:
					# Sex
					if person_data[i] == 'male':
						data.setdefault("Sex", []).append( 1 )
					else:
						data.setdefault("Sex", []).append( 0 )
				elif i == start+4:
					# Age
					if person_data[i]: 
						data.setdefault("Age", []).append( float(person_data[i]) )
					# give avg data from Train when data is empty in Test  
					else:
						data.setdefault("Age", []).append( Age_avg_in_Train_data )
				elif i == start+5:
					# SibSp
					data.setdefault("SibSp", []).append( int(person_data[i]) )
				elif i == start+6:
					# Parch
					data.setdefault("Parch", []).append( int(person_data[i]) )
				elif i == start+8:
					# Fare
					if person_data[i]: 
						data.setdefault("Fare", []).append( float(person_data[i]) )
					else:
						data.setdefault("Fare", []).append( Fare_avg_in_Train_data )
				elif i == start+10:
					# Embarked
					if person_data[i] == 'S':
						data.setdefault("Embarked", []).append( 0 )
					elif person_data[i] == 'C':
						data.setdefault("Embarked", []).append( 1 )
					elif person_data[i] == 'Q':
						data.setdefault("Embarked", []).append( 2 )
	return data


def one_hot_encoding(data: dict, feature: str):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: dict[str, list], remove the feature column and add its one-hot encoding features
	"""

	values_set = set( data[feature] )
	min_index_from_values_set = min(values_set)
	# add new feature
	for item in range(len(data[feature])):
			for index in values_set:
					if data[feature][item] == index :
							data.setdefault(feature + '_' + str(index - min_index_from_values_set), []).append( 1 )
					else:
							data.setdefault(feature + '_' + str(index - min_index_from_values_set), []).append( 0 )
	# pop unsed feature
	data.pop(feature)
	return data


def normalize(data: dict):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:return data: dict[str, list], key is the column name, value is its normalized data
	"""
	# normalize : x - min/(max - min)
	for key in data.keys():
			min_value = min(data[key])
			divisor = max(data[key]) - min_value
			for index, value in enumerate(data[key]):
				 data[key][index] = (value - min_value) / divisor
	return data


def learnPredictor(inputs: dict, labels: list, degree: int, num_epochs: int, alpha: float):
	"""
	:param inputs: dict[str, list], key is the column name, value is its data
	:param labels: list[int], indicating the true label for each data
	:param degree: int, degree of polynomial features
	:param num_epochs: int, the number of epochs for training
	:param alpha: float, known as step size or learning rate
	:return weights: dict[str, float], feature name and its weight
	"""
	# Step 1 : Initialize weights
	weights = {}  # feature => weight
	keys = list(inputs.keys())
	if degree == 1:
		for i in range(len(keys)):
			weights[keys[i]] = 0
	elif degree == 2:
		for i in range(len(keys)):
			weights[keys[i]] = 0
		for i in range(len(keys)):
			for j in range(i, len(keys)):
				weights[keys[i] + keys[j]] = 0

	# Step 2 : Start training
	for epoch in range(num_epochs):

		phi_x = {}
		def sigmoid(k):
			return 1 / (1 + math.exp(-k))

		for index in range(len(labels)):
			# Step 3 : Feature Extract
			if degree == 1:
				for i in range(len(keys)):
					phi_x[keys[i]] = inputs[keys[i]][index]
			elif degree == 2:
				for i in range(len(keys)):
					phi_x[keys[i]] = inputs[keys[i]][index]
				# Squared Feature and Cross Feature
				for i in range(len(keys)):
					for j in range(i, len(keys)):
						phi_x[keys[i] + keys[j]] = inputs[keys[i]][index] * inputs[keys[j]][index]

			# Step 4 : Update weights
			h = sigmoid(dotProduct(weights, phi_x))
			scale = -alpha*(h-labels[index])
			increment(weights, scale, phi_x)

       #def predictor(review):
       #    """
       #    @param review: str, it's either x from trainExamples or validationExamples
       #    @return: int, prediction y' of review. It's either +1 or -1
       #    """
       #    phi_vector = featureExtractor(review)
       #    score = dotProduct(phi_vector, weights)
       #    return 1 if score >= 0 else -1

	return weights
