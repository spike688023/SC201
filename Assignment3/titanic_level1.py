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
		first_line, lines = file.readlines()[0], file.readlines()[1:]

		for line in file:
    		line = line.strip()
			person_data = line.split(',')
    		ans = []

			# start from column Pclass
			if mode == 'train':
				start = 2
			else:
				start = 1

        	# Name : index 3,4
        	# skip NaN case in index start+4 (Age) and index start+10 (Embarked)
			if person_data[start+4] == "" or person_data[start+10] == "":
					continue

            # add Survived info for train 
			if mode == 'train':
				data.setdefault("Survived", []).append( int(person_data[1]) )

			# Train data
            # 0          ,1       ,2     ,3 4 ,5  ,6  ,7    ,8    ,9     ,10  ,11   ,12
            # PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
            #
			# Test data
            # 0          ,1     ,2 3 ,4  ,5  ,6    ,7    ,8     ,9   ,10   ,11
            # PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
			for i in range(len(person_data)):
				if i == start:
					# Pclass, normalize from 0 to 1
				    data.setdefault("Pclass", []).append( int(person_data[i])-1) / (3-1) )
				elif i == start+3:
					# Sex
					if data_lst[i] == 'male':
				        data.setdefault("Sex", []).append( 1 )
					else:
				        data.setdefault("Sex", []).append( 0 )
				elif i == start+4:
					# Age
					if data_lst[i]:
						ans.append( (float(data_lst[i]) - 0.42) / (80-0.42) )
						ans.append( ((float(data_lst[i]) - 0.42) / (80-0.42)) **2 )
					else:
						ans.append( (29.699 - 0.42) / (80-0.42) )
						ans.append( ((29.699 - 0.42) / (80-0.42)) **2 )
				elif i == start+5:
					# SibSp, normalize from 0 to 1
				    data.setdefault("SibSp", []).append( (int(person_data[i]) - 0) / 8 )
				elif i == start+6:
					# Parch, normalize from 0 to 1
				    data.setdefault("SibSp", []).append( (int(person_data[i]) - 0) / 6 )
				elif i == start+8:
					# Fare
					if person_data[i]:
						ans.append((float(person_data[i]) - 0) / 512.3292)
						ans.append(((float(person_data[i]) - 0) / 512.3292) **2)
					else:
						ans.append( 32.2/512.3292)
						ans.append( (32.2/512.3292) **2)
			if mode == 'train':
				return ans, y
	return data


def one_hot_encoding(data: dict, feature: str):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: dict[str, list], remove the feature column and add its one-hot encoding features
	"""
	############################
	#                          #
	#          TODO:           #
	#                          #
	############################
	return data


def normalize(data: dict):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:return data: dict[str, list], key is the column name, value is its normalized data
	"""
	############################
	#                          #
	#          TODO:           #
	#                          #
	############################
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
	# TODO:
	# Step 3 : Feature Extract
	# TODO:
	# Step 4 : Update weights
	# TODO:
	return weights
