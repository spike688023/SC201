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
	# extract rm as feature to do training
	x_train = np.array(train_data.rm).reshape(-1, 1)

	'''
	h = linear_model.LinearRegression()
	regressor = h.fit(x_train, y)
	print(sum(regressor.predict(x_train))/len(x_train))
	'''
	# y = w*x+b
	w, b, c = 0, 0, 0.6
	alpha = 0.01
	num_epoch = 20
	history = []
	for num in range(num_epoch):
		total = 0
		for i in range(len(x_train)):
			x, label = x_train[i], y[i]
			h = w*x+b
			loss = (h - label)**2
			total += loss
			# G.D
			#w = w - alpha*2*(h - label)*x
			#b = b - alpha*2*(h - label)*1
			w = w - alpha*2*(h - label)*x*(sign(h-label)-c)**2
			b = b - alpha*2*(h - label)*1*(sign(h-label)-c)**2
		history.append(total/len(train_data))
	#plt.plot(history)
	#plt.show()
	
	predictions = []
	for x in x_train:
		predictions.append( w*x+b )
	print(sum(predictions)/len(x_train) )

	# RMS error
	print( metrics.mean_squared_error(predictions, y)**0.5 )

def sign(x):
	if x > 0:
		return 1
	elif x < 0:
		return -1
	else:
		return 0


if __name__ == '__main__':
	main()
