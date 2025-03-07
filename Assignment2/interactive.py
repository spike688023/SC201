"""
File: interactive.py
Name: 
------------------------
This file uses the function interactivePrompt
from util.py to predict the reviews input by 
users on Console. Remember to read the weights
and build a Dict[str: float]
"""
import util
import submission


def main():
    featureExtractor = submission.extractWordFeatures
    weights = dict()
    with open("weights", 'r') as file:
        for line in file:
            key, value = line.split()
            weights[key] = float(value)

    util.interactivePrompt(featureExtractor, weights)

if __name__ == '__main__':
	main()
