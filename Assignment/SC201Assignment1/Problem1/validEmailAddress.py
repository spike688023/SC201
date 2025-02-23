"""
File: validEmailAddress.py
Name: 
----------------------------
This file shows what a feature vector is
and what a weight vector is for valid email 
address classifier. You will use a given 
weight vector to classify what is the percentage
of correct classification.

Accuracy of this model: TODO:
"""

WEIGHT = [                           # The weight vector selected by Jerry
	[0.4],                           # (see assignment handout for more details)
	[0.4],
	[0.2],
	[0.2],
	[0.9],
	[-0.65],
	[0.1],
	[0.1],
	[0.1],
	[-0.7]
]

DATA_FILE = 'is_valid_email.txt'     # This is the file name to be processed


def main():
	maybe_email_list = read_in_data()
	for maybe_email in maybe_email_list:
		feature_vector = feature_extractor(maybe_email)
		# TODO:


def feature_extractor(maybe_email):
	"""
	:param maybe_email: str, the string to be processed
	:return: list, feature vector with 10 values of 0's or 1's
	"""
	feature_vector = [0] * len(WEIGHT)
	for i in range(len(feature_vector)):
		if i == 0:
			feature_vector[i] = 1 if '@' in maybe_email else 0
		elif i == 1:
			if feature_vector[0]:
				feature_vector[i] = 1 if '.' not in maybe_email.split('@')[0] else 0
		###################################
		#                                 #
		#              TODO:              #
		#                                 #
		###################################
	return feature_vector


def read_in_data():
	"""
	:return: list, containing strings that might be valid email addresses
	"""
	# TODO:
	pass


if __name__ == '__main__':
	main()
