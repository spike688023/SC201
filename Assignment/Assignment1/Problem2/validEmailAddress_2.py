"""
File: validEmailAddress_2.py
Name: 
----------------------------
Please construct your own feature vectors
and try to surpass the accuracy achieved by
Jerry's feature vector in validEmailAddress.py.
feature1:  TODO:
feature2:  TODO:
feature3:  TODO:
feature4:  TODO:
feature5:  TODO:
feature6:  TODO:
feature7:  TODO:
feature8:  TODO:
feature9:  TODO:
feature10: TODO:

Accuracy of your model: TODO:
"""

WEIGHT = [                           # The weight vector selected by you
	[],                              # (Please fill in your own weights)
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[]
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
	:return: list, feature vector with value 0's and 1's
	"""
	feature_vector = [0] * len(WEIGHT)
	for i in range(len(feature_vector)):
		pass
		###################################
		#                                 #
		#              TODO:              #
		#                                 #
		###################################
	return feature_vector


def read_in_data():
	"""
	:return: list, containing strings that may be valid email addresses
	"""
	# TODO:
	pass


if __name__ == '__main__':
	main()
