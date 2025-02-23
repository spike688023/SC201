"""
File: validEmailAddress_2.py
Name: 
----------------------------
Please construct your own feature vectors
and try to surpass the accuracy achieved by
Jerry's feature vector in validEmailAddress.py.
feature1:  '@' in the str and only once
feature2:  No '.' or '-' show up at the first char befor '@'
feature3:  char must be in [a-zA-Z0-9.-_] before '@'
feature4:  No '.' show up next to '@' and at the end
feature5:  char must be in [a-zA-Z0-9.-_] after '@'
feature6:  There is no white space
feature7:  Ends with '.com'
feature8:  Ends with '.edu'
feature9:  Ends with '.tw'
feature10:  Length > 10

Accuracy of your model: TODO:
"""

WEIGHT = [                           # The weight vector selected by you
	[-1],                              # (Please fill in your own weights)
	[],
	[],
	[],
	[],
	[],
	[0.1],
	[0.1],
	[0.1],
	[0.1]
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

def is_valid_string(s):
    valid_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_"
    
    for char in s:
        if char not in valid_chars:
            return False
    return True

if __name__ == '__main__':
	main()
