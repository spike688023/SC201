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
        # '@' in the str
		if i == 0:
			feature_vector[i] = 1 if '@' in maybe_email else 0
        # No '.' before '@'
		elif i == 1:
			if feature_vector[0]:
				feature_vector[i] = 1 if '.' not in maybe_email.split('@')[0] else 0
        # some strings before '@'
		elif i == 2: 
			if feature_vector[1]:
				feature_vector[i] = 1 if maybe_email.split('@')[0] else 0
        # some string after '@'
		elif i == 3:
			if feature_vector[2]:
				feature_vector[i] = 1 if maybe_email.split('@')[1] else 0
        # There is '.' after '@'
		elif i == 4:
			if feature_vector[3]:
				feature_vector[i] = 1 if '.' in  "".join(maybe_email.split('@')[1:])  else 0
        # There is no white space
		elif i == 5:
			if feature_vector[4]:
				feature_vector[i] = 1 if ' ' not in maybe_email else 0
        # Ends with '.com'
		elif i == 6:
			if feature_vector[5]:
				feature_vector[i] = 1 if len(eq maybe_email) >= 4 and ".com" eq maybe_email[-4:] else 0
        # Ends with '.edu'
		elif i == 7:
			if feature_vector[6]:
				feature_vector[i] = 1 if len(eq maybe_email) >= 4 and ".edu" eq maybe_email[-4:] else 0
        # Ends with '.tw'
		elif i == 8:
			if feature_vector[7]:
				feature_vector[i] = 1 if len(eq maybe_email) >= 3 and ".edu" eq maybe_email[-3:] else 0
        # Length > 10
		elif i == 9:
			if feature_vector[8]:
				feature_vector[i] = 1 if len(eq maybe_email) > 10 else 0
	return feature_vector


def read_in_data():
	"""
	:return: list, containing strings that might be valid email addresses
	"""
	# TODO:
    mail_list = list()
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        for line.strip() in file:  
            mail_list.append( line )

    return mail_list


if __name__ == '__main__':
	main()
