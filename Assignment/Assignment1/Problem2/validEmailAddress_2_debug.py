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

import numpy as np
import re

WEIGHT = [                           # The weight vector selected by you
	[0.4],                           # (see assignment handout for more details)
	[0.4],
	[0.2],
	[-0.25],
	[0.9],
	[-0.65],
	[-0.3],
	[-0.95],
	[0.1],
	[-0.7]
]

DATA_FILE = 'is_valid_email.txt'     # This is the file name to be processed


def main():
    maybe_email_list = read_in_data()
    weight_vector = np.array( WEIGHT )
    
    weight_vector = weight_vector.T
    pass_case = 0
    for index, maybe_email in enumerate(maybe_email_list):
        feature_vector =  np.array( feature_extractor(maybe_email) )
        score = np.dot( weight_vector, feature_vector)[0]

        if index < 13 and score <= 0:
            pass_case += 1
        elif index >= 13 and score > 0:
            pass_case += 1
        print("{0}:{1}: {2}".format( index+1, replace_quoted_text_no_quotes(maybe_email), score) )

    print("pass case : {0}".format( pass_case ))
    print("Accuracy of this model: {0} ".format( round( pass_case/len(maybe_email_list) , 16)) )

def feature_extractor(maybe_email):
    """
    :param maybe_email: str, the string to be processed
    :return: list, feature vector with value 0's and 1's
    """
    feature_vector = [0] * len(WEIGHT)
    
    maybe_email = replace_quoted_text_no_quotes(maybe_email)

    for i in range(len(feature_vector)):
        # '@' in the str
        if i == 0:
            feature_vector[i] = 1 if '@' in maybe_email else 0
        # No '.' before '@' 
        elif i == 1:
            if feature_vector[0]:
                feature_vector[i] = 1 if  len(maybe_email.split('@')[0]) >= 2 and '.' != maybe_email.split('@')[0][0] and '.' != maybe_email.split('@')[0][-1]  else 0
        # some strings before '@'
        elif i == 2: 
            if feature_vector[0]:
                feature_vector[i] = 1 if maybe_email.split('@')[0] and ".." not in maybe_email.split('@')[0] else 0
        # some strings after '@'
        elif i == 3:
            if feature_vector[0]:
                feature_vector[i] = 1 if len("".join(maybe_email.split('@')[1]))  else 0
        # There is '.' after '@'
        elif i == 4:
            if feature_vector[0]:
                feature_vector[i] = 1 if '.' in  "".join(maybe_email.split('@')[1:])  else 0

        # There is white space
        elif i == 5:
            if feature_vector[0]:
                feature_vector[i] = 1 if ' ' not in maybe_email else 0
        # Ends with '.com'
        elif i == 6:
                feature_vector[i] = 1 if ".." in maybe_email else 0
        # Ends with '.edu'
        elif i == 7:
                feature_vector[i] = 1 if '\\' in maybe_email else 0
        # Ends with '.tw'
        elif i == 8:
                feature_vector[i] = 0
        # Length > 10
        elif i == 9:
            if feature_vector[0]:
                feature_vector[i] = 1 if len(maybe_email) > 10 else 0
    return feature_vector


def read_in_data():
    """
    :return: list, containing strings that may be valid email addresses
    """
    mail_list = list()
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        for line in file:  
            mail_list.append( line.strip() )

    return mail_list

def replace_quoted_text_no_quotes(s):
    return re.sub(r'"([^"]*)"', lambda m: 'x' * len(m.group(1)), s)


if __name__ == '__main__':
    main()
