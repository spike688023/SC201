"""
File: validEmailAddress_2.py
Name: 
----------------------------
Please construct your own feature vectors
and try to surpass the accuracy achieved by
Jerry's feature vector in validEmailAddress.py.
feature1:  '@' in the str and only one
feature2:  length of local-part greather 2  and the first char and the last char is not '.'
feature3:  local-part exist and local-part is alnum and ".." not in the str
feature4:  some strings after '@'
feature5:  There is '.' after '@' but not the first char
feature6:  There is no white space
feature7:  ".." in the str
feature8:  "\\" in the str
feature9:  length of local-part exist and the firest char and the last char is '"'
feature10:  Length > 10

Accuracy of your model: 1.0
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
	[0.2],
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
        print("{0}:{1}: {2}".format( index+1, mask_quoted_text(maybe_email), score) )

    print("Accuracy of this model: {0} ".format( round( pass_case/len(maybe_email_list) , 16)) )

def feature_extractor(maybe_email):
    """
    :param maybe_email: str, the string to be processed
    :return: list, feature vector with value 0's and 1's
    """
    feature_vector = [0] * len(WEIGHT)
    
    maybe_email = mask_quoted_text(maybe_email)

    for i in range(len(feature_vector)):
        # '@' in the str and only one
        if i == 0:
            feature_vector[i] = 1 if '@' in maybe_email and len(maybe_email.split('@')) == 2 else 0
        # length of local-part greather 2  and the first char and the last char is not '.'
        elif i == 1:
            if feature_vector[0]:
                feature_vector[i] = 1 if  len(maybe_email.split('@')[0]) >= 2 and '.' != maybe_email.split('@')[0][0] and '.' != maybe_email.split('@')[0][-1]  else 0
        # local-part exist and local-part is alnum and ".." not in the str
        elif i == 2: 
            if feature_vector[0]:
                feature_vector[i] = 1 if maybe_email.split('@')[0] and ".." not in maybe_email.split('@')[0] and maybe_email.split('@')[0].isalnum() else 0
        # some strings after '@'
        elif i == 3:
            if feature_vector[0]:
                feature_vector[i] = 1 if len("".join(maybe_email.split('@')[1]))  else 0
        # There is '.' after '@' but not the first char
        elif i == 4:
            if feature_vector[0]:
                feature_vector[i] = 1 if '.' in  "".join(maybe_email.split('@')[1:])  else 0

        # There is no white space
        elif i == 5:
            if feature_vector[0]:
                feature_vector[i] = 1 if ' ' not in maybe_email else 0
        # ".." in the str
        elif i == 6:
                feature_vector[i] = 1 if ".." in maybe_email else 0
        # "\\" in the str
        elif i == 7:
                feature_vector[i] = 1 if '\\' in maybe_email else 0
        # length of local-part exist and the firest char and the last char is '"'
        elif i == 8:
            if feature_vector[0]:
                feature_vector[i] = 1 if maybe_email.split('@')[0] and '"' == maybe_email.split('@')[0][0] and '"' == maybe_email.split('@')[0][-1]  else 0
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

def mask_quoted_text(email):
    def replace_match(m):
        content = m.group(1)  # 提取雙引號中的內容
        # 檢查內容是否僅由字母和數字組成
        if content.isalnum():  # 只包含字母和數字
            return '\\'  # 替換為\\
        else:
            return 'x' * len(content)  # 否則替換為 x 字符

    return re.sub(r'"([^"]*)"', replace_match, email)


if __name__ == '__main__':
    main()
