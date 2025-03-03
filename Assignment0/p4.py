class Lexicon:
    def __init__(self):
        """Initialize your data structure here."""
        self.words = set()  
        self.len_dict = dict() 

    def add_word(self, word):
        """
        :param word: str, the word that needs to be stored into your data structure.
        ------------------------------------------------------------------------
        Add the word into your data structure. The word will never be a duplicate.
        """
        if len(word) >= 1:
            if word.islower() and word.isalpha():
                self.words.add(word) 
                self.len_dict.setdefault(len(word),set())
                self.len_dict[len(word)].add(word)
            else:
                print("The letters must be lowercase.")
        else:
            print("Length of word must greater than 1")

    def search(self, word):
        """
        :param word: str, the word to be checked if it is in your data structure.
        :return: bool, if the word is in your data structure or not.
        ------------------------------------------------------------------------
        Search through your data structure to see if the word is in or not
        """
        # check word
        for i in range(len(word)):
            if word[i] == "." :
                 continue
            if not (word[i].islower() and word[i].isalpha()) :
                return False

        if word in self.words:
            return True
        # len is the same
        elif len(word) in self.len_dict.keys() :
            
            # get letter index
            letter_index = list()
            for i in range(len(word)):
                if word[i].isalpha():
                    letter_index.append(i)

            for i in range(len(word)):
                # skip .  case
                if word[i] == "." :
                    continue
                # loop all words with the same len
                match_or_not = False
                match_case_list = list()
                for item in self.len_dict[len(word)] :
                    if word[i] == item[i]:
                        match_or_not = True
                        match_case_list.append(item)
                
                if not match_or_not :
                    return False

                # check the rest letter
                for item2 in match_case_list :
                    final_check_score = 0
                    for index in letter_index :
                        if word[index] != item2[index]:
                            final_check_score += -1
                    if final_check_score == 0 :
                        return True
                # not match at all
                return False
            # skip all the time
            return True
        # len is match
        else:
            return False


lex = Lexicon()

lex.add_word("apple")

print( "lex.search(\"apple\") : " + str(lex.search("apple") ) )
print( "lex.search(\"app\") : " + str(lex.search("app") ) )
print( "lex.search(\"app..\") : " + str(lex.search("app..") ) )
print( "lex.search(\"..p.e\") : " + str(lex.search("..p.e") ) )
print( "lex.search(\".....\") : " + str(lex.search(".....") ) )
print( "lex.search(\"...\") : " + str(lex.search("...") ) )
print( "lex.search(\"a....\") : " + str(lex.search("a....") ) )
print( "lex.search(\"a...e\") : " + str(lex.search("a...e") ) )
print( "lex.search(\"a...3\") : " + str(lex.search("a...3") ) )

