# -*- coding: UTF-8 -*-
# # Autor: Ewerton Lopes
# # Data: 25/09/2014
# # This class is intended to remove repeated characters of a word in order to transform it into a
# # standard English word. It was built from inspiration gathered from "Python text processing with 
# # NLTK" book, page 35.
# # Dependencies: nltk module.


import re
from nltk.corpus import words

class RepeatReplacer(object):
    """This function replaces more than one character, e.g., 'coool' becomes 'cool'. During
    its process it uses a corpus of English words (from nltk module) to check if the word is, 
    actually, a standard English word."""
    
    def __init__(self):
        #regular expressions
        self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
        self.repl = r'\1\2\3'

    def replace(self, word):
        #checking if the word is a 'standard English word'. If so, carry it on without changes.
        if word in words.words('en'): #words.words('en') is a large list of English words with over 200,000 elements
            return word
            
        repl_word = self.repeat_regexp.sub(self.repl, word)
    
        if repl_word != word:
            return self.replace(repl_word) #Recursion
        else:
            return repl_word

#USAGE
##replacer = RepeatReplacer()
##print replacer.replace('assss')
