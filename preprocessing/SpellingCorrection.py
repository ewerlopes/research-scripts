# -*- coding: UTF-8 -*-
# # Autor: Ewerton Lopes
# # Data: 24/09/2014
# # Replacing repeating characters (as done by replaceReptCharacter fuction in Pre_Util.py
# # is actually an extreme form of spelling correction. This script takes on the less extreme 
# # case of correcting minor spelling issues using Enchant— a spelling correction API. This script
# # was built from "Python text processing with NLTK" book, page 36.
# # Dependencies: 'Enchant' <http://www.abisource.com/projects/enchant/>
# #               'aspell' (a good open source spellchecker and dictionary) <http://aspell.net/>
# #               'pyenchant' library <http://www.rfk.id.au/software/pyenchant/>.

import enchant
from nltk.metrics import edit_distance


class SpellingReplacer(object):
    """This class uses the replace() method to check Enchant and see whether the word is 
    valid (by the English spelling point of view) or not. If not, we will look up suggested 
    alternatives and return the best match using nltk.metrics.edit_distance()"""
    
    def __init__(self, dict_name='en', max_dist=2):
        self.spell_dict = enchant.Dict(dict_name)
        self.max_dist = 2


    def replace(self, word):
        if self.spell_dict.check(word):
            return word
        suggestions = self.spell_dict.suggest(word)
        if suggestions and edit_distance(word, suggestions[0]) <= self.max_dist:
            return suggestions[0]
        else:
            return word

#USAGE:
# replacer = SpellingReplacer()
# print replacer.replace('cookbok')
