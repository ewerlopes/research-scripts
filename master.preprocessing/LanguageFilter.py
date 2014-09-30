# -*- coding: UTF-8 -*-

# # Autor: Ewerton Lopes
# # Data: 27/05/2014
# # This script contains functions for detecting the language of a specific sentence or tweet.


from nltk.corpus import stopwords  # stopwords resource for detecting language
from nltk import wordpunct_tokenize  # function to split up our words


def get_language_likelihood(input_text):
    """Return a dictionary of languages and their likelihood of being the 
    natural language of the input text. This estimated likelihood is computed by taken into 
    account the amount of words in the input_text that is in the intersection between different
    language stopwords sets
    """
    #lowercase input_text
    input_text = input_text.lower()
    #tokenizing
    input_words = wordpunct_tokenize(input_text)
 
    language_likelihood = {}
    
    #for each different language stopword set described in stopword._fileids:
    # get the number of words in the intersection between the input_words and the given stopword set.
    for language in stopwords._fileids:
        language_likelihood[language] = len(set(input_words) & set(stopwords.words(language)))
 
    return language_likelihood
# end of the function 
 
def get_language(input_text):
    """Return the most likely language of the given text
    """
    likelihoods = get_language_likelihood(input_text)
    return sorted(likelihoods, key=likelihoods.get, reverse=True)[0]
# end of the function

#USAGE: get_language('Put your text here')