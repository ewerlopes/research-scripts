# -*- coding: UTF-8 -*-
# # Autor: Ewerton Lopes
# # Data: 23/09/2014
# # This script contains functions for preprocessing tweets.

import re
import string
from RepeatedCharRemover import RepeatReplacer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


def processTweet(tweet, rid_puct=True, rm_stopwords=False, stopwords_set=set([])):
    """This functions performs preprocess steps on each new tweet, returning
    it at the end"""
       
    # Convert to lower case
    tweet = tweet.lower()
    # Convert www.* or https?://* to URL
    tweet = re.sub(r'(www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)|(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)', 'URL', tweet)
    # Convert @username to AT_USER
    tweet = re.sub(r'@[^\s]+', 'AT_USER', tweet)
    # remove additional white spaces
    tweet = re.sub(r'[\s]+', ' ', tweet)
    # trim
    tweet = tweet.strip('\'"')
    #replacing dollar
    tweet = re.sub(r'\$+', 'dollar ', tweet)
     
    if rid_puct:
        #tokenization that get rid of the punctuation
        tokenizer = RegexpTokenizer("#?[\w']+")
    else:
        #tokenization that keeps punctuation
        tokenizer = RegexpTokenizer('\s+', gaps=True)
        
    #breaking the tweet into a list of words for processing each word separately.
    tokens = tokenizer.tokenize(tweet)
   
    # The following for look uses 'tokens' (list of tweet's words) for processing each word separately in order to achieve the following goals:
    # 1. Replacing any token with number but letters, i.e., if a word has a number on it, it is ignored (e.g.: h1n1),
    #    else it is replaced by the word "number"
    # 2. Replace occurrences of repeated characters in word, like: 'cooooolll' and 'greaaat'.
    
    list_of_words = []  # auxiliar list of words
    rep_char_replacer = RepeatReplacer() #a repeated character remover for restoring a misspelled word into a standard English word.
    for t in tokens:
        matchNumbers = re.search("(\d+)", t)
        matchLetters = re.search("[a-zA-Z]+", t)
        
        if (matchNumbers and not matchLetters): #if the token is just a number 
            list_of_words.append("number")
        elif (t[0]=='#'): #if it is a hashtag
            list_of_words.append(t)
        else:
            # check if the token has a repeated character and it's not a normal English word (tokens like h1n1 are kept intact)
            list_of_words.append(rep_char_replacer.replace(t))             
    
    #Removing stopwords if rm_stopwords == True.
    if (rm_stopwords and stopwords_set): #uses the set of stopwords provided by the user
        list_of_words = [word for word in list_of_words if word not in stopwords_set]
    elif (rm_stopwords and stopwords_set==set([])): #uses the default set of stopwords (English stopwords)
        english_stops = set(stopwords.words('english'))
        list_of_words = [word for word in list_of_words if word not in english_stops]
        

    tweet = string.join(list_of_words, sep=' ')  #resembling tweet
    return tweet
# end_of_the_function

def saveToFile(file_name, data):
    """This functions saves data into a specific file"""
    try:
        f = open(file_name, 'a')
        f.write(data)
        f.close()
        print 'Data successfully processed and stored!'
    except Exception, e:
        print 'Ops! Error on saving data to ' + file_name + 'file...'
        print str(e)
# end_of_the_function


# USAGE: 
# print processTweet("I THINK THE THOUGHT OF STUDYING FOR CHEM AND BIO MCQ IS A DAMN PAIN IN THE ASSSS")
# #Read the tweets one by one and process it
# fp = open('../data/tuarob_dataset.txt', 'r')
# lines = fp.readlines()
# from SpellingCorrection import SpellingReplacer
# spellingCorrector = SpellingReplacer()
# count = 0
# for line in lines:
#     line = line.split(' :&: ')
#     processedTweet = spellingCorrector.replace(line[1])
#     print '['+str(count)+']' + ' -- ' + processedTweet
#     count += 1
# #end loop
# fp.close()