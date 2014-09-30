# Author: Ewerton Lopes <ewerlopes@gmail.com>
# Date: May 26th 2014
# License: Creative commons

from tabulate import tabulate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from Evaluation_Functions import train_and_evaluate, evaluate_cross_validation, model_selection
from Feature_Extractor import process_raw_dataset
import numpy as np

 

def buildClassifiers(data_set_name, sep = " :&: ", tweet_pos = 1, target_pos=2):

    X_train, y_train, X_test,y_test = process_raw_dataset(data_set_name) #reading the data set for extracting training and testing set
        
    #Classifier 1 - Multinomial Naive Bayes
    clf1 = Pipeline([
        ('vect', CountVectorizer()),
        ('nb', MultinomialNB())
        ])
    
    train_and_evaluate(clf1, X_train, X_test, y_train, y_test)
    

#USAGE
buildClassifiers('tuarob_dataset.txt')
    
    

