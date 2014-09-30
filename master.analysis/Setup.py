# Author: Ewerton Lopes <ewerlopes@gmail.com>
# Date: May 26th 2014
# This script contains main classes for building classifier's setup 

from tabulate import tabulate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from Evaluation_Functions import train_and_evaluate, evaluate_cross_validation, model_selection
from Feature_Extractor import process_raw_dataset
import numpy as np

 

def buildClassifiers(data_set_name, sep = " :&: ", tweet_pos = 1, target_pos=2):
    """This function defines the classifier and then performs training and evaluation on it"""
    

    #reading the data set for extracting processed data and targets
    data, targets = process_raw_dataset(data_set_name) 
       
    #Classifier 1 - Multinomial Naive Bayes
    clf1 = Pipeline([
        ('vect', CountVectorizer()),
        ('nb', MultinomialNB())
        ])
    
    train_and_evaluate(clf1, data,targets)
    

def main():
    """'Main' function for the whole project"""
    buildClassifiers('tuarob_dataset.txt')

if __name__== '__main__':
    main()
    

