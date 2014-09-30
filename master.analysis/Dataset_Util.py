# -*- coding: UTF-8 -*-
# # Autor: Ewerton Lopes
# # Data: 24/09/2014
# # This script contains functions for dealing with data sets. It assumes that the data set are
# # stored in a plain text file (*.txt) and it is passed as a list of sentences where each item
# # represents a tweet.

def train_test_split(data, target, SPLIT_PERC=0.66):
    """Partitions the data set available (data and target) into training and testing set. By default,
    the function assumes the amount of 2/3 for training and 1/3 for testing."""
    
    print "Partitioning data...."
    split_size = int(len(data) * SPLIT_PERC)
    X_train = data[:split_size]
    X_test =  data[split_size:]
    y_train = target[:split_size]
    y_test = target[split_size:]
    print "Done!"
    print

    return X_train, X_test, y_train, y_test