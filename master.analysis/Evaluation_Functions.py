# -*- coding: UTF-8 -*-
# # Autor: Ewerton Lopes
# # Data: 24/09/2014
# # This script contains functions for performing evaluation procedures, such as cross validation 
# # and model selection using scikit-learn for machine learning python module.

from sklearn.cross_validation import KFold
from sklearn import metrics
from scipy.stats import sem
from tabulate import tabulate
import numpy as np
import pylab as pl


def evaluate_cross_validation(clf, X, y, K):
    """Function that takes a classifier and perform the K-fold cross-validation (CV) over
    the specified x and y array as well as the specified K value. The metrics used for CV are precision, recall and F1"""    
    
    # initializing the vectors for storing the scores obtained for each fold
    k_train_precision = np.zeros(K)
    k_train_recall = np.zeros(K)
    k_train_f1 = np.zeros(K)
    k_test_precision = np.zeros(K)
    k_test_recall = np.zeros(K)
    k_test_f1 = np.zeros(K)

    #getting the number of samples using the number of elements in the target vector. Note that it is the
    #same of using the number of elements in X given that they have the same number.
    n_samples = len(y)
        
    # create KFold cross validation
    cv = KFold(n_samples, K, shuffle=True, random_state=0)
    
    #Tuple for keeping the Best F1-based classifier got from the K partitions
    best_clf = (clf, 0.0) 
    
    print '[+] Training the classifier on multiple partitions...'
            
    #----- ITERATE OVER THE K folds -----#
    # NOTE: by scikit-learn default, the method 'cross_val_score(clf, X, y, cv=k_fold)' returns values from the 
    #       'score method' of the estimator (clf), which in case of MultinomialNB is the accuracy. Since we are interested
    #       in other metrics, we call metric functions 'manually' by iterating over the K-fold partitions defined by cv.
    for j, (train_indexes, test_indexes) in enumerate(cv):
        # fitting the classifier in the corresponding fold for obtaining the corresponding 
        # score measurements on train_indexes and test_indexes sets
        clf.fit([X[k] for k in train_indexes], y[train_indexes])
        
        #----- USING PRECISION, RECALL AND F1 AS MEASUREMENTS OF PERFORMANCE (for training set) -----#
        train_data = [X[k] for k in train_indexes]
        y_train_pred = clf.predict(train_data) #getting predicted results for computing scores
        k_train_precision[j] = metrics.precision_score(y[train_indexes], y_train_pred, pos_label='pos')
        k_train_recall[j] = metrics.recall_score(y[train_indexes], y_train_pred, pos_label='pos')
        k_train_f1[j] = metrics.f1_score(y[train_indexes], y_train_pred, pos_label='pos')
        
        #----- USING PRECISION, RECALL AND F1 AS MEASUREMENTS OF PERFORMANCE (for testing set) -----# 
        y_test = [X[k] for k in test_indexes] #reassemble the test_indexes set given the indexes returned via enumerate(cv) in 'test_indexes' variable
        y_test_pred = clf.predict(y_test) #getting predicted results for computing scores
        k_test_precision[j] = metrics.precision_score(y[test_indexes],y_test_pred, pos_label='pos')
        k_test_recall[j] = metrics.recall_score(y[test_indexes],y_test_pred, pos_label='pos')
        k_test_f1[j] = metrics.f1_score(y[test_indexes],y_test_pred, pos_label='pos')
        
        #refreshing the best F1-based classifier (Final model)
        if (best_clf[1] < k_test_f1[j]):
            best_clf = (clf,k_test_f1[j],y[test_indexes],y_test_pred)
        
            
    #----- Storing the mean of the K fold scores (for Testing set) -----#
    train_precision_mean_score = np.mean(k_train_precision)
    train_recall_mean_score = np.mean(k_train_recall)
    train_f1_mean_score = np.mean(k_train_f1)
    # Storing the standard deviation of the F1 score (common practice)
    train_f1_sem = sem(k_train_f1)
    #----- Storing the mean of the K fold scores (for Training set) -----#
    test_precision_mean_score = np.mean(k_test_precision)
    test_recall_mean_score = np.mean(k_test_recall)
    test_f1_mean_score = np.mean(k_test_f1)
    # Storing the standard deviation of the F1 score (common practice)
    test_f1_sem = sem(k_test_f1)
    
    #-----printing results ---#
    print "\n\n### RESULTS FOR THE " + str(K)+"-FOLD CROSS-VALIDATION TEST ###"
    
    print '----- TRAINING SCORES -----'
    #creating a table header for visualization purposes
    header = ['Metric/Folds'] +  ['#'+ str(x) for x in range(1,11)] + ['Best Fold'] + ['Mean/Std_Dev']

    #creating table instance for presenting measurements on training data
    table = [['Precision'] + k_train_precision.tolist() + [k_train_precision.argmax(axis=0)+1] + [train_precision_mean_score], \
            ['Recall'] + k_train_recall.tolist() + [k_train_recall.argmax(axis=0)+1] + [train_recall_mean_score], \
            ['F1'] + k_train_f1.tolist() + [k_train_f1.argmax(axis=0)+1] + [train_f1_mean_score]]
    
    #printing the table in the console
    print tabulate(table, header, numalign="right",floatfmt=".3f")
    
    print #just a blank line space :)
    
    print '----- TESTING SCORES -----'
    #creating table for presenting measurements on testing data
    table = [['Precision'] + k_test_precision.tolist() + [k_test_precision.argmax(axis=0)+1] + [test_precision_mean_score], \
            ['Recall'] + k_test_recall.tolist() + [k_test_recall.argmax(axis=0)+1] + [test_recall_mean_score], \
            ['F1'] + k_test_f1.tolist() + [k_test_f1.argmax(axis=0)+1] + [test_f1_mean_score]]
    
    #printing the table in the console
    print tabulate(table,header, numalign="right",floatfmt=".3f")
    
    #returns the classifier who achieved the best F1 measure as well as the training and testing partition
    return best_clf
    
###End of the function###

def train_and_evaluate(clf, data, targets):
    """Function that train the classifier (model represented by clf parameter) in the entire training set and evaluate the accuracy
        in the training and in the testing sets. It will also print a classification report
        (precision and recall on every class) and the corresponding confusion matrix."""

   
    print '[+] Using the classifier setup for performing Cross-validation...'
    #performing 10-Fold cross-validation and getting the best F1-based classifier considering all partitions
    best_clf_info = evaluate_cross_validation(clf,data,np.array(targets),10)
    
    y_test = best_clf_info[2] #retrieving y_test data set from best_clf_info
    y_pred = best_clf_info[3] #retrieving y_pred (predicted test data target by the classifier) from best_clf_info

    #printing sklearn classification report
    print "\n[+] Exclusive classification report for the best classifier..."
    print metrics.classification_report(y_test, y_pred)

    #computing confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    print "[+] Confusion Matrix:"
    print (cm)        #print in the console

    # Show confusion matrix in a separate window
    pl.matshow(cm)
    pl.title('Confusion matrix')
    pl.colorbar()
    pl.ylabel('True label')
    pl.xlabel('Predicted label')
    pl.show()
###End of the function###


def model_selection(X, y, clf, param_values, param_name, K):
    """
        This function will train the algorithm (clf parameter) with a list of values, each time obtaining
        an accuracy score calculated by performing k-fold cros-validation on the training instances.
        After that, it will plot the training and testing scores as a function of the parameter values.
    """
    # initialize training and testing scores with zeros
    train_precision_score = np.zeros(len(param_values))
    train_recall_score = np.zeros(len(param_values))
    train_f1_score = np.zeros(len(param_values))
    train_f1_sem = np.zeros(len(param_values)) #standard error of the mean
    test_precision_score = np.zeros(len(param_values))
    test_recall_score = np.zeros(len(param_values))
    test_f1_score = np.zeros(len(param_values))
    test_f1_sem = np.zeros(len(param_values)) #standard error of the mean

    n_samples = len(y)

    print 'Initializing MODEL SELECTION PROCESS - Iterating over the different parameter values (10-folding)...'
    # iterate over the different parameter values
    for i, param_value in enumerate(param_values):

        print param_name, ' = ', param_value
        
        # set classifier parameters
        clf.set_params(**{param_name:param_value})
        
        # initialize the K scores obtained for each fold
        k_train_precision = np.zeros(K)
        k_train_recall = np.zeros(K)
        k_train_f1 = np.zeros(K)
        k_test_precision = np.zeros(K)
        k_test_recall = np.zeros(K)
        k_test_f1 = np.zeros(K)

        
        # create KFold cross validation
        cv = KFold(n_samples, K, shuffle=True, random_state=0)
        
        # iterate over the K folds
        for j, (train, test) in enumerate(cv):
            # fit the classifier in the corresponding fold
            # and obtain the corresponding accuracy scores on train and test sets
            clf.fit([X[k] for k in train], y[train])


            #using precision, recall and f1 as a measurements of performance
            y_train_pred = clf.predict([X[k] for k in train])
            k_train_precision[j] = metrics.precision_score(y[train], y_train_pred, pos_label='pos')
            k_train_recall[j] = metrics.recall_score(y[train], y_train_pred, pos_label='pos')
            k_train_f1[j] = metrics.f1_score(y[train], y_train_pred, pos_label='pos')
            
            y_test_pred = clf.predict([X[k] for k in test])
            k_test_precision[j] = metrics.precision_score(y[test],y_test_pred, pos_label='pos')
            k_test_recall[j] = metrics.recall_score(y[test],y_test_pred, pos_label='pos')
            k_test_f1[j] = metrics.f1_score(y[test],y_test_pred, pos_label='pos')
            

            #------ USING ACCURACY AS A MEASURE OF PERFORMANCE-------------#
            ##k_train_scores[j] = clf.score([X[k] for k in train], y[train])
            ##k_test_scores[j] = clf.score([X[k] for k in test], y[test])
            
        # store the mean of the K fold scores
        train_precision_score[i] = np.mean(k_train_precision)
        train_recall_score[i] = np.mean(k_train_recall)
        train_f1_score[i] = np.mean(k_train_f1)
        train_f1_sem[i] = sem(k_train_f1)
        test_precision_score[i] = np.mean(k_test_precision)
        test_recall_score[i] = np.mean(k_test_recall)
        test_f1_score[i] = np.mean(k_test_f1)
        test_f1_sem[i] = sem(k_test_f1)
       
    # plot the training and testing scores in a log scale
    #pl.semilogx(param_values, train_precision_score, alpha=0.4, lw=2, c='b')
    #pl.semilogx(param_values, train_recall_score, alpha=0.4, lw=2, c='g')
    pl.semilogx(param_values, train_f1_score, alpha=0.4, lw=2, c='r', label='F1_train')
    #pl.semilogx(param_values, train_f1_sem, alpha=0.4, ls= '-', lw=2, c='g')
    #pl.semilogx(param_values, test_precision_score, alpha=0.4, lw=2)
    #pl.semilogx(param_values, test_recall_score, alpha=0.4, lw=2)
    pl.semilogx(param_values, test_f1_score, alpha=0.4, lw=2, c='b', label='F1_test')
    #pl.semilogx(param_values, test_f1_sem, alpha=0.4, ls= '-', lw=2, c='y')

    pl.title('MultinomialNB - freq, bigram, normalization')
    pl.xlabel(param_name + " values")
    pl.ylabel("Cross validation mean")
    pl.grid(True)
    pl.legend(loc="lower right")

    pl.show()

    # return the training and testing scores on each parameter value
    return train_precision_score, train_recall_score, train_f1_score, \
           test_precision_score,test_recall_score,test_f1_score, train_f1_sem, \
           test_f1_sem
    