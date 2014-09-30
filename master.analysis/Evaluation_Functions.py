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
    """Function that takes a classifier and perform the K-fold cross-validation over
    the specified x an y values"""    
    
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
        
    #----- ITERATE OVER THE K folds -----#
    # NOTE: by scikit-learn default, the method 'cross_val_score(clf, X, y, cv=k_fold)' returns values from the 
    #       'score method' of the estimator (clf), which in case of MultinomialNB is the accuracy. Since we are interested
    #       in other metrics, we call metric functions 'manually' by iterating over the K-fold partitions defined by cv.
    for j, (train, test) in enumerate(cv):
        # fitting the classifier in the corresponding fold for obtaining the corresponding 
        # score measurements on train and test sets
        clf.fit([X[k] for k in train], y[train])
        
        #----- USING PRECISION, RECALL AND F1 AS MEASUREMENTS OF PERFORMANCE (for training set) -----#
        y_train_pred = clf.predict([X[k] for k in train]) #getting predicted results for computing scores
        k_train_precision[j] = metrics.precision_score(y[train], y_train_pred, pos_label='pos')
        k_train_recall[j] = metrics.recall_score(y[train], y_train_pred, pos_label='pos')
        k_train_f1[j] = metrics.f1_score(y[train], y_train_pred, pos_label='pos')
        
        #----- USING PRECISION, RECALL AND F1 AS MEASUREMENTS OF PERFORMANCE (for testing set) -----#    
        y_test_pred = clf.predict([X[k] for k in test]) #getting predicted results for computing scores
        k_test_precision[j] = metrics.precision_score(y[test],y_test_pred, pos_label='pos')
        k_test_recall[j] = metrics.recall_score(y[test],y_test_pred, pos_label='pos')
        k_test_f1[j] = metrics.f1_score(y[test],y_test_pred, pos_label='pos')
        
            
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
    print "### RESULTS FOR THE " + str(K)+"-FOLD CROSS-VALIDATION TEST ###"
    
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
    
    print #just a blank line space :)
    
###End of the function###

def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    """Function that train the classifier (model) in the entire training set and evaluate the accuracy
        in the training and in the testing sets. It will also print a classification report
        (precision and recall on every class) and the corresponding confusion matrix."""

#     print '[+] Training the classifier...'
#     #training the classifier (clf)
#     clf.fit(X_train, y_train)
#     
    print '[+] Performing Cross-validation...'
    #performing 10-Fold cross-validation (only on training set)
    evaluate_cross_validation(clf,X_train,np.array(y_train),10)
    
    print '[+] Accuracy on training set:{:^55}'.format("{0:.3f}".format(clf.score(X_train, y_train)))
    print '[+] Accuracy on testing set:{:^57}'.format("{0:.3f}".format(clf.score(X_test, y_test)))
    
    #testing the classifier on new data for getting final F1 measure (on testing set)
    y_pred = clf.predict(X_test)
    #printing F1 measure
    print "[+] F1-measure: " + str(metrics.f1_score(y_test, y_pred, average=None))

    #printing sklearn classification report
    print "[+] Classification report on test set:"
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
    