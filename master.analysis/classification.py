# Author: Ewerton Lopes <ewerlopes@gmail.com>
# Date: May 26th 2014
# License: Creative commons

from tabulate import tabulate
from nltk.corpus import stopwords
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.cross_validation import cross_val_score, KFold
from sklearn import metrics
from scipy.stats import sem
from numpy import argmax
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pylab as pl
import random


#================================ HELPER FUNCTIONS ============================================================

#######################################################################
def train_test_split(data, target, SPLIT_PERC=0.66):
    """Partitions the dataset available (data and target) into training and testing set"""
    
    print "Partitioning data...."
    split_size = int(len(data) * SPLIT_PERC)
    X_train = data[:split_size]
    X_test =  data[split_size:]
    y_train = target[:split_size]
    y_test = target[split_size:]
    print "Done!"
    print

    return X_train, X_test, y_train, y_test
    

#######################################################################
def evaluate_cross_validation(clf, X, y, K):
    """Function that takes a classifier and perform the K-fold cross-validation over
        the specified x an y values"""
    
    #create a k-fold cross validation iterator of K-folds
    k_fold = KFold(len(y), K, shuffle=True, random_state=0)
    for train_indices, test_indices in k_fold:
        print('Train: %s | test: %s' % (train_indices, test_indices))


    #by default the score used is the one returned by score method of the estimator (accuracy)
##    scores = cross_val_score(clf, X, y, cv=cv)
##    print "Results for the " + str(K)+"-fold (Only on test data)...."
##    print scores
##    print (str(K)+"-fold " + "Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))
    

#######################################################################
def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    """Function that train the model in the entire training set and evaluate the accuracy
        in the training and in the testing sets. It will also print a classification report
        (precision and recall on every class) and the corresponding confusion matrix."""

    clf.fit(X_train, y_train)

    evaluate_cross_validation(clf,X_test,np.array(y_test),10)
    
    print 'Accuracy on training set:{:^55}'.format("{0:.3f}".format(clf.score(X_train, y_train)))
    print 'Accuracy on testing set:{:^57}'.format("{0:.3f}".format(clf.score(X_test, y_test)))

    y_pred = clf.predict(X_test)
    print "F1-measure: " + str(metrics.f1_score(y_test, y_pred, average=None))

    print "Classification report on test set:"
    print metrics.classification_report(y_test, y_pred)

    #computing confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    print "Confusion Matrix:"
    print (cm)        #print in the console

    # Show confusion matrix in a separate window
    pl.matshow(cm)
    pl.title('Confusion matrix')
    pl.colorbar()
    pl.ylabel('True label')
    pl.xlabel('Predicted label')
    pl.show()

#######################################################################
def calc_params(X, y, clf, param_values, param_name, K):
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
    
    


#================================ MAIN ============================================================

def main(corpus_name, sep = " :&: ", tweet_pos = 1, target_pos=2):

    try:
        print "Reading "+corpus_name+ " data...."
        f = open(corpus_name, 'r')
    except Exception, e:
        print "File-in erros: " + str(e)
    
    file_lines = f.readlines() #each line corresponds to a data
    print "Done!"
    print
    
    random.shuffle(file_lines) #randomizing datapoints

    data = [] #list of predictors (tweets)
    targets = [] #list of y (targets)
    
    for line in file_lines:
        datapoint = line.split(sep)
        data.append(datapoint[tweet_pos])
        targets.append(datapoint[target_pos].rstrip('\n'))

    X_train, X_test, y_train, y_test =  train_test_split(data,targets)
        
    #Classifier 1
    clf_1 = Pipeline([
        ('vect', CountVectorizer()),#stop_words = set(stopwords.words('english')))),
        ('nb', MultinomialNB()),
        ])
    

    #Classifier 2
    clf_2 = Pipeline([
        ('vect', CountVectorizer(binary=True, ngram_range=(2,2))), #stop_words = set(stopwords.words('english')))),
        ('nb', MultinomialNB()),
        ])

    #Classifier 3
    clf_3 = Pipeline([
        ('vect', TfidfVectorizer(ngram_range=(2,2))),#stop_words = set(stopwords.words('english')))),
        ('nb', MultinomialNB()),
        ])
    
    

    #classifiers
    clfs = [clf_1]
    #count for # of tests
    trials_count = 0

    for clf in clfs:
        
        print '-'*100
        print '{:*^100}'.format(' TRAINING AND TESTING - number: ' + str(trials_count))
        print '-'*100
        print "Model details: " + str(clf)
        print
        
        alphas = np.logspace(-7, 0, 8)

        #evaluating performance (model evaluation)
        train_precision_score, train_recall_score,train_f1_score, \
                               test_precision_score,test_recall_score,test_f1_score, train_f1_sem, test_f1_sem = \
                               calc_params(data, np.array(targets), clf, alphas, 'nb__alpha', 10)
        print
        print 'TRAINING SCORES... (10-Fold cross-validation)'
        header = ['nb__alpha'] +  ["%.3e" % number for number in alphas] + ['Best Score']
        #creating table for measurements on training data
        table = [['Precision'] + train_precision_score.tolist() + [train_precision_score[train_precision_score.argmax(axis=0)]], \
                 ['Recall'] + train_recall_score.tolist() + [train_recall_score[train_recall_score.argmax(axis=0)]], \
                 ['F1'] + train_f1_score.tolist() + [train_f1_score[train_f1_score.argmax(axis=0)]]]
        print tabulate(table,header, numalign="right",floatfmt=".3f")
        print
        print 'TESTING SCORES... (10-Fold cross-validation)'
        #creating table for measurements on testing data
        table = [['Precision'] + test_precision_score.tolist() + [test_precision_score[test_precision_score.argmax(axis=0)]], \
                 ['Recall'] + test_recall_score.tolist() + [test_recall_score[test_recall_score.argmax(axis=0)]], \
                 ['F1'] + test_f1_score.tolist() + [test_f1_score[test_f1_score.argmax(axis=0)]]]

        print tabulate(table,header, numalign="right",floatfmt=".3f")
        print test_f1_sem
        print
        print '# TOKENS IN THE DICTIONARY: ' + str(len(clf.named_steps['vect'].get_feature_names()))
        print
        print
        #updating counter
        trials_count += 1



#===== RUNNING =====
main("tuarob_dataset.txt")
