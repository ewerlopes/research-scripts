# -*- coding: UTF-8 -*-
# # Autor: Ewerton Lopes
# # Data: 24/09/2014
# # This script contains functions for extracting features of a data set and making (returning) the
# # feature vector for tweet representation. It calls pre-processing steps in order to reduce noise
# # associated with the raw data. It also assumes that the raw data is stored in a plain text file 
# # (*.txt) and it is passed as a list of sentences where each item represents a tweet instance.

import random
import string
import os
import sys
from Preprocess_Util import processTweet
from Dataset_Util import train_test_split
#from Project_Util import size_mb


def process_raw_dataset(data_set_name, sep = " :&: ", tweet_pos = 1, target_pos=2):
    """This function parses the raw data set in order to get a preprocess version of it. Also
    it call partition functions to divide the data into training and test sets."""
    try:
        print "[+] Reading "+data_set_name+ " in 'master.data' directory...."
        f = open('../master.data/'+ data_set_name, 'r') #reads the data set file into a file object (f)
        data_set_lines = f.readlines() #each line corresponds to a data
        f.close() #closing opened data file
    
        print "[+] Done!\n"
        print '[+] Checking if there is already a preprocessed version (pp_data_set_name.txt) of the data set in the directory!'
        
        if os.path.isfile('../master.data/pp_'+data_set_name):
            print "[+] A preprocessed version of the file already exists in the directory..."
            #retrieving preprocessed cache in ../master.data/ directory
            data_set_lines = RetrievingCache(data_set_name,len(data_set_lines),sep, tweet_pos)
        else:
            print "[+] No previous preprocessed version of the data set!"
            data_set_lines = PreprocData(data_set_name, data_set_lines, sep, tweet_pos) #initializing preprocessing steps on the entire data set. 

    except Exception, e:
        print "[*] File-in error: Error in reading the data set! "
        print str(e)
        sys.exit()
        
    
    random.shuffle(data_set_lines) #randomizing data points

    data = [] #list of predictors (tweets)
    targets = [] #list of y (targets)
    for line in data_set_lines:
        datapoint = line.split(sep) #getting different data in one line of data using 'sep' as delimiter
        data.append(datapoint[tweet_pos]) #using tweet_pos to get only the tweet data
        targets.append(datapoint[target_pos].rstrip('\n')) #getting the target variable of the data line

    #returning da data and targets
    return data, targets
#End of the function

def PreprocData(data_set_name, data_set_lines, sep, tweet_pos):
    """This function preprocesses the data set and makes a cache copy (by adding pp_prefix) of it 
    in master.data directory"""
    
    print "[+] Preprocessing " + data_set_name + ' file'
    
    if data_set_lines == None: #Read the file again
        try:
            lines = open('../master.data/'+ data_set_name).readlines()
            #print "Data set size:" + str(size_mb(lines)) #printing data set size in MB
        except Exception, e:
            print '[*] It is not possible to read ' + data_set_name
            print str(e)
            sys.exit()
    else:
        lines = data_set_lines #original lines of the data set
        #print "Data set size:" + str(size_mb(lines)) #printing data set size in MB
    
    count = 0 #counter for visualization issues.
    
    proc_dataset = [] #list of preprocessed data lines
    
    try:
        #append 'pp_' prefix in front of the data set name in order to distinguish 
        # between preprocessed and original version.
        procf = open('../master.data/pp_'+data_set_name, 'a')
    except Exception, e:
        print '[*] Error at creating a new preprocessed file!'
        print str(e)
        sys.exit()
        
    for line in lines: #loop over the data set lines
        line = line.split(' :&: ') #spliting data in order to get access to the text data.
        processedTweet = processTweet(line[tweet_pos]) #preprocess tweet data in line[tweet_pos] (becomes 'None' is the data isn't English-based text)
        
        if processedTweet != None: #in case of the tweet being English-based text
            print '[Line '+str(count)+']' + ' -OK- ' + processedTweet #Visualization
            line[tweet_pos] = processedTweet #persisting processedTweet into line[twee_pos]
            line = string.join(line, sep) #recombining the data line (pos-processing)
            procf.write(line) #saving in a cache file for avoiding unnecessary computation resources in the future.
            proc_dataset.append(line) #appending processed data for returning the function
        else:
            line[tweet_pos] = "### NOT ENGLISH (AVOIDING) ###"
            print '[Line '+str(count)+']' + ' -FAILED- ' + line[tweet_pos]
            line = string.join(line, sep) #recombining the data line (pos-processing)
            procf.write(line) #saving in a cache file for avoiding unnecessary computation resources in the future.
        
        count += 1 #updating counter for visualization purposes.
        
    procf.close() #closing file
    print '[+] Preprocessed data set cache was saved as: ' 'pp_' + data_set_name
    return proc_dataset #returns the preprocess data set as a list of string (each tweet per line)
#end of the function

def RetrievingCache(data_set_name,dataset_length,sep,tweet_pos):
    """This function retrieves the cache of an already preprocessed data set based on the match between its original name and
    the existence of a file with 'pp_' + 'data_set_file_name' in the master.data directory. This process is intended to avoid
    unnecessary computational resources usage since preprocessing a large data set is an expensive task."""
    
    try:
        #append 'pp_' prefix in front of the data set name in order to distinguish 
        # between preprocessed and original version.
        print '[+] Retrieving preprocessed data set cache...'
        procf = open('../master.data/pp_'+data_set_name, 'r') #open cache file
        dataset_cache = procf.readlines() #list of cache data lines
        #print "Data set size:" + str(size_mb(dataset_cache)) #printing data set size in MB
        procf.close() #closing file
        
        print('[+] Checking cache consistency...')
        #Checking if the cache has the same numbers of lines as its original unprocessed version
        if len(dataset_cache) == dataset_length:
            print('OK!')
            #eliminating "### NOT ENGLISH (AVOIDING) ###"
            proc_dataset = []  #list of preprocessed data lines
            for line in dataset_cache:
                #avoiding non-english lines
                if line.split(sep)[tweet_pos] != "### NOT ENGLISH (AVOIDING) ###":
                    proc_dataset.append(line)
            print '[+] Data set cache was loaded successfully!'
            return proc_dataset #returns the preprocess data set as a list of string (each tweet per line)              
        else:
            procf.close() #closing opened file
            print('FAILED!') #Cache isn't consistent (same number of lines) as its original unprocessed version.
            
            try: #trying to remove the inconsistent cache from the directory for making a new one
                os.remove('../master.data/pp_'+ data_set_name)
            except Exception, e:
                print '[*] It was not possible to delete the inconsistent cache. Please, try it manually and restart the code!'
                print str(e)
                sys.exit()
            
            print '[+] Making a new cache from the original data set...'
            return PreprocData(data_set_name, open('../master.data/'+ data_set_name).readlines(),sep,tweet_pos) #making a new preprocessed cache.
    
    except Exception, e:
        print '[*] Error on getting preprocessed cache file!'
        print str(e)
        sys.exit()
#end of the function

#USAGE:
#X_train, y_train, X_test,y_test = process_raw_dataset('tuarob_dataset.txt')