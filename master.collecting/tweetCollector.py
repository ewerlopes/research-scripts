# Finding topics of interest by using the filtering capabilities it offers.
# Autor: Ewerton Lopes
# Last Modification:  07/05/2014


from OauthLogin import oauth_login
import twitter
import time
import sys
import re
#import psycopg2  #Database Library
from TwiSecureRequest import *
#from sendingEmail import sendEmail


#global variables
loopCount = 1 # count how many tweets were analized
reinit_count = 1 # Count how many times the streaming API was called

#connection to a database
#con = psycopg2.connect(host='localhost', user='postgres', password='admin',dbname='DecemberCollection') 

#cursors to operate on database
#c = con.cursor() 


#========================== HELPER FUNCTIONS ==================================

#Function to create table
def createTable():
    #The following command creates a new table named rawData into the database specified by the 'con' object.
    c.execute('CREATE TABLE FluLikeDB_BRAZIL(userUniqID text, userName text, profile_image_url_https text, tweetCreationTime text, tweet text, tweetGeo text, c_class text)')
    #Notice the standard databasing requirement: You need to commit your work. Otherwise, the transaction gets rolled back, and nobody sees your changes. 
    con.commit()


#Function to save the data gathered into a backup file.
def saveDataONFile(data):
    
    try:
        saveFile = open('large_dataset_C.txt','a') #Creating/Open the named file.
        saveFile.write(data) #Write in the file
        saveFile.close() #Closing the file
    except Exception, e:
        print '$$FAILED ON SAVING DATA FOR BACKUP.\n', str(e)

#Function to save log error
def saveLog(errorSource,fDataForBackUp,errorTraceBack):
    tempo = time.localtime(time.time()) #Getting system time
    saveLog = open('log_large_dataset_C.txt','a')#Creating/Open log file.
    saveLog.write(errorSource + ' -- ' + str(tempo[0])+'-'+str(tempo[1])+'-'+str(tempo[2])+', '+str(tempo[3])+':'+str(tempo[4])+':'+str(tempo[5]) +'\n') #Log time
    saveLog.write(str(errorTraceBack)) #Save the error log in the file.
    saveLog.write('Data to be recovered: ' + fDataForBackUp + '\n----------------\n') #data to be backed up.
    saveLog.close()
    
        
#Function to save data into database
def saveIntoDatabase(userID, userName, profileImage, tweetCreationTime, parsedTweet, latitude, longitude, fDataForBackUp):
    try:
        #INSERTING TO A DATABASE
        c.execute("INSERT INTO flulikedb_brazil VALUES ('"+ userID + "','" + userName + "','" + profileImage + "','" + tweetCreationTime + "','" + parsedTweet + "','"+ latitude + ' ' + longitude + "')")
        con.commit() #Uploading data to the database
    except Exception, e:
        print 'DATABASE ERROR: A log file was created.\n', str(e)
        con.rollback() #rollback database transaction
        saveLog('DATABASE ERROR:\n',fDataForBackUp,str(e))




#========================================================================


def main():
    global loopCount, reinit_count

    # define Query terms
    # Comma-separated list of terms
    q = """flu, influenza, cough, sore throat, nose, lung, breath, medication, remedy,
        medical, pill, respiratory, infection, symptom, chest, ache, pain,
        apnea, asthma, difficulties,discomfort, distress, stop breathing,asprin, antibiotic, cold, painful, sneezing, sniffling, bacteria infection
        bacterial infection, ebola,paracetamol, pulmonary, meds, medical, runny"""

    # Returns an instance of twitter.Twitter logged in
    twitter_api = oauth_login()


    # Reference the self.auth parameter to creat a 'stream object'
    twitter_stream = twitter.TwitterStream(auth=twitter_api.auth)


    while (True):
        # See https://dev.twitter.com/docs/streaming-apis
        try:
            stream = make_twitter_request(twitter_stream.statuses.filter,track=q)
            print "........  (Re)Initializing the tweet stream collection ........"
        except BaseException, e:
                error = 'FAILED ON CONECTING TO API,' + str(e)
                print error
                saveLog(error,'None',str(e))
                time.sleep(5)
        
        for tweet in stream:
            
            try:

                print "** Data collected from Twitter: " #+ str(tweet)
                
                #user = tweet['user'] #returns a dictionary with user atributes
                
                #Parsing tweet ID
                tweetID = str(tweet['id'])
                          
                #Parsing tweets to avoid \n return symbol
                parsedTweet = unicode(tweet['text']).encode('utf-8')
                
                #print 'Is_List:' + str(type(parsedTweet)==list) + '\n'
                #print 'Type:' + str(type(parsedTweet)) + '\n'
                                    
                parsedTweet = parsedTweet.replace('\n',' ').replace('\r\n',' ').replace('\r',' ') #removing newline characters
                parsedTweet = re.sub(r'\s{2,140}', ' ', parsedTweet) #removing aditional spaces

                #Dealing with apostrophe (\') character that may cause string brake at the moment of saving into database, i.e.: don't, I'm...
                #The following split() method return a list with one element if its argument doen't exist in the string. Otherwise it returns
                #a list sized with the number of its argument ocurrences.
                parsedTweet = parsedTweet.split('\'') 

                #Auxiliar variable to fix apostrophe problem if it is detected.
                #Initially it is assigned to the first element in the list returned by the method split
                fixTweet = parsedTweet[0] 

                if len(parsedTweet) != 1:
                    for iter in parsedTweet[1:]:
                        fixTweet = fixTweet + "''" + str(iter)
                    parsedTweet = fixTweet
                else:
                    parsedTweet = fixTweet
                    
                fixTweet = '' #Clear fixTweet for next iteration
                        
                #Enconding all data gathered to avoid problems with unicode/utf-8/ascII in storage       
                tweetID = unicode(tweetID).encode('utf-8')
                tweetCreationTime = unicode(tweet['created_at']).encode('utf-8')
                
                    
                    
                #Formating data to save it into a backup file
                data = tweetID + ' :&: ' + tweetCreationTime + ' :&: ' + parsedTweet + '\n'
                print data
                
                print #printing blank lines
                print "** Tweets Downloaded: " + str(loopCount) #Print this information in the python console
                print "** # Call to API: " + str(reinit_count) #Print this information in the python console
                print #printing blank lines
                    
                saveDataONFile(data) #saving data variable on file.

                loopCount += 1 #updating the counter
                

            except BaseException, e:
                error = 'FAILED ON COLLECTING DATA FROM STREAM,' + str(e)
                print error
                saveLog(error,'None',str(e))
                time.sleep(5)

            print  #printing blank lines
            print ".........."
            print  #printing blank lines

        reinit_count +=1 #updating the API calls counter

#Starting the script
main()


'''
--USEFUL COMMENTS--

-Issues of encoding
In Python 2.x, str() tries to convert its contents to 8-bit strings. Since you're passing it Unicode objects, it uses
the default encoding (ascii) and fails, before the .encode('utf-8') call is reached. Using unicode() is redundant here
if the data you're getting is all text, but will be useful if some of it is, say, integers, so I recommend the latter.


'''





