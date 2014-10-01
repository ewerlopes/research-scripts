import twitter

def oauth_login():
    # XXX: Go to http://twitter.com/apps/new to create an app and get values
    # for these credentials that you'll need to provide in place of these
    # empty string values that are defined as placeholders.
    # See https://dev.twitter.com/docs/auth/oauth for more information 
    # on Twitter's OAuth implementation.
    
    CONSUMER_KEY = 'enter the key'
    CONSUMER_SECRET = 'enter the key'
    OAUTH_TOKEN = 'enter the key'
    OAUTH_TOKEN_SECRET = 'enter the key'
    
    auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
                               CONSUMER_KEY, CONSUMER_SECRET)
    
    twitter_api = twitter.Twitter(auth=auth)
    return twitter_api

# Sample usage: twitter_api = oauth_login()    

# Nothing to see by displaying twitter_api except that it's now a
# defined variable: print twitter_api
