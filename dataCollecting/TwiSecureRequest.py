import sys
import time
from urllib2 import URLError
from httplib import BadStatusLine
import json
import twitter

def make_twitter_request(twitter_api_func, max_errors=10, *args, **kw):

    # A nested helper function that handles common HTTPErrors. Return an updated
    # value for wait_period if the problem is a 500 level error. Block until the
    # rate limit is reset if it's a rate limiting isse (429 Error). Returns None
    # for 401 and 404 errors, which requires special handling by the caller.
    def handle_twitter_http_error(e, wait_period=2, sleep_when_rate_limited=True):
            if wait_period > 3600: #seconds
                print 'Too many retries. Quitting.'
                raise e

            if e.e.code == 401:
                print 'Encountered 401 Error (Not Authorized)'
                return None
            elif e.e.code == 404:
                # The URI requested is invalid or the resource requested, such as a user,
                # does not exists. Also returned when the requested format is not supported
                # by the requested method.
                print 'Encountered 404 Error (Not Found)'
                return None
            elif e.e.code == 429:
                #Returned in API v1.1 when a request cannot be served due to the application's
                #rate limit having been exhausted for the resource. See Rate Limiting in API v1.1.
                print 'Encountered 429 Error (Rate Limit Exceed)'
                if sleep_when_rate_limited:
                    print 'Retrying in 15 minutes...ZzZ...'
                    time.sleep(60*15+10)
                    print '...ZzZ... Awake now and trying again.'
                    return 2
                else:
                    raise e #Caller must handle the rate limiting issue

            elif e.e.code in (500,502,503,504):
                print 'Encountered '+ e.e.code + 'Error. Retrying in ' + wait_period + ' seconds'
                time.sleep(wait_period)
                wait_period *= 1.5
                return wait_period
            else:
                raise e

    #End of nested helper function

    wait_period = 2
    error_count = 0

    while True:
        try:
            
            return twitter_api_func(*args,**kw)
        except twitter.api.TwitterHTTPError, e:
            error_count = 0
            wait_period = handle_twitter_http_error(e, wait_period)
            if wait_period is None:
                return
        except URLError, e:
            error_count += 1
            print 'URLError encountered. Continuing.'
            if error_count > max_errors:
                print 'Too many consecutive errors... bailing out.'
                raise
        except BadStatusLine, e:
            error_count += 1
            print 'BadStatusLine encountered. Continuing.'
            if error_count > max_errors:
                print 'Too many consecutive errors... bailing out.'
                raise
