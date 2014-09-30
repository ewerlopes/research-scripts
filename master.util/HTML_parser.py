# -*- coding: UTF-8 -*-
# # Autor: Ewerton Lopes
# # Data: 26/09/2014
# # This script contains custom functions for cleaning up HTML tags on a string.
# # Source: http://stackoverflow.com/questions/753052/strip-html-from-strings-in-python

from HTMLParser import HTMLParser

class MLStripper(HTMLParser):
    """This class is intended to provide methods for removing HTML tags on a string"""
    def __init__(self):
        self.reset()
        self.fed = []
    
    def handle_data(self, d):
        self.fed.append(d)
        
    def get_data(self):
        return ''.join(self.fed)