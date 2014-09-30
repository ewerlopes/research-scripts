# -*- coding: UTF-8 -*-
# # Autor: Ewerton Lopes
# # Data: 25/09/2014
# # This script contains custom functions for performing simples tasks.

from HTML_parser import MLStripper

def size_mb(docs):
    """Function that returns the size of docs in MB"""
    return sum(len(s.decode('utf-8')) for s in docs) / 1e6

def strip_tags(html):
    """This function calls MLStripper class in order to get rid of HTML tags and entities."""
    s = MLStripper()
    s.feed(html)
    return s.get_data()