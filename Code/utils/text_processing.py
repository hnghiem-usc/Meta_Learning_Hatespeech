# Modified from base version for the Twitter_RACISM project.

# from __future__ import print_function
import numpy as np
import pandas as pd
import re
import json
import os, sys 
import matplotlib.pyplot as plt
import argparse
import string
# import spacy #might be problematic if not 
import nltk
from ekphrasis.classes.segmenter import Segmenter

# %run -n

# pd.set_option('display.max_columns',5000)
# pd.set_option('display.max_rows',5000)
# pd.options.display.max_rows = 4000
# pd.options.display.max_seq_items = 2000
# pd.set_option('display.max_colwidth', 2000)




######################## I/O ######################
def load_json(file): 
    'read a json with the provided full file path'
    with open(file, ) as f:
        data = json.load(f)
    f.close()
    return data




##### TEXT PROCESSING #####
def clean_text(text):
    '''
    Clean tweets/string with the current steps
    '''
    text = text.encode("ascii", "ignore").decode()
    text = text.lower().replace('rt','')
    #No mentions or link
    text_no_url = re.sub(r"(?:\@|http|https?\://)\S+", "", text)
    text_no_url = re.sub(r'www\S+', '', text_no_url)
    #remove emoticon 
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)s
                           "]+", flags = re.UNICODE)
    text_no_emojis = regrex_pattern.sub(r'', text_no_url)
    # remove punctuations and convert characters to lower case
    text_nopunct = "".join([char for char in text_no_emojis if char not in string.punctuation]) 
    # remove all non-alphanumeric
    text_noalpha = text_nopunct.replace('([^0-9a-z \t])',' ')
    # remove number with blank
    text_nonum = re.sub('\d+', '', text_noalpha)
    # substitute multiple whitespace with single whitespace and strip
    text_no_doublespace = re.sub('\s+', ' ', text_nonum).strip()
    
    return text_no_doublespace


# def standardize_hashtag(s, ldict):    
#     #### Second implementation ####
#     """
#     process hashtags in desired fashions before feeding into the BERT models
#     @param s: string, tweet/document
#     @param ldict: dict, lookup dict of keys to be hashta: values standard versions
#     """
#     # Convert all hashtag to space
#     s = re.sub(r'#', ' ', s)
#     # Iterate over the keys in dict and modify only if tag is not a part of other words
#     for k, v in ldict.items():
#         reg_pattern = re.compile(pattern = r'\b({0})\b'.format(k), flags=re.IGNORECASE)
#         s = reg_pattern.sub(v, s)
        
#     return s

def standardize_hashtag(s, ldict, seg):    
    #### Second implementation ####
    """
    process hashtags in desired fashions before feeding into the BERT models
    @param s: string, tweet/document
    @param ldict: dict, lookup dict of keys to be hashta: values standard versions
    @param seg: dict, segmenter object of Ekphrasis package , prefer Twitter corpus
    """
    # split all hashtag that is not in the list 
    hash_pattern = "#(\w+)"
    hash_list = re.findall(hash_pattern, s)
    for k in hash_list: 
        if k not in ldict: 
            s = s.replace('#'+k, ' ' + seg.segment(k))
            
    # Convert leftover  hashtag to space
    s = re.sub(r'#', ' ', s)
    # Iterate over the keys in dict and modify only if tag is not a part of other words
    for k, v in ldict.items():
        reg_pattern = re.compile(pattern = r'\b({0})\b'.format(k), flags=re.IGNORECASE)
        s = reg_pattern.sub(v, s)
        
    return s


# def process_tweet_bert_orig(text, ldict = None, base=0):
#     '''
#     Clean tweets/string with the current steps
#     '''
#     text = text.encode("ascii", "ignore").decode()
#     text = text.lower().replace('rt:','')
#     # replace @user with <user>
#     text = re.sub(r'(?:\@)\S+', '<user>', text)
#     # replace link with token [url]
#     text = re.sub(r"(?:\www|http|https?\://)\S+", "<url>", text)
#     # process hashtah 
#     text = standardize_hashtag(text, ldict)
#     # remove emojis and emoticons
#     regrex_pattern = re.compile(pattern = "["
#         u"\U0001F600-\U0001F64F"  # emoticons
#         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#         u"\U0001F680-\U0001F6FF"  # transport & map symbols
#         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)s
#                            "]+", flags = re.UNICODE)
#     text = regrex_pattern.sub(r'', text) 
#     # substitute multiple whitespace with single whitespace and strip
#     text = re.sub('\s+', ' ', text).strip()
#     # group multiple [user] or [url] tokens into 1
#     text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
#     # remove duplicate <user> characters
#     text = re.sub(r'[\b|.]|(<user>)( \1)+', r'\1', text)
    
#     return text.strip()


def process_tweet_bert(text, ldict = None, seg=None, base=0, verbose=False):
    '''
    Clean tweets/string with the current steps
    '''
    text = text.encode("ascii", "ignore").decode()
    text = text.lower()
    text = re.sub(r'\b(rt)\b','',text)
    # replace @user with <user>
    text = re.sub(r'(?:\@)\S+', '<user>', text)
    # replace link with token [url]
    text = re.sub(r"(?:\www|http|https?\://)\S+", "<url>", text)
    # process hashtah 
    text = standardize_hashtag(text, ldict,seg)
    # remove emojis and emoticons
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)s
                           "]+", flags = re.UNICODE)
    text = regrex_pattern.sub(r'', text) 
    # substitute multiple whitespace with single whitespace and strip
    text = re.sub('\s+', ' ', text).strip()
    # group multiple [user] or [url] tokens into 1
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    # remove duplicate <user> characters
    text = re.sub(r'(<user>)( \1)+', r'\1', text)
    # substitute multiple double quote with single quote and strip
    text = re.sub(r'(\")( \1)+', r'\1', text)
    # compress space in quotation marks
    text = re.sub(r'\"\s+', '"', text) 
    text = re.sub(r'\b\s+\"', '"', text)
    if verbose: print(text)
    # For LATENT-HATRED data,remove ": in the beginning 
    text = re.sub(r'^(\"\s*:\s*)', '' , text)
    
    return text.strip()


def process_text_bert(text, ldict = None, seg=None, base=0, verbose=False):
    '''
    Clean tweets/string with the current steps
    '''
    text = text.encode("ascii", "ignore").decode()
    text = text.lower()
    text = re.sub(r'\b(rt)\b','',text)
    # replace reddit u\**user**
    text = re.sub(r'([\/]*u\/\S+)', '<user>', text)
    # replace r/reddit with <sub>
    text = re.sub(r'([\/]*r\/\S+)', '<sub>', text)
    # replace [linebreak] with none
    text = re.sub(r'\[linebreak\]', '', text)
    # replace @user with <user>
    text = re.sub(r'(?:\@)\S+', '<user>', text)
    # replace link with token [url]
    text = re.sub(r"(?:\www|http|https?\://)\S+", "<url>", text)
    # process hashtah 
    text = standardize_hashtag(text, ldict,seg)
    # remove emojis and emoticons
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)s
                           "]+", flags = re.UNICODE)
    text = regrex_pattern.sub(r'', text) 
    # substitute multiple whitespace with single whitespace and strip
    text = re.sub('\s+', ' ', text).strip()
    # group multiple [user] or [url] tokens into 1
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    # remove duplicate <user> characters
    text = re.sub(r'(<user>)( \1)+', r'\1', text)
    # remove duplicate <sub> characters
    text = re.sub(r'(<sub>)( \1)+', r'\1', text)
    # substitute multiple double quote with single quote and strip
    text = re.sub(r'(\")( \1)+', r'\1', text)
    # compress space in quotation marks
    text = re.sub(r'\"\s+', '"', text) 
    text = re.sub(r'\b\s+\"', '"', text)
    if verbose: print(text)
    # For LATENT-HATRED data,remove ": in the beginning 
    text = re.sub(r'^(\"\s*:\s*)', '' , text)
    # remove > at the begining of sentence
    text = re.sub(r'^(>|\*)\s*', '' , text)
    
    return text.strip()