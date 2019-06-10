#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from nltk.stem.porter import *
from nltk.corpus import stopwords
from stemmer import Stemmer
import re,string
from itertools import chain

###########################
#                         #
#    FEATURE EXTRACTION   #
#                         #
###########################

def tweet_cleaner(tweet, stopwords = None, s = None):
    tweet = tweet.lower() # convert text to lower-case
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
    tweet = tweet.translate(str.maketrans(string.punctuation,len(string.punctuation)*' ')) # remove punctuation
    tweet = re.findall(r'[a-z]+', tweet) # split words
    if s is not None:
        tweet = [s.stem(word) for word in tweet] # stemming words
    if stopwords is not None:
        tweet = [word for word in tweet if word not in stopwords] # removing stopwords
    return tweet

def tweets_cleaner(tweets, stopwords = None, s = None):
    cleaned_tweets = []
    for i in tweets:
        cleaned_tweets.append(tweet_cleaner(i, stopwords, s))
    return cleaned_tweets

def ngrams(word, n):
    return [word[i:i+n] for i in range(len(word)-1)]

def build_vocab(cleaned_tweets, n = 0):
    vocab = np.array([])
    
    for i in cleaned_tweets:
        if n == 0:
            vocab = np.append(vocab, i)
        else:
            vocab = np.append(vocab, list(chain.from_iterable([ngrams(word,n) for word in i])))
        
    #print(vocab[:5])
    vocab = np.unique(vocab.flatten())
    
    return vocab

def build_features(tweet, vocab, rep = 'bow', n = 0):
    features = np.zeros(len(vocab))
    if rep == 'ngrams':
        tweet = list(chain.from_iterable([ngrams(word,n) for word in tweet]))
    unique, counts = np.unique(tweet, return_counts=True)
    features[np.searchsorted(vocab, unique).astype(int)] = counts
    #np.insert(features,np.searchsorted(vocab,unique),counts)
    #features = np.isin(vocab, tweet).astype(int)
    return features.astype(int)

def build_representation(tweets, vocab, rep = 'bow', n = 0):
    data = []
    for idx, i in enumerate(tweets):
        data.append(build_features(i,vocab, rep, n))
    return data

def extract_features(dataset, rep = 'bow', n=0):
    df_dataset = pd.read_csv( dataset, sep=',', index_col=None, header=0)

    X = df_dataset['TWEET'].values

    Y = df_dataset['CLASS'].values.astype(int)
    Y = np.where(Y != -1, Y, 0)
    Y = np.where(Y != 1, Y, 1)
    
    #limpa e tokeniza os tweets
    cleaned_tweets = tweets_cleaner(X, stopwords = stopwords.words('english'), s = PorterStemmer())
    #constrói o vocabulário
    vocab = build_vocab(cleaned_tweets, n)
    #constrói o vetor de features para cada tweet
    Xfeatures = np.array(build_representation(cleaned_tweets,vocab,rep,n))
    #f = util.build_features(X[1067], vocab)
    #print(Y.shape)
    
    return Xfeatures, Y, vocab

def tweet_preProcess(tweet, vocab, rep = 'bow', n = 0):
    cleaned_tweets = tweets_cleaner(np.array([tweet]), stopwords = stopwords.words('english'), s = PorterStemmer())
    tweet_features = np.array(build_representation(cleaned_tweets,vocab,rep,n))
    return tweet_features