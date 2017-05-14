'''
Yuyu Li, May 2017
Featurizes Tweets from TwitterTrails data.
'''

import sys
import re 
import csv
import numpy as np
import sentiment_analysis_LR as salr
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer

def load_csv(filename):
    with open(filename, 'rb') as csv_data:
        data = []
        X = [] # only features
        y = [] # only true label 
        all_text = []
        all_other_data = []
        
        reader = csv.reader(csv_data, delimiter=',')
        for row in reader:
            text = row[0]
            story_id = row[1]
            tweet_id = row[2]
            skept_bool = row[3]
            truth_bool = row[4]
            prob_deny = row[5]
            prob_query = row[6]
            prob_support = row[7]
            
            all_text.append(text)
            all_other_data.append([story_id, tweet_id, skept_bool, truth_bool])
            X.append([prob_deny, prob_query, prob_support])
            y.append(truth_bool)
            data.append([text, story_id, tweet_id, skept_bool, truth_bool, prob_deny, prob_query, prob_support])
        
        return (all_text, X, y, all_other_data, data)

def sentiment_featurize(filename, text, X):
    lr = linear_model.SGDClassifier(loss='log', penalty='l2', alpha=0.5, l1_ratio=0, n_iter=10,
    epsilon=0.1, eta0=0.001, class_weight='balanced')
    trainX, trainy = salr.load_data(filename)
    vec1 = TfidfVectorizer(lowercase=True, ngram_range=(1,4))
    vec1 = vec1.fit(trainX)
    tfidf_X = vec1.transform(text).toarray()
    vec2 = TfidfVectorizer(lowercase=True, ngram_range=(1,4))
    tfidf_trainX = vec2.fit_transform(trainX).toarray()
    lr.fit(tfidf_trainX, trainy)
    prob = lr.predict_proba(tfidf_X)
    return prob

def combine(filename):
    all_text, X, y, all_other_data, data = load_csv(filename)
    prob = sentiment_featurize("sentiment_train.txt", all_text, X)
    print X
    print prob.shape[0]
    for i in range(prob.shape[0]): 
        X[i].extend(prob[i])
    print X
    
combine("testcsv.csv")