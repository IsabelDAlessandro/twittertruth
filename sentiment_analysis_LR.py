'''
Yuyu Li, April 2017
Sentiment analysis portion. Based off of SemEval Task 4E, to classify Tweets into 5 different categories:
very negative, negative, neutral, positive, very positive on a 5-point scale (-2 to 2). 
'''
import sys
import re 
import numpy as np
import tweepy
from sklearn import linear_model

#auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
#auth.set_access_token(access_token, access_token_secret)

# api = tweepy.API(auth)

def load_training(filename):
    matrix = []
    with open(filename) as txt_data:
        for line in txt_data:
            row = re.split(r'\t+', line.rstrip())
            matrix.append(row)
    # print matrix
    matrix[:][0] = get_tweet(matrix[:][0])
    trainX = np.array(matrix[:][0:1])
    trainy = np.array(matrix[:][2])
    return (X, y)
    
def get_tweet(tweet_id):
    # content = urllib2.urlopen("http://twitter.com/anyuser/status/628949369883000832")
    # print content
    return "test"
    
def train(filename):
    trainX, trainy = load_training(filename)
    lr = linear_model.SGDClassifier(loss='log', penalty='l2', alpha=0.0001, l1_ratio=0, n_iter=5,
    epsilon=0.1, eta0=0.0)
    lr.fit(trainX, trainy)
    # preds = lr.predict(testX, testy)
