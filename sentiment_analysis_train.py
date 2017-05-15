'''
Yuyu Li, May 2017
Featurizes Tweets from TwitterTrails data.
'''
import random
import sys
import re 
import csv
import numpy as np
import sentiment_analysis_LR as salr
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer

hyperparams, accuracy = salr.gridsearch("sentiment_train.txt")

def load_csv(filename):
    with open(filename, 'rU') as csv_data:
        data = []
        X = [] # only features
        y = [] # only true label 
        all_text = []
        all_other_data = []
        
        reader = csv.reader(csv_data, dialect='excel')
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
            # X.extend([prob_deny, prob_query, prob_support]) 
            y.append(truth_bool)
            data.append([text, story_id, tweet_id, skept_bool, truth_bool, prob_deny, prob_query, prob_support])
        
        return (all_text, X, y, all_other_data, data)

def sentiment_featurize(filename, text, X):
    lr = linear_model.SGDClassifier(loss='log', penalty='l2', alpha=hyperparams[1], l1_ratio=0, n_iter=hyperparams[2],
    epsilon=0.1, eta0=hyperparams[3], class_weight='balanced')
    pre_trainX, pre_trainy = salr.load_data(filename)
    trainX, trainy = salr.filtered(pre_trainX, pre_trainy)
    vec1 = TfidfVectorizer(lowercase=True, ngram_range=(1,hyperparams[0]), encoding="latin1")
    vec1 = vec1.fit(trainX)
    tfidf_X = vec1.transform(text).toarray()
    tfidf_trainX = vec1.transform(trainX).toarray()
    lr.fit(tfidf_trainX, trainy)
    prob = lr.predict_proba(tfidf_X)
    return prob

def combine(filename):
    all_text, X, y, all_other_data, data = load_csv(filename)
    prob = sentiment_featurize("sentiment_train.txt", all_text, X)
    #print X
    #print prob.shape[0]
    for i in range(prob.shape[0]): 
        X[i].extend(X[i])
        X[i].extend(prob[i])
    X = np.array(X)
    #print X
    
    '''
    for row in range(X.shape[0]):
        for col in range(X.shape[1]):
            X[row][col] = float(X[row][col])
    #print "floated: {}".format(X)
    '''

    unique_story_ids = set([row[0] for row in all_other_data])
    random.seed(5) 
    story_train = random.sample(unique_story_ids, 40)
    remainder = set([item for item in unique_story_ids if item not in story_train])
    story_dev = random.sample(remainder, 5)
    
    trainX = []
    trainy = []
    devX = []
    devy = []
    testX = []
    testy = []
    for row in range(X.shape[0]):
        if all_other_data[row][0] in story_train:
            trainX.append(X[row])
            trainy.append(all_other_data[row][3])
        elif all_other_data[row][0] in story_dev:
            devX.append(X[row])
            devy.append(all_other_data[row][3])
        else: 
            testX.append(X[row])
            testy.append(all_other_data[row][3])
    
    trainX = np.array(trainX)
    trainy = np.array(trainy)
    devX = np.array(devX)
    devy = np.array(devy)
    testX = np.array(testX)
    testy = np.array(testy)
    trainX = trainX.astype(float)
    devX = devX.astype(float)
    testX = testX.astype(float)
    return (trainX, trainy, devX, devy, testX, testy)
  
    
def logreg_gridsearch(trainX, trainy, devX, devy, testX, testy):
    preds = []
    probs = []
    accuracies = []
    hyperparams = []
    test_accs = []
    
    alpha_vals = [.5, .1, 1e-2, 1e-3, 1e-4 ,1e-5]
    n_iter_vals = [10, 25, 50]
    eta_vals = [1e-3, 1e-4, 1e-5]
    
    for alpha in alpha_vals:
        for n_iter in n_iter_vals:
            for eta in eta_vals:
                lr = linear_model.SGDClassifier(loss='log', penalty='l2', alpha=alpha, l1_ratio=0, n_iter=n_iter,
                epsilon=0.1, eta0=eta, class_weight='balanced')
                lr.fit(trainX, trainy)
                pred = lr.predict(devX)
                prob = lr.predict_proba(devX)
                accuracy = lr.score(devX, devy)
                preds.append(pred)
                probs.append(prob)
                accuracies.append(accuracy)
                hyperparams.append([alpha, n_iter, eta])
                test_acc = lr.score(testX, testy)
                test_accs.append(test_acc)
    return (preds, probs, accuracies, hyperparams, test_accs)

trainX, trainy, devX, devy, testX, testy = combine("pred_skeptVals_all.csv")
preds, probs, accuracies, hyperparams, test_accs = logreg_gridsearch(trainX, trainy, devX, devy, testX, testy)
max_ind = np.argmax(accuracies)
print "------------------"
print "COMBINED OPTIMIZED PARAMETERS:"
print hyperparams[max_ind]
print hyperparams
print "COMBINED OPTIMIZED ACCURACY:"
print accuracies[max_ind]
print accuracies

optimized = hyperparams[max_ind]
lr = linear_model.SGDClassifier(loss='log', penalty='l2', alpha=optimized[0], l1_ratio=0, n_iter=optimized[1], 
epsilon=0.1, eta0=optimized[2], class_weight='balanced')
lr.fit(trainX, trainy)
print "FINAL ON TEST SET:"
print lr.score(testX, testy)
print "------------------"
print test_accs
