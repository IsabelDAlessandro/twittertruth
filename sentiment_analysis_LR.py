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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

auth = tweepy.OAuthHandler("YFRzX9bhodRdkbVglbM1SCxQY", "6KS4EpzT8PoqPIfLzwWhby4uYo5lN153eizEqaUfWb4WprJy6o")
auth.set_access_token("860217211616677888-UrbfhHrpFISehSsee1unY4USOduJu5o", "YFqA6tbgGxzsAefcEUJPe4fZjLzyZ92tnbOMackIroWPG")

api = tweepy.API(auth)

def load_data(filename):
    error_count = 0
    matrix = []
    with open(filename) as txt_data:
        for line in txt_data:
            row = re.split(r'\t+', line.rstrip())
            matrix.append(row)
    # print matrix
    #matrix[:][0] = api.get_status(matrix[:][0])
    matrix = np.array(matrix)
    dataX = np.array(matrix[:, 0])
    try:
        for i in range(0, dataX.shape[0]):
            dataX[i] = api.get_status(dataX[i])
    except:
        error_count += 1
    datay = np.array(matrix[:, 2])
    #print error_count
    #print trainX.shape[0]
    return (dataX, datay)
    
def cross_validate(X, y):
    train_matX = []
    train_maty = []
    test_matX = []
    test_maty = []
    kf = KFold(n_splits=10)
    for train_split, test_split in kf.split(X):
        train_tempX = []
        train_tempy = []
        test_tempX = []
        test_tempy = []
        for i in range(len(train_split)):
            train_tempX.append(X[train_split[i], :])
            train_tempy.append(y[train_split[i]])
        for i in range(len(test_split)):
            test_tempX.append(X[test_split[i], :])
            test_tempy.append(y[test_split[i]])
        train_matX.append(train_tempX)
        train_maty.append(train_tempy)
        test_matX.append(np.array(test_tempX))
        test_maty.append(np.array(test_tempy))
    train_matX = np.array(train_matX)
    train_maty = np.array(train_maty)
    test_matX = np.array(test_matX)
    test_maty = np.array(test_maty)
    return (train_matX, train_maty, test_matX, test_maty)
       
def tfidf(filename, n_gram):
    dataX, datay = load_data(filename)
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1,n_gram))
    dataX = vec.fit_transform(dataX).toarray()
    return (dataX, datay)
    
def train_test(dataX, datay, alpha, n_iter, eta):
    pred_list = []
    prob_list = []
    accuracy_list = []
    train_matX, train_maty, test_matX, test_maty = cross_validate(dataX, datay)
    lr = linear_model.SGDClassifier(loss='log', penalty='l2', alpha=alpha, l1_ratio=0, n_iter=n_iter,
    epsilon=0.1, eta0=eta, class_weight='balanced')
    for split in range(len(train_maty)):
        #print len(train_matX)
        #print np.array(train_matX[split]).shape
        #print np.array(train_maty[split]).shape
        #print np.array(test_matX[split]).shape
        #print np.array(test_maty[split]).shape
        lr.fit(train_matX[split], train_maty[split])
        pred = lr.predict(test_matX[split])
        pred_list.append(pred)
        prob = lr.predict_proba(test_matX[split])
        prob_list.append(prob)
        accuracy = lr.score(test_matX[split], test_maty[split])
        accuracy_list.append(accuracy)
    preds = np.array(pred_list)
    probs = np.array(prob_list)
    return (preds, probs, accuracy_list)

def gridsearch(filename):
    n_gram_vals = [3,4,5,6,7]
    alpha_vals = [.5, .1, 1e-2, 1e-3, 1e-4 ,1e-5]
    n_iter_vals = [10, 25, 50]
    eta_vals = [1e-3, 1e-4, 1e-5]
    
    all_accuracy = []
    hyperparams = []
    for n_gram in n_gram_vals:
        X, y = tfidf(filename, n_gram)
        for alpha in alpha_vals:
            for n_iter in n_iter_vals:
                for eta in eta_vals:
                    preds, probs, accuracy_list = train_test(X, y, alpha, n_iter, eta)
                    avg_accuracy = np.mean(accuracy_list)
                    all_accuracy.append(avg_accuracy)
                    string = "Average accuracy with cross-validation with n_gram = {}, alpha = {}, n_iter = {}, eta = {} is  {}".format(n_gram, alpha, n_iter, eta, avg_accuracy)
                    hyperparams.append(string)
                    #print "Average accuracy with cross-validation with n_gram = {}, alpha = {}, n_iter = {}, eta = {} is  {}".format(n_gram, alpha, n_iter, eta, avg_accuracy)
    max_ind = np.argmax(all_accuracy)
    print "------------------"
    print "OPTIMIZED HYPERPARAMETERS:"
    print hyperparams[max_ind]
    
# gridsearch("sentiment_train.txt")
'''
X = np.array([[10,11,12], [20,21,22], [30,31,32], [40,41,41], [50,51,52], [60,61,62], [70,71,72], [80,81,82], [90,91,92], [100,101,102]])
train_mat, test_mat = cross_validate(X)
print train_mat
print test_mat
'''