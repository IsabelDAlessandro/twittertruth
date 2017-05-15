'''
Yuyu Li, April 2017
Sentiment analysis portion. Based off of SemEval Task 4E, to classify Tweets into 4 different categories:
very negative, negative, positive, very positive on a 5-point scale excluding 3, since this was the
neutral class (1,2,4,5). 
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

''' obtain the tweet IDs and labels to make the X and y matrices, respectively.
use the Twitter API to get the text of the tweets and return the X and y. '''
def load_data(filename):
    error_count = 0
    matrix = []
    with open(filename) as txt_data:
        for line in txt_data:
            row = re.split(r'\t+', line.rstrip())
            matrix.append(row)
    # print matrix
    # matrix[:][0] = api.get_status(matrix[:][0])
    matrix = np.array(matrix)
    dataX = np.array(matrix[:, 0])
    try:
        for i in range(0, dataX.shape[0]):
            dataX[i] = api.get_status(dataX[i])
    except:
        error_count += 1
    datay = np.array(matrix[:, 2])
    #print error_count
    return (dataX, datay)

def filtered(dataX, datay):
    # count = 0
    # total = datay.shape[0]
    X = []
    y = []
    for row in range(datay.shape[0]):
        if datay[row] != '3':
            X.append(dataX[row])
            y.append(datay[row])
    X = np.array(X)
    y = np.array(y)
    # print y
    return (X,y)
    
''' divide up the data into 10 training and testing sets since the data set is
relatively small. save these into training and testing matrices where each row 
corresponds to the indices for a given cross-validation split. '''
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
            train_tempX.append(X[train_split[i]])
            train_tempy.append(y[train_split[i]])
        for i in range(len(test_split)):
            test_tempX.append(X[test_split[i]])
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

'''              
def tfidf(filename, n_gram):
    pre_X, pre_y = load_data(filename)
    # print (pre_X.shape, pre_y.shape)
    dataX, datay = filtered(pre_X, pre_y)
    # print (dataX.shape, datay.shape)
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1,n_gram))
    dataX = vec.fit_transform(dataX).toarray()
    # print dataX.shape
    return (dataX, datay)
'''

''' featurize the text using TFIDF values.'''
def tfidf(n_gram, trainX, testX):
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1,n_gram))
    vec = vec.fit(trainX)
    trainX = vec.transform(trainX).toarray()
    testX = vec.transform(testX).toarray()
    return (trainX, testX)

'''
def meansq_accuracy(y, pred):
    num_test = y.shape[0]
    error_sum = 0
    for i in range(y.shape[0]):
        if ((pred[i] == y[i]) or (pred[i] == '5' and y[i] == '4') or (pred[i] == '4' and y[i] == '5')
        or (pred[i] == '1' and y[i] == '2') or (pred[i] == '2' and y[i] == '1')):
            error_sum = error_sum
        else:
            error_sum += 1
    msa = 1-((1/float(num_test)) * error_sum)
    return msa
'''

''' using logistic regression, train and test the model for each cross-validation
split and obtain predictions, probabilities, and accuracy values for each. '''
def train_test(dataX, datay, n_gram, alpha, n_iter, eta):
    pred_list = []
    prob_list = []
    accuracy_list = []
    train_matX, train_maty, test_matX, test_maty = cross_validate(dataX, datay)
    lr = linear_model.SGDClassifier(loss='log', penalty='l2', alpha=alpha, l1_ratio=0, n_iter=n_iter,
    epsilon=0.1, eta0=eta, class_weight='balanced')
    for split in range(len(train_maty)):
        tfidf_trainX, tfidf_testX = tfidf(n_gram, train_matX[split], test_matX[split])
        lr.fit(tfidf_trainX, train_maty[split])
        pred = lr.predict(tfidf_testX)
        #print test_matX[split]
        pred_list.append(pred)
        prob = lr.predict_proba(tfidf_testX)
        prob_list.append(prob)
        #print prob
        accuracy = lr.score(tfidf_testX, test_maty[split])
        accuracy_list.append(accuracy)
    preds = np.array(pred_list)
    print "predictions for this hyperparameter combo: {}".format(preds)
    probs = np.array(prob_list)
    print "probabilities for this hyperparameter combo: {}".format(probs)
    return (preds, probs, accuracy_list)

''' tune hyperparameters using gridsearch and obtain the combination used to get 
the highest accuracy. '''
def gridsearch(filename):
    n_gram_vals = [3,4,5,6,7]
    alpha_vals = [.5, .1, 1e-2, 1e-3, 1e-4 ,1e-5]
    n_iter_vals = [10, 25, 50]
    eta_vals = [1e-3, 1e-4, 1e-5]
    
    pre_X, pre_y = load_data(filename)
    X, y = filtered(pre_X, pre_y)
    
    all_accuracy = []
    hyperparams = []
    for n_gram in n_gram_vals:
        # X, y = tfidf(filename, n_gram)
        for alpha in alpha_vals:
            for n_iter in n_iter_vals:
                for eta in eta_vals:
                    preds, probs, accuracy_list = train_test(X, y, n_gram, alpha, n_iter, eta)
                    avg_accuracy = np.mean(accuracy_list)
                    all_accuracy.append(avg_accuracy)
                    string = "Average accuracy with cross-validation for n_gram = {}, alpha = {}, n_iter = {}, eta = {} is  {}".format(n_gram, alpha, n_iter, eta, avg_accuracy)
                    hyperparams.append([n_gram, alpha, n_iter, eta])
                    print string
    max_ind = np.argmax(all_accuracy)
    print "------------------"
    print "OPTIMIZED HYPERPARAMETERS:"
    print hyperparams[max_ind] 
    print "OPTIMIZED ACCURACY:"
    print all_accuracy[max_ind]
    return (hyperparams[max_ind], all_accuracy[max_ind])

gridsearch("sentiment_train.txt")
# hyperparams, accuracy = gridsearch("sentiment_train.txt")
