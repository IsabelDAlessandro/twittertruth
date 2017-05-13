import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model 
from sklearn.linear_model import LogisticRegression

def featurizetext(inputfile):
    df = pd.read_csv(inputfile,dtype='str',usecols=[0,5])
    Xdat=[]
    labels=[]
    numpyMatrix = df.as_matrix()
    for elem in numpyMatrix: 
        Xdat.append(elem[1])
        labels.append(elem[0])
    Xdat=np.asarray(Xdat)
    labels=np.asarray(labels)
    return (Xdat,labels)

trainX,trainy=featurizetext('semtrain.csv')
testX,testy=featurizetext('semtest.csv')
print 'testX'+str(testX[1])

vectorizer = TfidfVectorizer(decode_error='ignore') #(max_features=140000)
trainXtfidf = vectorizer.fit_transform(trainX).toarray()
testXtfidf=vectorizer.fit_transform(testX).toarray()
print 'tfidf'

clf=linear_model.SGDClassifier(alpha=0.0000001, average=False, class_weight='balanced',
epsilon=0.1, eta0=0.0000, fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
loss='hinge', n_iter=5, n_jobs=1, penalty='l2', power_t=0.5, random_state=None, 
shuffle=True, verbose=0, warm_start=False) 
clf.fit(trainXtfidf,trainy)
print('done with model')
print clf.score(testXtfidf,testy)
