


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model 
from sklearn.linear_model import LogisticRegression
#LBFGS
#from numpy import genfromtxt
#my_data = genfromtxt('trainingData2.csv', delimiter=',')
#my_data.shape
#print my_data[1]

def featurizetext(inputfile):
    
    df = pd.read_csv(inputfile, header = 1,usecols=[0,2,4])
    numpyMatrix = df.as_matrix()
    storydict={}
    truthdict={}
    for tweet in numpyMatrix: 
        if tweet[0] not in storydict: 
            storydict[tweet[0]]=[tweet[2]]#add tweet to story dictionary 
            truthdict[tweet[0]]=[tweet[1]]#truth value
        else: 
            storydict[tweet[0]].append(tweet[2])#add tweet to dictionary 
    for story in storydict: 
        storydict[story]=[''.join(storydict[story])]#concetenate all tweets for each story 
#join the dictionaries, truth label is last element in list 
    for story in truthdict: 
        storydict[story].append(truthdict[story])

    Xdat=[]
    labels=[]
    for elem in storydict.values():
        labels.append(elem[1])
        Xdat.append(elem[0])
    Xdat=np.asarray(Xdat)
    labels=np.asarray(labels)
    return(Xdat,labels)
    

train_file = "trainingData2.csv"
dev_file = "dev.csv"
test_file="test.csv"

trainX,trainy=featurizetext(train_file)
devX,devy=featurizetext(dev_file)
testX,testy=featurizetext(test_file)


#Tfidf 
#vectorizer = TfidfVectorizer(max_features=1400,ngram_range=(1,3),min_df=10)
vectorizer = TfidfVectorizer(max_features=140000)
trainXtfidf = vectorizer.fit_transform(trainX).toarray()
devXtfidf=vectorizer.fit_transform(devX).toarray()
testXtfidf=vectorizer.fit_transform(testX).toarray()


#count vectorizer 
#vectorizer = CountVectorizer()
#trainX = vectorizer.fit_transform(Xdata).toarray()
#devX=vectorizer.transform(dev_text).toarray()
#textX=vectorizer.transform(test_text).toarray()

#model = LogisticRegression(max_iter=200,class_weight='balanced')
#model.fit(trainXtfidf,trainy)
#accuracy = model.score(devXtfidf, devy)
#print 'Classification accuracy', accuracy

clf=linear_model.SGDClassifier(alpha=0.0000001, average=False, 
class_weight='balanced', epsilon=0.001, eta0=0.0000, fit_intercept=True,
l1_ratio=0.15, learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1,
penalty='l2', power_t=0.5, random_state=None, shuffle=True, verbose=0, 
warm_start=False) 
clf.fit(trainXtfidf,trainy)
print clf.score(devXtfidf,devy)

#gridsearch: 
#for l in ['hinge','log','modified_huber','squared_hinge','perceptron']:#loss 
#    for a in [0.1,0.01,0.001,1e-4,1e-5]: #alpha 
#        for e in [0.1,0.01,0.001,0.0001]: #epsilon 
#            for et in [0,0.1,0.01,0.001]: #eta
#                for n in [5,10,15]: #n_iter
#                    clf=linear_model.SGDClassifier(alpha=a, average=False, class_weight='balanced',
#                    epsilon=e, eta=et, fit_intercept=True, l1_ratio=0.15, learning_rate='optimal', 
#                    loss=l, n_iter=n, n_jobs=1, penalty='l2',power_t=0.5, random_state=None, 
#                    shuffle=True, verbose=0, warm_start=False) 



