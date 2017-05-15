#CS 349 -  Isabel D'Alessandro, Kelly Kung, Yuyu Li
#4/17
#Featurize original twitter data set using tfidf and generate true/false predictions using text only


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model 
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_approximation import RBFSampler


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


#Tfidf Vectorization 
vect=TfidfVectorizer(ngram_range=(1,7),min_df=10)
vect2=vect.fit(trainX)
vect3=vect2.transform(devX)
devXtfidf=vect3.toarray()
trainXtfidf=vect2.transform(trainX).toarray()


#Linear SVM Model 
clf=linear_model.SGDClassifier(alpha=0.0000001, average=False,class_weight='balanced',
epsilon=0.1, eta0=0.0000, fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
loss='log', n_iter=5, n_jobs=1, penalty='l2', power_t=0.5, random_state=None, 
shuffle=True, verbose=0, warm_start=False)

clf.fit(trainXtfidf,trainy)
print clf.score(devXtfidf,devy)#accuracy 


#Get most predictive features 
def show_most_informative_features(vectorizer, clf, n=100):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)

#print show_most_informative_features(vectorizer,clf)




#gridsearch for hyperparmater tuning 
def gridsearch(trainX,trainy,devX,devy):
    max_acc=0
    best_params=''
    for l in ['hinge','log','modified_huber','squared_hinge','perceptron']:#loss 
        for a in [0.1,0.01,0.001,1e-4,1e-5]: #alpha 
            for e in [0.1,0.01,0.001,0.0001]: #epsilon 
                for et in [0,0.1,0.01,0.001]: #eta
                    for n in [5,10,15]: #n_iter
                        clf=linear_model.SGDClassifier(alpha=a, average=False, class_weight='balanced',
                    epsilon=e, eta=et, fit_intercept=True, l1_ratio=0.15, learning_rate='optimal', 
                    loss=l, n_iter=n, n_jobs=1, penalty='l2',power_t=0.5, random_state=None, 
                    shuffle=True, verbose=0, warm_start=False) 
                        clf.fit(trainX,trainy)
                        acc=clf.score(devX,devy)
                        if acc>max_acc: 
                            max_acc=acc
                            best_params='loss:'+str(l)+'alpha:'+str(a)+'epsilon:'+str(e)+'eta:'+str(e)+'n_iter:'+str(n)
    print max_acc
    print best_params 
    return(max_acc,best_params)

#gridsearch(trainXtfidf,trainy,devXtfidf,devy)



