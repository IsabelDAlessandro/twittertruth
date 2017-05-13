import numpy as np
import pandas as pd
import random 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model 
from sklearn.linear_model import LogisticRegression

def crossValidate(file1,file2,file3):
    df1 = pd.read_csv(file1, header = 1,usecols=[0,2,4])
    numpyMatrix1 = df1.as_matrix()
    df2 = pd.read_csv(file2, header = 1,usecols=[0,2,4])
    numpyMatrix2 = df2.as_matrix()
    df3 = pd.read_csv(file3, header = 1,usecols=[0,2,4])
    numpyMatrix3 = df3.as_matrix()
    numpyMatrix=np.vstack((numpyMatrix1,numpyMatrix2,numpyMatrix3))
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
    for story in truthdict: 
        storydict[story].append(truthdict[story])
        
    data=[]
    for elem in storydict.values():
        data.append(elem)
    random.shuffle(data)
    
    trainX=[]
    trainy=[]
    devX=[]
    devy=[]
    testX=[]
    testy=[]
    
    count=0
    for elem in range(1,int(round(0.8*len(data)))):
        trainX.append(data[elem][0])
        trainy.append(data[elem][1])
        count+=1
    for elem in range(count,int(count+round(0.1*len(data)))):
        devX.append(data[elem][0])
        devy.append(data[elem][1])
        count+=1
    for elem in range(count,len(data)):
        testX.append(data[elem][0])
        testy.append(data[elem][1])
    
    trainX=np.asarray(trainX)
    trainy=np.asarray(trainy)
    devX=np.asarray(devX)
    devy=np.asarray(devy)
    testX=np.asarray(testX)
    testy=np.asarray(testy) 
    
    return(trainX,trainy,devX,devy,testX,testy) 
    

trainX,trainy,devX,devy,testX,testy=crossValidate('trainingData2.csv','dev.csv','test.csv')

vectorizer = TfidfVectorizer(max_features=30000)
trainXtfidf = vectorizer.fit_transform(trainX).toarray()
devXtfidf=vectorizer.fit_transform(devX).toarray()
testXtfidf=vectorizer.fit_transform(testX).toarray()

clf=linear_model.SGDClassifier(alpha=0.0000001, average=False, 
class_weight='balanced', epsilon=0.001, eta0=0.0000, fit_intercept=True,
l1_ratio=0.15, learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1,
penalty='l2', power_t=0.5, random_state=None, shuffle=True, verbose=0, 
warm_start=False) 
clf.fit(trainXtfidf,trainy)
print clf.score(devXtfidf,devy)