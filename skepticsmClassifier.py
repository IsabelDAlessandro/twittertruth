#Kelly Kung
#CS 349
#Classification of skeptcism

import json
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

#read data
def readData(filePath):
    with open(filePath) as fileData:
        data = json.load(fileData)
    
    dataReturn = {}
    for i in data.keys():
        for j in data[i].keys():
            if j != "truth" and j != "text":
                 if data[i][j]["skepticism"] != "comment" and data[i][j]["skepticism"] != "null":
                    dataReturn[j] = {"text": data[i][j]["text"]}
               
                    dataReturn[j]["skepticism"] = data[i][j]["skepticism"]
    
    return dataReturn
          
              
#supervised learning- sci kit learn
#do tfidf
def tfidfFeaturizer(jsonData, ngram_num, mindf):
    data = []
    yVals = []
    for i in jsonData.keys():
        data.append(jsonData[i]["text"])
        yVals.append(jsonData[i]["skepticism"])
        
    vectorizer = TfidfVectorizer(min_df = mindf, lowercase = True, ngram_range = (1,ngram_num))
    trainX = vectorizer.fit(data).transform(data)


    return (trainX.toarray(), yVals)

def tfidfFeaturizerTest(jsonData, trainData, vectorizer):
    data = []
    yVals = []
    train = []
    for i in jsonData.keys():
        data.append(jsonData[i]["text"])
        yVals.append(jsonData[i]["skepticism"])
    
    for j in trainData.keys():
        train.append(trainData[j]["text"])
    
    vect = vectorizer.fit(train)    
    trainX = vect.transform(data)


    return (trainX.toarray(), yVals)


def listData(jsonData):
    xData = []
    yData = []
    for i in jsonData.keys():
        xData.append([jsonData[i]["text"]])
        yData.append([jsonData[i]["skepticism"]])
    return (xData, yData)
       
def logisticReg(trainX, yVals, regularization, maxIter):
    logreg = LogisticRegression(max_iter = maxIter, C = regularization, penalty = "l2", class_weight = "balanced",multi_class="multinomial",solver = "lbfgs" )
    logreg.fit(trainX, yVals)
    
    return logreg
    
def predict(testX, model):
    predictedVals = model.predict(testX)
    return predictedVals
    
def accuracy(model, testX, testY):
    accuracy = model.score(testX, testY)
    return accuracy
    
def trainModel(jsonData):
    ngram = [2,4,5, 7]
    mindf = [5, 10, 15]
    reg = [.5, .1, .01, .001, .0001]
    maxIter = [10, 20, 50]
    accuracies = []
    param = []
    predictions = []
    probabilities = []
    #data = {}
    #ids = []
    #for i in jsonData:
    #    data[i] = jsonData[i]
    #    ids.append(i)
    #for j in devData:
    #    data[j] = devData[j]
    #    ids.append(j)

    for i in ngram:
        for j in reg:
            for l in mindf: 
                for k in maxIter:
                    trainX, trainY = tfidfFeaturizer(jsonData, i, l) 
                    #trainX = []
                    #trainY = []
                    #devX = []
                    #devY = []
                    #print(len(ids))    
                    #for i in range(0,len(ids)):
                    #    if ids[i] in jsonData.keys():
                    #        trainX.append(X[i])
                    #        trainY.append(Y[i])
                    #    else:
                    #        devX.append(X[i])
                    #        devY.append(Y[i])  
                                                 
                    model = logisticReg(trainX, trainY, j, k)
                    accuracies.append(accuracy(model, trainX, trainY))
                    param.append((i,j,k, l))
                    predictions.append(predict(trainX, model))
                    probabilities.append(model.predict_proba(trainX))
                    print "Accuracy: " + str(accuracy(model, trainX, trainY)) + " with ngram = " + str(i) + " reg = " + str(j) + " mindf = " + str(l) +  " maxIter = " + str(k) 
            
    return(accuracies, param, predictions, probabilities)


def returnTrainandVect(maxParams):
    ngram, reg, maxIter, mindf = maxParams
    model = logisticReg(train0X, train0Y, reg, maxIter)
    vect = TfidfVectorizer(min_df = mindf, lowercase = True, ngram_range = (1,ngram))
    return (model, vect, trainData0)
    
    

with open("trainJSONset_0.json") as fileData:
    trainData0 = json.load(fileData)
#fileData.close()
#with open("trainJSONset_1.json") as fileData:
#    trainData1 = json.load(fileData)
#fileData.close()
#with open("trainJSONset_2.json") as fileData:
#    trainData2 = json.load(fileData)
#fileData.close()
#with open("trainJSONset_3.json") as fileData:
#    trainData3 = json.load(fileData)
#fileData.close()
#with open("trainJSONset_4.json") as fileData:
#    trainData4 = json.load(fileData)
#fileData.close()
#with open("trainJSONset_5.json") as fileData:
#    trainData5 = json.load(fileData)
#fileData.close()
#with open("trainJSONset_6.json") as fileData:
#    trainData6 = json.load(fileData)
#fileData.close()
#with open("trainJSONset_7.json") as fileData:
#    trainData7 = json.load(fileData)
#fileData.close()
#with open("trainJSONset_8.json") as fileData:
#    trainData8 = json.load(fileData)
#fileData.close()
#with open("trainJSONset_9.json") as fileData:
#    trainData9 = json.load(fileData)
#fileData.close() 
#            
#acc0, param0, pred0, prob0 = trainModel(trainData0)
#acc1, param1, pred1, prob1 = trainModel(trainData1)
#acc2, param2, pred2, prob2 = trainModel(trainData2)
#acc3, param3, pred3, prob3 = trainModel(trainData3)
#acc4, param4, pred4, prob4 = trainModel(trainData4)
#acc5, param5, pred5, prob5 = trainModel(trainData5)
#acc6, param6, pred6, prob6 = trainModel(trainData6)
#acc7, param7, pred7, prob7 = trainModel(trainData7)
#acc8, param8, pred8, prob8 = trainModel(trainData8)           
#acc9, param9, pred9, prob9 = trainModel(trainData9)
#
#accuracies = np.array([acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9])
#meanAcc = np.mean(accuracies, axis = 0)
##minAcc = np.argmin(meanAcc)
##meanAcc = np.delete(meanAcc, minAcc)
##meanAcc = meanAcc[meanAcc!=meanAcc[np.argmin(meanAcc)]]
##medianAcc = np.median(meanAcc)
##maxAcc, = np.where(meanAcc==medianAcc)
#maxAcc = np.argmax(meanAcc)
#param1[maxAcc]

#end up choosing: 5, .1, 10, 3
#

with open("testJSONset_0.json") as fileData:
    testData0 = json.load(fileData)
fileData.close()
with open("testJSONset_1.json") as fileData:
    testData1 = json.load(fileData)
fileData.close()
with open("testJSONset_2.json") as fileData:
    testData2 = json.load(fileData)
fileData.close()
with open("testJSONset_3.json") as fileData:
    testData3 = json.load(fileData)
fileData.close()
with open("testJSONset_4.json") as fileData:
    testData4 = json.load(fileData)
fileData.close()
with open("testJSONset_5.json") as fileData:
    testData5 = json.load(fileData)
fileData.close()
with open("testJSONset_6.json") as fileData:
    testData6 = json.load(fileData)
fileData.close()
with open("testJSONset_7.json") as fileData:
    testData7 = json.load(fileData)
fileData.close()
with open("testJSONset_8.json") as fileData:
    testData8 = json.load(fileData)
fileData.close()
with open("testJSONset_9.json") as fileData:
    testData9 = json.load(fileData)
fileData.close()  

accuraciesTest = []
predictionsTest = []
probabilitiesTest = []
#data = {}
#ids = []
#for i in trainData0:
#    data[i] = trainData0[i]
#    ids.append(i)
#for j in testData9:
#    data[j] = testData9[j]
#    ids.append(j)
vectorizer = TfidfVectorizer(min_df = 5, lowercase = True, ngram_range = (1,7))
train0X, train0Y =tfidfFeaturizer(trainData0, 7, 5)
test0X, test0Y = tfidfFeaturizerTest(testData9,trainData0, vectorizer)
#mindf = 5, ngram = 7, .5 = reg, max iter = 20

#train0X = []
#train0Y = []
#test0X = []
#test0Y = []
#for i in range(0,len(X)):
#    if ids[i] in trainData0.keys():
#        train0X.append(X[i])
#        train0Y.append(Y[i])
#    if ids[i] in testData9.keys():
#        test0X.append(X[i])
#        test0Y.append(Y[i])

model = logisticReg(train0X, train0Y, .5, 20)
accuraciesTest.append(accuracy(model, test0X, test0Y))
predictionsTest.append(predict(test0X, model))
probabilitiesTest.append(model.predict_proba(test0X))
