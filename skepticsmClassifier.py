#Kelly Kung
#CS 349
#Classification of skeptcism

import json
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

#read data
def readData(filePath):
    with open(filePath) as fileData:
        data = json.load(fileData)
    
    dataReturn = {}
    for i in data.keys():
        for j in data[i].keys():
            if j != "truth" and j != "text":
                dataReturn[j] = {"text": data[i][j]["text"]}
                dataReturn[j]["skepticism"] = data[i][j]["skepticism"]
    
    return dataReturn
          
              
#supervised learning- sci kit learn
#do tfidf
def tfidfFeaturizer(jsonData, ngram_num):
    data = []
    yVals = []
    for i in jsonData.keys():
        data.append(jsonData[i]["text"])
        yVals.append(jsonData[i]["skepticism"])
        
    vectorizer = TfidfVectorizer(min_df = 10, lowercase = True, ngram_range = (1,ngram_num))
    trainX = vectorizer.fit_transform(data)

    return (trainX.toarray(), yVals)

def listData(jsonData):
    xData = []
    yData = []
    for i in jsonData.keys():
        xData.append([jsonData[i]["text"]])
        yData.append([jsonData[i]["skepticism"]])
    return (xData, yData)
       
def logisticReg(trainX, yVals, regularization, maxIter):
    logreg = LogisticRegression(max_iter = maxIter, C = regularization)
    logreg.fit(trainX, yVals)
    
    return logreg
    
def predict(testX, model):
    predictedVals = model.predict(testX)
    return predictedVals
    
def accuracy(model, testX, testY):
    accuracy = model.score(testX, testY)
    return accuracy
    
def trainModel(trainfilePath):
    jsonData = readData(trainfilePath)
    ngram = [2,4,5,10]
    reg = [.001, .01, .1]
    maxIter = [10, 50]
    accuracies = []
    param = []
    predictions = []
    for i in ngram:
        for j in reg:
            for k in maxIter:
                trainX, trainY = tfidfFeaturizer(jsonData, i)
                model = logisticReg(trainX, trainY, j, k)
                accuracies.append(accuracy(model, trainX, trainY))
                param.append((i,j,k))
                predictions.append(predict(trainX, model))
                print "Accuracy: " + str(accuracy(model, trainX, trainY)) + " with ngram = " + str(i) + " reg = " + str(j) + " maxIter = " + str(k)
    return(accuracies, param, predictions)
    
    
            
    