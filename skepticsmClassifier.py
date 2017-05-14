#Kelly Kung
#CS 349
#5/14/17
#Classification of skeptcism

import json
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

#read data from JSON file 
def readData(filePath):
    """reads data from JSON file and separates the values into skepticism (y value) and text (x value)
    and puts it into a dictionary where the keys are the tweet ids.
    returns the dictionary"""
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
          
              

def tfidfFeaturizer(jsonData, ngram_num, mindf):
    """gets the TFIDF value for the text, using given parameter values"""
    data = []
    yVals = []
    for i in jsonData.keys():
        data.append(jsonData[i]["text"])
        yVals.append(jsonData[i]["skepticism"])
        
    vectorizer = TfidfVectorizer(min_df = mindf, lowercase = True, ngram_range = (1,ngram_num))
    trainX = vectorizer.fit(data).transform(data)


    return (trainX.toarray(), yVals)

def tfidfFeaturizerTest(jsonData, trainData, vectorizer):
    """this is the same as above, but this returns the transformed version of the test data, given 
    the vectorizer that was fitted to train data"""
    data = []
    yVals = []
    train = []
    for i in jsonData.keys():
        data.append(jsonData[i]["text"])
        yVals.append(jsonData[i]["skepticism"])
    
    for j in trainData.keys():
        train.append(trainData[j]["text"])
    
    #fit the vectorizer to the test data
    vect = vectorizer.fit(train)    
    trainX = vect.transform(data)


    return (trainX.toarray(), yVals)


def listData(jsonData):
    """puts the X and Y values from the jsonData into lists"""
    xData = []
    yData = []
    for i in jsonData.keys():
        xData.append([jsonData[i]["text"]])
        yData.append([jsonData[i]["skepticism"]])
    return (xData, yData)
       
def logisticReg(trainX, yVals, regularization, maxIter):
    """sets up the logistic regression model from sci-kit-learn given the parameters"""
    logreg = LogisticRegression(max_iter = maxIter, C = regularization, penalty = "l2", class_weight = "balanced",multi_class="multinomial",solver = "lbfgs" )
    logreg.fit(trainX, yVals)
    
    return logreg
    
def predict(testX, model):
    """given the test data and model, we predict the values"""
    predictedVals = model.predict(testX)
    return predictedVals
    
def accuracy(model, testX, testY):
    """calculates the accuracy, given the test X and test Y data and model"""
    accuracy = model.score(testX, testY)
    return accuracy
    
def trainModel(jsonData):
    """train the model using the data in order to tune the hyperparameters"""
    ngram = [2,4,5, 7]
    mindf = [5, 10, 15]
    reg = [.5, .1, .01, .001, .0001]
    maxIter = [10, 20, 50]
    accuracies = []
    param = []
    predictions = []
    probabilities = []
    
    #goes through every combination of the parameters to choose the best combo
    for i in ngram:
        for j in reg:
            for l in mindf: 
                for k in maxIter:
                    trainX, trainY = tfidfFeaturizer(jsonData, i, l) 
                    model = logisticReg(trainX, trainY, j, k)
                    #find the accuracies, predictions, and probabilities of each class
                    accuracies.append(accuracy(model, trainX, trainY))
                    param.append((i,j,k, l))
                    predictions.append(predict(trainX, model))
                    probabilities.append(model.predict_proba(trainX))
                    print "Accuracy: " + str(accuracy(model, trainX, trainY)) + " with ngram = " + str(i) + " reg = " + str(j) + " mindf = " + str(l) +  " maxIter = " + str(k) 
            
    return(accuracies, param, predictions, probabilities)

#read in the training data that was created using cross validation 
with open("trainJSONset_0.json") as fileData:
    trainData0 = json.load(fileData)
fileData.close()
with open("trainJSONset_1.json") as fileData:
    trainData1 = json.load(fileData)
fileData.close()
with open("trainJSONset_2.json") as fileData:
    trainData2 = json.load(fileData)
fileData.close()
with open("trainJSONset_3.json") as fileData:
    trainData3 = json.load(fileData)
fileData.close()
with open("trainJSONset_4.json") as fileData:
    trainData4 = json.load(fileData)
fileData.close()
with open("trainJSONset_5.json") as fileData:
    trainData5 = json.load(fileData)
fileData.close()
with open("trainJSONset_6.json") as fileData:
    trainData6 = json.load(fileData)
fileData.close()
with open("trainJSONset_7.json") as fileData:
    trainData7 = json.load(fileData)
fileData.close()
with open("trainJSONset_8.json") as fileData:
    trainData8 = json.load(fileData)
fileData.close()
with open("trainJSONset_9.json") as fileData:
    trainData9 = json.load(fileData)
fileData.close() 

#train the model           
acc0, param0, pred0, prob0 = trainModel(trainData0)
acc1, param1, pred1, prob1 = trainModel(trainData1)
acc2, param2, pred2, prob2 = trainModel(trainData2)
acc3, param3, pred3, prob3 = trainModel(trainData3)
acc4, param4, pred4, prob4 = trainModel(trainData4)
acc5, param5, pred5, prob5 = trainModel(trainData5)
acc6, param6, pred6, prob6 = trainModel(trainData6)
acc7, param7, pred7, prob7 = trainModel(trainData7)
acc8, param8, pred8, prob8 = trainModel(trainData8)           
acc9, param9, pred9, prob9 = trainModel(trainData9)

#find the parameters with the highest mean accuracy 
accuracies = np.array([acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9])
meanAcc = np.mean(accuracies, axis = 0)
maxAcc = np.argmax(meanAcc)
#look to see which of the parameters are the best
param1[maxAcc]

#end up choosing: 7, .5, 20,5)

#read in the test SemEval data sets created by cross validation
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

#set up the lists to be used 
accuraciesTest = []
predictionsTest = []
probabilitiesTest = []
yVals = []

#we go through each of the test data to predict the skepticism values using the best hyperparameters
testData = [testData0, testData1, testData2, testData3, testData4, testData5, testData6, testData7, testData8, testData9]
vectorizer = TfidfVectorizer(min_df = 5, lowercase = True, ngram_range = (1,7))
train0X, train0Y =tfidfFeaturizer(trainData0, 7, 5)
model = logisticReg(train0X, train0Y, .5, 20)

#we then find the accuracies, predictions, and probabilties
for i in range(0,len(testData)):
    test0X, test0Y = tfidfFeaturizerTest(testData[i],trainData0, vectorizer)
    yVals.append(test0Y)
    accuraciesTest.append(accuracy(model, test0X, test0Y))
    predictionsTest.append(predict(test0X, model))
    probabilitiesTest.append(model.predict_proba(test0X))
    
def returnVals():
    """return the following values: the actual Y values, the predictions from the model, and the probabilties of each class"""
    return yVals, predictionsTest, probabilitiesTest


def returnTrainandVect(maxParams):
    """returns the model, vectorizer, and a data set given the parameters"""
    ngram, reg, maxIter, mindf = maxParams
    model = logisticReg(train0X, train0Y, reg, maxIter)
    vect = TfidfVectorizer(min_df = mindf, lowercase = True, ngram_range = (1,ngram))
    return (model, vect, trainData0)
    
    
