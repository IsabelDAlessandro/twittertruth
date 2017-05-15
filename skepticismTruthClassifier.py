#Kelly Kung
#5/14/17
#run skepticism values on classifier for actual truth value
import csv
import numpy as np 
import random
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

def load_csv(filename):
    """read data in CSV format"""
    with open(filename, 'rU') as csvfile:
        csvreader = csv.reader(csvfile, delimiter = ',')
    
        data = []
        next(csvreader, None)
        for row in csvreader:
            data.append(row)
    return data
    
def splitData(data):
    """splits the data into training and testing randomly. splits it so it's 90% training and 10% testing"""
    random.seed(5)
    storyIds = set([row[1] for row in data])
    trainingStories = random.sample(storyIds, 45)
    
    train = []
    test = []
    for i in range(0, len(data)):
        if data[i][1] in trainingStories:
            train.append(data[i])
        else:
            test.append(data[i])
    return(train, test)
    
def getXandYVals(data):
    """this separtes a given data set into X and Y values"""
    X = []
    Y = []
    
    for i in range(0, len(data)):
        X.append([data[i][5], data[i][6], data[i][7]])
        Y.append(data[i][4])
    return (X,Y)


def skeptClassifierforTruth(trainX, trainY, maxIter, regularization):
    """sets up the logistic regression model"""
    logreg = LogisticRegression(max_iter = maxIter, C = regularization, penalty = "l2", class_weight = "balanced",multi_class="multinomial",solver = "lbfgs" )
    logreg.fit(trainX, trainY)
    return logreg
    
def predict(testX, model):
    """given the test data and model, we predict the values"""
    predictedVals = model.predict(testX)
    return predictedVals
    
def accuracy(model, testX, testY):
    """calculates the accuracy, given the test X and test Y data and model"""
    accuracy = model.score(testX, testY)
    return accuracy
    
def trainModel(trainX, Y):
    """trains the model given different values for the parameters"""
    reg = [.5, .1, .01, .001, .0001]
    maxIter = [10, 20, 50]
    accuracies = []
    param = []
    predictions = []
    probabilities = []
    
    
    for j in reg:
        for k in maxIter:         
                model = skeptClassifierforTruth(trainX, Y, k, j)
                #find the accuracies, predictions, and probabilities of each class
                accuracies.append(accuracy(model, trainX, Y))
                param.append((k,j))
                predictions.append(predict(trainX, model))
                probabilities.append(model.predict_proba(trainX))
                print "Accuracy: " + str(accuracy(model, trainX, Y)) + " with reg = " + str(j) +  " maxIter = " + str(k) 
            
    return(accuracies, param, predictions, probabilities, trainX, Y)

               
def findMaxValsandTest():
    """trains the model and uses the model to predict values for the test data. then find accuracies"""
    data = load_csv("pred_skeptVals_all.csv")
    train, test = splitData(data)
    
    X,Y = getXandYVals(train)
    trainX = []
    for i in range(len(X)):
        trainX.append([float(x) for x in X[i]])
    
    acc, param, predict, prob, trainX, Y = trainModel(trainX, Y)
    maxAcc = np.argmax(acc)
    print param[maxAcc]
    
    testX, testY = getXandYVals(test)
    testXVals = []
    for i in range(0, len(testX)):
        testXVals.append([float(x) for x in testX[i]])
    testXVals = np.array(testXVals)
     
    model = skeptClassifierforTruth(trainX, Y, param[maxAcc][0], param[maxAcc][1])
    acc = accuracy(model, testXVals, testY)
    print "accuracy: " + str(acc)
    predictions = model.predict(testXVals)
    probabilities = model.predict_proba(testXVals)
    
    #see which values are most indicative for true and false values
    predictionTestVals = []
    for i in range(0, len(testXVals)):
        maxindex = np.where([testXVals[i] == (max(testXVals[i]))])
        if maxindex[1][0] == 0:
            predictionTestVals.append("deny")
        elif maxindex[1][0] == 1:
            predictionTestVals.append("query")
        else:
            predictionTestVals.append("support")
            
    indicativeF = np.zeros(3) 
    indicativeT = np.zeros(3) 
    for i in range(0, len(predictions)):
        if predictions[i] == "FALSE":
            if predictionTestVals[i] == "deny":
                indicativeF[0] +=1
            elif predictionTestVals[i] == "query":
                indicativeF[1] +=1
            else:
                indicativeF[2] +=1
    
    for i in range(0, len(predictions)):
        if predictions[i] == "TRUE":
            if predictionTestVals[i] == "deny":
                indicativeT[0] +=1
            elif predictionTestVals[i] == "query":
                indicativeT[1] +=1
            else:
                indicativeT[2] +=1
    return(acc, predictions, probabilities, indicativeF, indicativeT)
    
    