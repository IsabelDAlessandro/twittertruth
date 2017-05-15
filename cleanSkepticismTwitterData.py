
#CS 349 -  Isabel D'Alessandro, Kelly Kung, Yuyu Li
#Cleans data obtained from RumorEval for use in skepticism classifier

from skepticsmClassifier import *
import skepticsmClassifier
from sklearn.model_selection import KFold
import csv
import random 

def load_csv(filename):
    """read data in CSV format"""
    with open(filename) as csvfile:
        csvreader = csv.reader(csvfile, delimiter = ',')
    
        data = []
        for row in csvreader:
            data.append(row)
    return data
def removeNonEnglish(fileObject):
    """this takes a csv fileObject and only keeps the rows 
    that has English tweets"""
    returnObject = []

    #start at 1 because the first is column names 
    #end at 881881 because there's nothing in the last 2 indices
    for i in range(1, len(fileObject)-2):
        if fileObject[i][11] == "English":
            returnObject.append(fileObject[i])
            
    return returnObject
       
def readTweets(fileObjectEnglish):
    """this takes in a csv fileObject with only English tweets
    and returns five arrays: story id, tweet id, t/f, skepticism of user, and tweets"""
    storyIds = []
    tweetIds = []
    status = []
    skepticism = []
    tweets = []
    time = []
    user = []
    retweet = []
    
    
    for i in range(0, len(fileObjectEnglish)):
        storyIds.append(fileObjectEnglish[i][0])
        tweetIds.append(fileObjectEnglish[i][3])
        status.append(fileObjectEnglish[i][2])
        skepticism.append(fileObjectEnglish[i][6])
        tweets.append(fileObjectEnglish[i][5])
        time.append(fileObjectEnglish[i][4])
        user.append(fileObjectEnglish[i][7])
        retweet.append(fileObjectEnglish[i][8])
    
    return (storyIds, tweetIds, status, skepticism, tweets, time, user, retweet)
    
def setUpData(fileName, outfilename):
    """this function writes the csv files out according to the columns and values"""
    csvData = load_csv(fileName)
    english = removeNonEnglish(csvData)
    storyId, tweetId, status, skepticism, tweet, time, user, retweet = readTweets(english)
    
    with open(outfilename, 'wb') as doc:
        csvwriter = csv.writer(doc, dialect = "excel")
        #this writes the col names
        csvwriter.writerow([csvData[0][0], csvData[0][3], csvData[0][2], csvData[0][6], csvData[0][5], csvData[0][4], csvData[0][7], csvData[0][8]])
        for i in range(0, len(storyId)):
            csvwriter.writerow([storyId[i], tweetId[i], status[i], skepticism[i], tweet[i], time[i], user[i], retweet[i]])
    doc.close()
    return (tweetId, storyId)


def tfidfVect(trainText, ngram_num, mindf):
    """sets up the tfidf vectorizer given the parameters""" 
    vectorizer = TfidfVectorizer(min_df = mindf, lowercase = True, ngram_range = (1,ngram_num))
    trainX = vectorizer.fit(trainText)
    return trainX.toarray()
    
def logisticReg(trainX, yVals, regularization, maxIter):
    """sets up the logistic regression model"""
    logreg = LogisticRegression(max_iter = maxIter, C = regularization, penalty = "l2", class_weight = "balanced", multi_class="multinomial",solver = "lbfgs" )
    logreg.fit(trainX, yVals)   
    return logreg

def tfidfFeaturizerTest(jsonData, trainData, vectorizer):
    """featurizes the test data given the training data and vectorizer"""
    data = []
    train = []
    for i in range(0,len(jsonData)):
        data.append(jsonData[i])
    
    for j in trainData.keys():
        train.append(trainData[j]["text"])
    
    vect = vectorizer.fit(train)    
    trainX = vect.transform(data)
    return (trainX.toarray())
        


def writeCSV(text_param, storyId_param, textId_param, skept_param, truth_param, predSkept_param, outfilename):
    """writes the csv file with the probabilities of each skepticism class"""
    with open(outfilename, 'wb') as doc:
        csvwriter = csv.writer(doc, dialect = "excel")
        #this writes the col names
        csvwriter.writerow(["text", "storyId", "tweetId", "skept true", "truth", "prob deny", "prob query", "prob support"])
        
        for i in range(0, len(storyId_param)):
            csvwriter.writerow([text_param[i], storyId_param[i], textId_param[i], skept_param[i], truth_param[i], predSkept_param[i][0], predSkept_param[i][1], predSkept_param[i][2]])
    doc.close()
    return (textId_param, storyId_param, predSkept_param)


allCSV = load_csv("eng_refined_twitter_data.csv")
data0 = allCSV
print len(data0)


text = []
skeptVals = []
truthVals = []
storyId = []
tweetId = []

#stores the corresponding values into the different lists
for i in range(0, len(data0)):
    tweetId.append(data0[i][1])
    storyId.append(data0[i][0])
    text.append(data0[i][4])
    skeptVals.append(data0[i][3])
    truthVals.append(data0[i][2])


#we get the model, vectorizer, and training data needed    
model, vect, trainX = skepticsmClassifier.returnTrainandVect((7, .5, 20,5))
#vectorize the test data
testX = tfidfFeaturizerTest(text,trainX, vect)
#make predictions of the test data, given the model 
predictionsTest = (predict(testX, model))
#get the probabilities of each class 
probabilitiesTest = (model.predict_proba(testX))
#write the csv file of all the predictions along with the text and ids
tweet_id0, story_id0, pred_0 = writeCSV(text, storyId, tweetId, skeptVals, truthVals, probabilitiesTest, "pred_skeptVals_all.csv")

