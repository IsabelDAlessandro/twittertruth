#Kelly Kung
#5/9/17
#break up the Twitter Data and run the skepticism prediction 
from skepticsmClassifier import *
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

print "test"    
#allCSV = load_csv("eng_refined_twitter_data.csv")
random.seed(1234)
allCSV = load_csv("trainingData2.csv")
allCSV = random.sample(allCSV, 10000) 
print len(allCSV)
print allCSV[1]

text = []
skeptVals = []
truthVals = []

for i in range(0, len(allCSV)):
    text.append(allCSV[i][4])
    skeptVals.append(allCSV[i][3])
    truthVals.append(allCSV[i][2])

print "done with reading data"


def tfidfVect(trainText, ngram_num, mindf):
    vectorizer = TfidfVectorizer(min_df = mindf, lowercase = True, ngram_range = (1,ngram_num))
    trainX = vectorizer.fit_transform(trainText)
    return trainX.toarray()
    
def logisticReg(trainX, yVals, regularization, maxIter):
    logreg = LogisticRegression(max_iter = maxIter, C = regularization, penalty = "l2", class_weight = "balanced", multi_class="multinomial",solver = "lbfgs" )
    logreg.fit(trainX, yVals)   
    return logreg
    
vecText = tfidfVect(text, 5, 3)
logModel = logisticReg(vecText, skeptVals, .1, 10)
predictions = predict(vecText, logisticReg)
probabilities = logisticReg.predict_proba(vecText)

    

#storyId = [tweet[0] for tweet in allCSV]
#storyId = list(set(storyId))
#print len(storyId)
#train = []
#test = []
#k_fold = KFold(10)
#for train_index, test_index in k_fold.split(storyId):
#    train.append(train_index)
#    test.append(test_index)
#
#trainData = []
#testData = []
#for i in train:
#    tempData = [storyId[index] for index in i]
#    tempTrain = []
#    tempTest = []
#    for j in range(0, len(allCSV)):
#        if allCSV[j][0] in tempData:
#            tempTrain.append(allCSV[j])
#        else:
#            tempTest.append(allCSV[j])
#    trainData.append(tempTrain)
#    testData.append(tempTest)
#print "done separating"     
#for i in range(0,len(trainData)):
#    with open("twitterTrain_" + str(i) + ".csv", 'wb') as doc:
#        testWriter = csv.writer(test, dialect = "excel")
#        #testWriter.writerow([trainData[0][0], trainData[0][3], trainData[0][2], trainData[0][6], trainData[0][5], trainData[0][4], trainData[0][7], trainData[0][8]])
#        for j in range(0, len(trainData[i])):
#            testWriter.writerow(trainData[i][j])
#doc.close()
#
#print "done training"
#
#for i in range(0, len(testData)):
#    with open("twitterTest_" + str(i) + ".csv", 'wb') as doc:
#        testWriter = csv.writer(test, dialect = "excel")
#        for j in range(0, len(testData[i])):
#            testWriter.writerow(testData[i][j])
#doc.close()
#        
#   