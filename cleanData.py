import csv
import numpy as np
import random
import string

#Kelly Kung
#4/2/17
#read the data and clean it up

#read in the data

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
    
    
def findTrainingData(fileObject):
    """does a stratified sampling of tweets from each story so we have examples
    covering everything"""
    storyids = [fileObject[i][0] for i in range(0, len(fileObject))]
    uniqueStories = set(storyids)
    trainingTweets = []
    
    random.seed(1234)
    trainingStories = random.sample(uniqueStories, (len(uniqueStories)/3))
    for i in range(0, len(fileObject)):
        if fileObject[i][0] in trainingStories:
            trainingTweets.append(fileObject[i])
    
    return trainingTweets
    
def setUpData(fileName, outfilename):
    """sets up the data and then writes it into a csv file"""
    csvData = load_csv(fileName)
    english = removeNonEnglish(csvData)
    train = findTrainingData(english)
    storyId, tweetId, status, skepticism, tweet, time, user, retweet = readTweets(train)
    
    with open(outfilename, 'wb') as doc:
        csvwriter = csv.writer(doc, dialect = "excel")
        #this writes the col names
        csvwriter.writerow([csvData[0][0], csvData[0][3], csvData[0][2], csvData[0][6], csvData[0][5], csvData[0][4], csvData[0][7], csvData[0][8]])
        for i in range(0, len(storyId)):
            csvwriter.writerow([storyId[i], tweetId[i], status[i], skepticism[i], tweet[i], time[i], user[i], retweet[i]])
    doc.close()
    return (tweetId, storyId)

def getDevandTest(storyId, fileName, devFile, testFile):
    """write the dev and testing data by breaking the remaining data (after testing) and breaking it into 2 parts"""
    loadData = load_csv(fileName)
    fileObject = removeNonEnglish(loadData)
  
    storyids = [fileObject[i][0] for i in range(0, len(fileObject))]
    uniqueId = list(set(storyids))
    
    uniqueStoryId = set(storyId)
    remainingId = []
    for i in range(0, len(uniqueId)):
        if uniqueId[i] not in uniqueStoryId:
            remainingId.append(uniqueId[i])
    
    #randomly select 1/2 of the data as development set
    random.seed(1234)
    devStories = random.sample(remainingId, len(remainingId)/2)
    
    #the remaining is test data
    testStories = []
    for i in range(0, len(remainingId)):
        if remainingId[i] not in devStories:
            testStories.append(remainingId[i])
            
    devData = []
    testData = []
    
    for i in range(0, len(fileObject)):
        if fileObject[i][0] in devStories:
            devData.append(fileObject[i])
        elif fileObject[i][0] in testStories:
            testData.append(fileObject[i])

    devstory, devtweetId, devstatus, devskept, devtweet, devtime, devuser, devretweet = readTweets(devData)
    teststory, testtweetId, teststatus, testskept, testtweet, testtime, testuser, testretweet = readTweets(testData)
    
    #writes out the data sets for testing
    with open(devFile, 'wb') as dev:
        csvDevwriter = csv.writer(dev, dialect = "excel")
        csvDevwriter.writerow([fileObject[0][0], fileObject[0][3], fileObject[0][2], fileObject[0][6], fileObject[0][5], fileObject[0][4], fileObject[0][7], fileObject[0][8]])
        for i in range(0, len(devData)):
            csvDevwriter.writerow([devstory[i], devtweetId[i], devstatus[i], devskept[i], devtweet[i], devtime[i], devuser[i], devretweet[i]])
    dev.close()
    
    with open(testFile, 'wb') as test:
        testWriter = csv.writer(test, dialect = "excel")
        testWriter.writerow([fileObject[0][0], fileObject[0][3], fileObject[0][2], fileObject[0][6], fileObject[0][5], fileObject[0][4], fileObject[0][7], fileObject[0][8]])
        for i in range(0, len(testData)):
            testWriter.writerow([teststory[i], testtweetId[i], teststatus[i], testskept[i], testtweet[i], testtime[i], testuser[i], testretweet[i]])
    test.close()
    
    return (devstory, devtweetId, teststory, testtweetId)
    
    