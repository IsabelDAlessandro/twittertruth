#Kelly Kung
#5/9/17
#break up the Twitter Data and run the skepticism prediction 
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
data0 = random.sample(allCSV, 10000) 
print len(data0)
print data0[1]

text = []
skeptVals = []
truthVals = []
storyId = []
tweetId = []

for i in range(0, len(data0)):
    tweetId.append(data0[i][1])
    storyId.append(data0[i][0])
    text.append(data0[i][4])
    skeptVals.append(data0[i][3])
    truthVals.append(data0[i][2])

print "done with reading data"


def tfidfVect(trainText, ngram_num, mindf):
    vectorizer = TfidfVectorizer(min_df = mindf, lowercase = True, ngram_range = (1,ngram_num))
    trainX = vectorizer.fit(trainText)
    return trainX.toarray()
    
def logisticReg(trainX, yVals, regularization, maxIter):
    logreg = LogisticRegression(max_iter = maxIter, C = regularization, penalty = "l2", class_weight = "balanced", multi_class="multinomial",solver = "lbfgs" )
    logreg.fit(trainX, yVals)   
    return logreg

def tfidfFeaturizerTest(jsonData, trainData, vectorizer):
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
    
    with open(outfilename, 'wb') as doc:
        csvwriter = csv.writer(doc, dialect = "excel")
        #this writes the col names
        csvwriter.writerow(["text", "story ID", "text ID", "skepticism", "truth", "prob deny", "prob query", "prob support"])
        for i in range(0, len(storyId_param)):
            csvwriter.writerow([text_param[i], storyId_param[i], textId_param[i], skept_param[i], truth_param[i], predSkept_param[i][0], predSkept_param[i][1], predSkept_param[i][2]])
    doc.close()
    return (textId_param, storyId_param, predSkept_param)
    
model, vect, trainX = skepticsmClassifier.returnTrainandVect((7, .5, 20,5))
testX = tfidfFeaturizerTest(text,trainX, vect)
predictionsTest = (predict(testX, model))
probabilitiesTest = (model.predict_proba(testX))
#tweet_id0, story_id0, pred_0 = writeCSV(text, storyId, tweetId, skeptVals, truthVals, probabilitiesTest, "pred_skeptVals0.csv")

temp1 = []
for i in range(0, len(allCSV)):
    if allCSV[i][1] not in tweetId:
        temp1.append(allCSV[i])

random.seed(1234)
data1 = random.sample(temp1, 10000)

text1 = []
skeptVals1 = []
truthVals1 = []
storyId1 = []
tweetId1 = []

for i in range(0, len(data1)):
    tweetId1.append(data1[i][1])
    storyId1.append(data1[i][0])
    text1.append(data1[i][4])
    skeptVals1.append(data1[i][3])
    truthVals1.append(data1[i][2])

model1, vect1, trainX1 = skepticsmClassifier.returnTrainandVect((7, .5, 20,5))
testX1 = tfidfFeaturizerTest(text1,trainX1, vect1)
predictionsTest1 = (predict(testX1, model1))
probabilitiesTest1 = (model1.predict_proba(testX1))
#tweet_id1, story_id1, pred_1 = writeCSV(text1, storyId1, tweetId1, skeptVals1, truthVals1, probabilitiesTest1, "pred_skeptVals1.csv")

temp2 = []
for i in range(0, len(allCSV)):
    if allCSV[i][1] not in tweetId and allCSV[i][1] not in tweetId1:
        temp2.append(allCSV[i])

random.seed(1234)
data2 = random.sample(temp2, 10000)

text2 = []
skeptVals2 = []
truthVals2 = []
storyId2 = []
tweetId2 = []

for i in range(0, len(data2)):
    tweetId2.append(data2[i][1])
    storyId2.append(data2[i][0])
    text2.append(data2[i][4])
    skeptVals2.append(data2[i][3])
    truthVals2.append(data2[i][2])

model2, vect2, trainX2 = skepticsmClassifier.returnTrainandVect((7, .5, 20,5))
testX2 = tfidfFeaturizerTest(text2,trainX2, vect2)
predictionsTest2 = (predict(testX2, model2))
probabilitiesTest2 = (model2.predict_proba(testX2))
tweet_id2, story_id2, pred_2 = writeCSV(text2, storyId2, tweetId2, skeptVals2, truthVals2, probabilitiesTest2, "pred_skeptVals2.csv")


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