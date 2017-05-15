#CS 349- Isabel D'Alessandro, Kelly Kung, Yuyu Li
#5/14/17
#split original twitter data into training, development, and testing files

import skepticsmClassifier as skc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import json

allData = skc.readData("jsonTrainingFiles.json")
print len(allData)
print(allData[allData.keys()[0]])

ids = [idnum for idnum in allData.keys()]
#there are 1242 tweets in here-- now we want to try to separate into training and testing

#train, test = train_test_split(ids, test_size = .1)
train = []
test = []
k_fold = KFold(10)
for train_index, test_index in k_fold.split(ids):
    train.append(train_index)
    test.append(test_index)

#we go through the indices of the train and test data given to read in the appropriate data 
trainData = []
testData = []   
for i in train:
    tempTrain = {}
    tempTest = {}
    tempData = [ids[idNum] for idNum in i]
    for j in allData.keys():
        if j in tempData:
            tempTrain[j] = allData[j]
        else:
            tempTest[j] = allData[j]
    trainData.append(tempTrain)
    testData.append(tempTest)
    
#write out the JSON files for train and test data
for i in range(0,len(trainData)):
    with open("trainJSONset_" + str(i) + ".json", 'wb') as doc:
        json.dump(trainData[i], doc)
doc.close()

for i in range(0, len(testData)):
    with open("testJSONset_" + str(i) + ".json", 'wb') as doc:
        json.dump(testData[i], doc)
doc.close()



    
