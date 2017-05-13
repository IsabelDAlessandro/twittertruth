#Kelly Kung
#this file tries to separate the test and training Data for skepticism
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
    
#trainJson = {}
#testJson = {}
#for idnum in allData.keys():
#    if idnum in train:
#        trainJson[idnum] = allData[idnum]
#    else:
#        testJson[idnum] = allData[idnum]

for i in range(0,len(trainData)):
    with open("trainJSONset_" + str(i) + ".json", 'wb') as doc:
        json.dump(trainData[i], doc)
doc.close()

for i in range(0, len(testData)):
    with open("testJSONset_" + str(i) + ".json", 'wb') as doc:
        json.dump(testData[i], doc)
doc.close()



    