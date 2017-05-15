#Kelly Kung
#plot the accuracies to see how the hyperparameters are trained
#do analysis on the skepticism classifier
#5/14/17

import matplotlib.pyplot as plt
from skepticsmClassifier import *
import skepticsmClassifier 
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

#set up the data to be plotted
#read in the values given from the skepticsmClassifier code file 
ngram = np.unique(np.asarray([param[0] for param in param0]))
reg = np.unique(np.asarray([param[1] for param in param0]))
maxiter = np.unique(np.asarray([param[2] for param in param0]))
mindf = np.unique(np.asarray([param[3] for param in param0]))

parameters = np.asarray([param0, param1, param2, param3, param4, param5, param6, param7, param8, param9])
accuracies = np.asarray([acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9])
allAcc = np.asarray([acc2 for acc in accuracies for acc2 in acc])
ngramparam = np.asarray([param2[0] for param in parameters for param2 in param])
regparam = np.asarray([param2[1] for param in parameters for param2 in param])
maxIterparam = np.asarray([param2[2] for param in parameters for param2 in param])
mindfparam = np.asarray([param2[3] for param in parameters for param2 in param])

meanNgram = []
meanReg = []
meanMaxIter = []
meanMinDF = []

#returns the model, vectorizer, and one of the training data sets used, given the best hyperparameters
model, vectorizer, trainData = returnTrainandVect([7, .5, 20, 5])

#append the mean accuracies for each parameter in order to get the X and Y values for the different plots for parameters 
data = []
for i in trainData.keys():
    data.append(trainData[i]["text"])
vectorizer = vectorizer.fit(data) 

yVals, predictionsTest, probabilitiesTest, accuraciesTest, testData = returnVals()

for j in range(0, len(ngram)):
    meanNgram.append(np.mean(allAcc[ngramparam == ngram[j]]))
    
for j in range(0, len(reg)):
    meanReg.append(np.mean(allAcc[regparam == reg[j]]))
    
for j in range(0, len(maxiter)):
    meanMaxIter.append(np.mean(allAcc[maxIterparam == maxiter[j]]))
    
for j in range(0, len(mindf)):
    meanMinDF.append(np.mean(allAcc[mindfparam == mindf[j]]))

plt.plot(ngram, meanNgram, 'ro')
plt.xlabel("Ngram")
plt.ylabel("accuracy")
plt.title("Accuracy vs. Ngram Value")
plt.savefig("ngram_skept.jpeg")
plt.show()


plt.plot(reg, meanReg, 'bo')
plt.xlabel("Regularization")
plt.ylabel("accuracy")
plt.title("Accuracy vs. Regularization Values (alpha)")
plt.savefig("reg_skept.jpeg")
plt.show()


plt.plot(maxiter, meanMaxIter, 'go')
plt.xlabel("Max Iterations")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Max Iteration Values")
plt.savefig("maxiter_skept.jpeg")
plt.show()


plt.plot(mindf, meanMinDF, 'ro')
plt.xlabel("Min df")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Min df Values")
plt.savefig("mindf_skept.jpeg")
plt.show()

maxAccuracyIndex = np.argmax(accuraciesTest)
predictionMaxVals = predictionsTest[maxAccuracyIndex]
yMaxVals = yVals[maxAccuracyIndex]

#trying to do the confusion matrix now 
confusionMat = np.zeros((3,3))
indexMap = {"deny":0, "query":1, "support":2}
#have the true values along the rows
for i in range(0, len(yMaxVals)):
    confusionMat[indexMap[yMaxVals[i]], indexMap[predictionMaxVals[i]]] += 1

#deny, query, support     
#[[ 23.   4.   3.]
# [  2.  31.   3.]
# [  6.   3.  49.]]

#this function is taken from online source, but with modifications in order to plot the confusion matrix 
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,outFile,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
                 
    #plots the confusion matrix and then saves the file 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(outFile)
    plt.show()
    
plot_confusion_matrix(confusionMat, ["deny", "query", "support"], "confusionMat_skept.jpeg") 

   
#this is taken from online Stack Overflow and modified in order to get the level of importance as well
#http://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers  
def print_top10(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-20:]
        print("%s: %s" % (class_label,
              ", ".join(feature_names[j] for j in top10)))
        print("%s: %s" % (class_label,
              ", ".join(str(model.coef_[i][j]) for j in top10)))
              
              
print_top10(vectorizer, model, ["deny", "query", "support"])

#calculate precision and recall using the confusion matrix 
precision = np.zeros(np.shape(confusionMat)[0])
recall = np.zeros(np.shape(confusionMat)[0])
for i in range(0, np.shape(confusionMat)[0]):
    truePrediction = confusionMat[i,i]
    predicted = np.sum(confusionMat[:,i])
    actual = np.sum(confusionMat[i,:])
    precision[i] = truePrediction/predicted
    recall[i] = truePrediction/actual
    
print precision
print recall
print np.mean(precision) #81.62%
print np.mean(recall) #82.42%