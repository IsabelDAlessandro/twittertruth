#Kelly Kung
#plot the accuracies to see how the hyperparameters are trained

import matplotlib.pyplot as plt
from skepticsmClassifier import *
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix


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
#plt.savefig("ngram_skept.jpeg")
#plt.show()


plt.plot(reg, meanReg, 'bo')
plt.xlabel("Regularization")
plt.ylabel("accuracy")
plt.title("Accuracy vs. Regularization Values (alpha)")
#plt.savefig("reg_skept.jpeg")
#plt.show()


plt.plot(maxiter, meanMaxIter, 'go')
plt.xlabel("Max Iterations")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Max Iteration Values")
#plt.savefig("maxiter_skept.jpeg")
#plt.show()


plt.plot(mindf, meanMinDF, 'ro')
plt.xlabel("Min df")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Min df Values")
#plt.savefig("mindf_skept.jpeg")
#plt.show()


trying to do the confusion matrix now 
confusionMat = np.zeros((3,3))
indexMap = {"deny":0, "query":1, "support":2}
#have the true values along the columns 
for i in range(0, len(yVals)):
    for j in range(0, len(yVals[i])):
        confusionMat[indexMap[predictionsTest[i][j]], indexMap[yVals[i][j]]] += 1

#deny, query, support     
#[ 236.,   36.,   86.],
#[  36.,  272.,   86.],
#[  53.,   22.,  415.]

#this function is taken from online sources 
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

    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    plt.savefig(outFile)
    plt.show()