


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 

train_text=[]

for n in range(1,5):
    vectorizer = CountVectorizer(ngram_range=(1, n))  # initialize object with CountVectorizer defaults

    # convert to array where each row is an essay, each dimension is a word, 
    # and each value is the count of that word in the essay
    trainX = vectorizer.fit_transform(train_text)  

    trainX = trainX.toarray()   # make dense

    devX = vectorizer.transform(dev_text)  # featurize the development text
    devX = devX.toarray()

    testX = vectorizer.transform(test_text)  # featurize the testing text
    testX = testX.toarray()
    
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()  # one-versus rest logistic regression

    model.fit(trainX, trainy)
    accuracy = model.score(devX, devy)
    print 'CountVectorizer'+'n:'+str(n)+'Classification accuracy', accuracy
    

vectorizer=TfidfVectorizer()

trainX = vectorizer.fit_transform(train_text)  

trainX = trainX.toarray()   # make dense

devX = vectorizer.transform(dev_text)  # featurize the development text
devX = devX.toarray()

testX = vectorizer.transform(test_text)  # featurize the testing text
testX = testX.toarray()
    
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()  # one-versus rest logistic regression

model.fit(trainX, trainy)
accuracy = model.score(devX, devy)
print 'tfidf'+'n:'+str(n)+'Classification accuracy', accuracy
    
