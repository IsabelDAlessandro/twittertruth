# Isabel D’Alessandro, Kelly Kung, Yuyu Li
# CS 349
# 4/20/17
# CS 349 Update

Milestone 3

Original goal:
    
   In our proposal, our goal by 4/20 was to have completed the sentiment analysis portion of our project. However, the original plan for the project was to separate it into two stages, the first looking at Tweet-level sentiment analysis and the second classifying which stories are true using the sentiment measures as features. After further examining the data and honing in on which tasks seemed the most useful and interesting, we realized we would need three classifiers, not two. The current plan is still in two stages, although the first stage now not only includes the sentiment analysis portion but also one to classify the Tweets into different classes based on the extent to which they express skepticism. The second stage is to evenly weight the influence of sentiment analysis and skepticism in a final classifier to determine a real value for the predicted truth of a story.

Progress so far:
    
   Because we extended the breadth of the project to include both the sentiment analysis and skepticism aspects to better featurize our stage two classifier for truth value of a Twitter story, we are not yet halfway through the coding portion of our project (where we had hoped to be by this milestone). For the sentiment analysis portion, we originally tried to work with a prebuilt classifier for emotions (Indico.io) for a while, attempting to correct issues with the allotted byte size of the request since we are using the free version. However, after a sizable amount of time spent trying to wrestle with this API (which is built on top of a neural network), we thought it would be more feasible to simply build our own logistic regression classifier. We are using the SemEval Task 4E data as training data for this classifier (http://alt.qcri.org/semeval2016/task4/). Instead of using emotions like anger, sadness, etc. as classes, this will classify the Tweets on a -2 to 2 scale, and the classes include very negative, negative, neutral, positive, and very positive. Currently, we are working on figuring out the Twitter API to fetch the text successfully and apply TFIDF to this. We have a working classifier for already featurized text, but need to find the best way to featurize the Tweets.
   We’ve also explored different methods of featurizing the original text (Tweets), including a number of variants on the bag-of-words representation (TFIDF, n-grams, count vectorizing). We plan to pass this featurized test data into our final classifier to get a prediction of the truth value of each story. Since the final classifier has not been built or featurized yet, we have currently been seeing what kind of predictions we can get by looking at the text body alone (without sentiment or skepticism classes).

To do:
    
   The most significant part of the project left we have is to tackle stage 2, which will combine the previous classifications from stage 1 (sentiment analysis portion and skepticism portion) to generate our final prediction. In order to accomplish this, the most pressing task is to figure out the best way to fetch the text and figure out the best way to featurize it for the two classifiers in stage 1. Since we have some preliminary methods of predicting based on text body alone, it will not be as difficult to featurize the data for the stage 2 classifier since these just involve combining the previous classes with the featurized text. We are hoping to get both stage 1 classifiers to be optimized for accuracy by the May 1, which will leave two weeks for building the final classifier and  writing the paper.

