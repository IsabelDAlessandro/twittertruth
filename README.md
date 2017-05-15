# Machine Learning Project: Classifying Fake News Using Sentiment from Tweets

Isabel D'Alessandro, Kelly Kung, Yuyu Li

Report: 
https://docs.google.com/document/d/1v2F9lBEFXeheDuA5a9ETVMFDzaE0Ct6iuMUCF3p_jew/edit?ts=590b88f1 

Descriptions of Files in this Repository: 

analysisPlots.py: Analyze and display results of skepticism classifier 

cleanSkepticismTwitterData.py: Cleans data obtained from RumorEval for use in skepticism classifier

cleanTakisData.py: Cleans original twitter data by using only English-language text, removing irrelevant characters, etc. 

crossValidateText.py: Apply cross-validation to original twitter data set 

featurizeText.py: Featurize original twitter data set using tfidf and generate true/false predictions using text only 

makeTrainingandTestingData.py: split original twitter data into training, development, and testing files 

sentiment_analysis_LR.py: sentiment classifier; generates probabilities of each sentiment value('very positive', 'positive', 'OK','negative','very negative')

sentiment_analysis_train.py: featurizes tweets from TwitterTrails data for use with sentiment classifier 

sentiment_train.txt: Apply cross-validation to SemEval data for use in sentiment classifier 

skepticismClassifier.py: skepticism classifier; generates probabilities of each skepticism value(support, deny, query)


Data: 
https://drive.google.com/drive/folders/0B7umSMDTRTHBMmdUbzFQZkFQaTg?usp=sharing

The data folder contains 3 subfolders: 
1) Sentiment Data
This folder contains the data used for the sentiment classifier.

2) Skepticism Data
This folder contains the data from SemEval that was used for the skepticism classifier. This contains the data generated from cross validation as well as the JSON file created using the actual SemEval data (not included in this drive).

3) Takis Data
This folder contains the test data that we are looking at. These are the tweets from Twitter Trails that Takis provided us with. This folder contains the whole data set as well as when it is split into training, development, amd testing. 
