Isabel D’Alessandro, Kelly Kung, Yuyu Li
4/5/17
CS 349

Milestone 2 Report

Data Source Links:
Twitter Trail Test Data
http://cs.wellesley.edu/~pmetaxas/stories_spreadsheet.csv.zip
External Data (used for training and development)
http://alt.qcri.org/semeval2017/task8/index.php?id=data-and-tools 
Tweet, IDs, T/F of story,skepticism (deny support truth values), replies’ text

Updates:

For this milestone, we cleaned up the data set and separated it into three parts: training, development, and testing. To clean the data, we first removed all non-English tweets. This is because it would be more difficult to analyze tweets of other languages since we do not know them. Furthermore, English tweets make up a large portion of the entire data set, and so we can still analyze the majority of the data set. After removing the non-English tweets, we separated the data set into three groups by randomly choosing a third of the available stories (51 in total) and their corresponding tweets to be in each part (training, development, and testing). Lastly, we included columns that we thought would be useful for our analysis. Namely, they are: story ID, tweet ID, whether the story is true or false, whether the tweet is skeptical of the story, the tweet text, the date of the tweet, the user who tweeted the message, and whether or not the tweet is a retweet. 

The data is currently in the git repository. The data files are named: dev, training2, and test. They are in csv format. The code cleanData.py is the data used to clean the data sets. Besides the aforementioned features that we decided to keep, we also wanted to add some sentiment analysis besides just positive, negative, or neutral. By analyzing the emotion of the tweet text, we hope to combine this with the existing features to determine a better skepticism score or indicator, and then to use these tweet-level skepticism scores to predict whether a story is true. To do this, we used a pre-built tool that uses a convolutional neural net to determine a real value for a tweet’s expression of anger, joy, sadness, fear, and surprise. This can be found at https://indico.io/docs#emotion. We ran into several difficulties that will be more thoroughly documented later in this document, but are currently attempting to get this and append it to our data as the last five columns (one for each emotion). We have confirmed that the API works and we are accessing it correctly in test.py, but are currently trying to divide the data so that it can be split between API calls or between users, since running the sentiment analysis package on all the tweet texts resulted in us exceeding the number of allowed bytes for an API call.

We’ve identified a prioritized list of features that we ultimately plan to use in our final classification of a story’s truth value. First, we will pass the raw text of all tweets pertaining to a story into a sentiment classifier which will output the percentage of tweets pertaining to each story which express anger, surprise, sadness, fear and joy. We may initially try to obtain this data using a pre-trained classifier(indicio.io sentiment analysis), and if time permits, build our own classifier using data from this source. Next, we will build a multi-class logistic regression classifier to categorize the skepticism of the body of tweets related to each story using the raw text of the tweets. To do this, we will make use of the SemEval RumorEval dataset , which contains labeled training and test data containing the raw text of tweets pertaining to a set of news stories as well as the replies to these tweets, and the associated labels: comment, query, deny, or support. The purpose of this step in our process is to obtain the percentage of tweets related to each story which are classified as comments, queries, denials, or supports, and to use this parameter as one feature (along with sentiment and the raw text (tfidf values, etc)) in our final classifier, which will make use of logistic regression to determine if a story is true or false (expressed as a probability). 

Difficulties: 
One of the difficulties we had was figuring out how we wanted to approach the problem and what we wanted to prioritize for our project. We’re also questioning which algorithms to use for each of our classifiers (skepticism classifier and final true/false classifier). We originally planned to use multi-class logistic regression. However, we’re wondering if there are alternative types of classifiers that might be better suited to the application. 

Though in test.py, we were able to try the indico.io API on a couple of examples successfully, the first challenge we ran into was exceeding the allowable bytes for an API call. Since this site seems to be managed by a small team, we are contacting them to see if we can extend this. An alternative that we tried was splitting each set (training, development, and testing) into batches of no more than 10,000 tweets. However, when attempting to write this to the CSV-formatted outfile, the new columns for each of the five emotions does not show up. We originally suspected this has to do with indico.io, but when attempting to append the column labels for the outfile, even appending the strings “Anger”, “Joy”, “Sadness”, “Fear”, and “Surprise” are unsuccessful. 

Our plan for this is first to see if it is a machine-specific problem, since the sentiment analysis portion has only been run on one member’s computer. If this is still a problem, we will fix the issue of writing in additional columns to the outfile, then verifying that the batch division is working, and then calling the indico.io API on these smaller batches.

Next steps:
1. Examining the SemEval dataset, parsing the json files containing the text of the tweets, cleaning the data, organizing the tweets and the replies to those tweets → format into a json file of storyID:{[{tweetId:{[tweet,tweet],[tweet replies], [skepticism label]}},true/false label}
2. Write functions to work with the indicio API to easily output sentiments given raw tweet text
3. Featurize all text using a number of different methods (e.g. TFIDF, n-gram, etc.)- tune the skepticism classifier 
4. Pass this text into our skepticism classifier 
5. Combine the outputs of the skepticism classifier, sentiment classifier, and featurizations of the text itself to be used as features for our true/false prediction step
6. Train classifier to predict whether or not a story is fake/real


