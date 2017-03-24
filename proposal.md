# Isabel D’Alessandro, Kelly Kung, Yuyu Li
# CS 349
# 3/22/17
# CS 349 Project Proposal 

Description of Problem and Motivations:

   Fake news has become more prevalent, especially after this past election. However, many people are not able to distinguish between real and fake news. One reason for this is because of how widespread news is on social media platforms, as well as social media networks serving as an echo chamber for certain types of new articles, many of which can be inaccurate. These news articles gain momentum because of spammers on social media (such as Twitter) who post about these articles many times within a short period of time. 
   Takis and Eni began researching about Twitter Trails even before fake news became relevant. They collected tweets from Twitter using key tags and analyzed the reactions of the tweets and the timeline of stories. Using key tags, they were able to classify stories as being real or fake and identify the tweets that were the origin points of popular stories.
   In our project, we hope to analyze the text of the tweets that comprise a certain story, which can include both a story’s headline text as well as the specific Twitter user’s reaction to the story. Using sentiment analysis, we want to use supervised learning to classify and predict the stories as being fake or true. In doing so, we also want to identify the words and features that are most useful to identify a story as fake or real. 


Previous Work:

   In order to analyze stories and see whether they are fake or not, Takis and Eni have created TwitterTrails. TwitterTrails is a web program that allows users to search for tweets based on keywords and provides graphics that helps the user analyze a story. To evaluate the truth value of a story, the system uses features like propagation, timeline, and retweet and co-retweet counts. Specifically, the original system was built upon 3 main algorithms: a tweet relevance algorithm, used to determine which tweets are relevant to a story based upon keywords, a ‘burstiness algorithm’, used to determine who broke the story and when the story became relevant using metrics such as retweet count, and a ‘negation algorithm’, which identifies whether there is skepticism or denial about a circulating story using a set of keywords which indicate doubt, such as ‘hoax’,’fake’, and ‘untrue (Metaxas, P.T., Mustafaraj, E., et al, 2014)’. 
   Based on these measures, the system generates a prediction about a given story’s truth value, along with an associated probability (e.g. “This claim is likely false, with a probability of 87%”). Using these graphics and information provided, users are also able to track the tweet that started the widespread of the story, to look at the network of users and identify spammers, and combining facts about the tweets to indicate whether a story is fake or not (Metaxas, P.T., Mustafaraj, E., et al, 2014). 


   There has been some previous work focused on evaluating the credibility of tweets. For example, work by Gupta, Kumaraguru , Carlos Castillo, and Patrick Meier, 2015, constructed a system for evaluating the credibility of a tweet. However, this evaluation was based on high-level information about the user, such as username, number of followers, number of tweets and retweets, rather than the text of the tweets themselves. 
   Another related vein of research has focused on sentiment analysis of tweets, separate from the goal of predicting their validity. These studies have made use of sentiment-labeled twitter data (typically human-labeled positive, or negative), as well as non-text features, such as punctuation, hashtags, and emoticons (Go et al., 2009; Pak et al., 2011). These studies have made use of a number of different classification algorithms and models, from tree kernels (Argwal et al.2011) to  SVM and Naive Bayes (Pak et al., 2011). 


Data Set Source:
 
   We are currently waiting to obtain the data set used for TwitterTrails from Takis, which we will get by Friday, March 24. These contain over 200 stories represented in JSON format that have been tagged for TwitterTrails, as well as their truth probability as determined by the existing system. Though we are not sure what this looks like now, the graphics and data on the site show that there are values such as: overall skepticism demonstrated by retweeters, spread of story, what sort of keywords retweeters have on their profiles, timeline. We hope to also use the Twitter API to find additional information such as following/ follower statistics. 

Methods of Featurization and Classification:

   Our project can be separated in two general stages, the first looking at Tweet-level sentiment analysis and the second classifying which stories are true using the sentiment measures as features. For the sentiment analysis portion, because the text of the Tweets are the data, we can use methods and features like bag of words, n-grams, and TF-IDF scores. Additionally, since there are many existing lists of words associated with particular emotions or truth value, we can also look at the frequency these show up in the Tweets (a keyword count). 
   For the classification component, we will use logistic regression to predict probability of a certain label, tuning hyperparameters based on our accuracy scores. If time permits, we would also like to explore turning the features we used into variables for linear regression. 
	
Methods of Analysis/Evaluation:

   We would like to first analyze our results against TwitterTrails findings. Because both those results and our predictions are a likelihood percentage, we will calculate mean squared error. However, because one of the tasks Takis expressed interest in was whether there was a more sophisticated way to measure the extent and effect of skepticism, we would also like to evaluate our findings against a different source of news verification (yet to be determined) to see if perhaps we can improve the TwitterTrails predictions. If this new verification source also uses percentages, we can again calculate mean squared error. If this source uses a binary label, we will simply give these real values between 0 and 1 to assess mean squared error against our predictions.


Primary Goal of Project:

   We hope to be able to classify and predict whether the story/ news that the tweets are about are real or fake using the sentiment analysis of words in the tweets. In doing so, we also hope to find the words that are most indicative of whether a story is real or fake. 


Responsibilities of Members of Group:

   Currently, we have broken up our project into 5 main tasks: data cleaning, research about sentiment analysis and emotions, feature design, coding and designing the model, and the evaluation and improvement of the model. The most difficult and crucial tasks for the project that we all want to gain skills in are feature design and modeling, which we will all take part in. The remaining 3 tasks are split up as follows:
   Kelly will be working on cleaning up the data. Isabel will look at previous research in order to identify ways to featurize and perform sentiment analysis on the text. Once we have finished with our models (regression, etc), we will train them using our featurized training data, and test them against a reserved set of testing data (set of featurized stories).  Yuyu will analyze the accuracy and evaluate our model in order to find ways to improve it.

Goals and Timeline:

The proposed timeline of our project and milestones are shown in the calendar. 

Link to our document with the calendar shown: https://docs.google.com/document/d/1rK1eKWhJzcxo7bAWLkGomE0Bro3bLFNdy9vVZ_x8ftk/edit



References: 

Apoorv Agarwal, A., Xie, B., Vovsha, I.,  Passonneau, O. (2011). Sentiment Analysis of Twitter Data  Department of Computer Science Columbia University New York, NY 10027 USA
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.207.5100&rep=rep1&type=pdf#page=40

Finn, S., Metaxas, P.T., Mustafaraj, E., Okeefe, M., Tang, L., Tang, S., Zeng, L. (2014). TRAILS: A System for Monitoring the Propagation of Rumors On Twitter S. Finn, P. T. Computer Science Department Wellesley College, Wellesley, MA 02481
http://compute-cuj.org/cj-2014/cj2014_session2_paper2.pdf 

Go, A., Huang,L., & Bhayani, R. (2009). Twitter sentiment analysis. Final Projects from CS224N for Spring 2008/2009 at The Stanford Natural Language Processing Group

Pak, A. & Paroubek, P. (2011). Twitter as a Corpus for Sentiment Analysis and Opinion Mining Universit´e de Paris-Sud, Laboratoire LIMSI-CNRS, Bˆatiment 508, F-91405 Orsay Cedex, France 
http://web.archive.org/web/20111119181304/http://deepthoughtinc.com/wp-content/uploads/2011/01/Twitter-as-a-Corpus-for-Sentiment-Analysis-and-Opinion-Mining.pdf
