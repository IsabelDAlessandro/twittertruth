import indicoio
import itertools
indicoio.config.api_key = '679a106bfcb43933527ec61eb2a17647'

# single example
#var = indicoio.emotion("There are so many things you can learn from text.")
#print var

# batch example
empty = []
tweets = [
    "There are so many things you can learn from text.",
    "You could use the features to train a model!"
]
batch = indicoio.emotion(tweets)
empty.append(batch)
#print empty

merged = list(itertools.chain.from_iterable(empty))
#print merged

print batch 
for i in merged:
    print i['anger']
    

'''
    # emotion scores
    sentiments = []
    for group in math.ceil(tweets/100):
        batch = []
        for i in range (0,100):
            if group * 100 + i >= tweets.size:
                break
            else:
                batch.append = tweets[group * 100 + i]
        sent = indicoio.emotion(batch)
        sentiments.append(sent)
'''
    
'''
    sentiments = indicoio.emotion(tweets)
    #print sentiments.size 
    
    anger = []
    joy = []
    fear = []
    sadness = []
    surprise = []
    sentiments = list(itertools.chain.from_iterable(sentiments)) #flatten list
    for s in sentiments:
        anger.append(s[0])
        joy.append(s[1])
        fear.append(s[2])
        sadness.append(s[3])
        surprise.append(s[4])
'''
    
# anger[i], joy[i], sadness[i], fear[i], surprise[i]