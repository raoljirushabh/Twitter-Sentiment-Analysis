# Twitter-Sentiment-Analysis
connected real time tweets by providing a keyword and performed sentiment analysis

STEP1. CREATE TWEETER APP

STEP2. INSTALL DATABASE. I PICKED MONGOLDB - NOSQL DATABASE THAT ALLOWS YOU
PERFORM DATA MANIPULATION

STEP3. WRITE A FUNCTION THAT CONNECTS TO THE TWITTER APP, CATCHES TWEETS WITH
#CryptoCurrency AND SAVES THEM INTO MONGODB COLLECTION (PYTHON)


from __future__ import print_function
import tweepy
import json
from pymongo import MongoClient
import pymongo
import pandas as pd
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer


MONGO_HOST = 'mongodb://localhost:27017/twitterdb'
WORDS = ['#CryptoCurrency']
CONSUMER_KEY = "MG65OoScuuEKkd1zd93V4rYiN"
CONSUMER_SECRET = "m5vZSDtGnDZcHuttTYFfYQBXrsgwcuXtfHZwW8JAjBv5Wzkyhz"
ACCESS_TOKEN = "981633626403110913-dGtMpRxvTVQ7kaCZihkH6fY2YUIDuHg"
ACCESS_TOKEN_SECRET = "VLEsj95xcY8mTW11OOCJJRx6qLRZjyBJ7U57B0p2bWAj1"


# Methods that connects to tweeter app and retrieves messages by key words specified in WORDS[]
def load_tweets_into_mongo():
    class StreamListener(tweepy.StreamListener):
        # This is a class provided by tweepy to access the Twitter Streaming API.

        def on_connect(self):
            # Called initially to connect to the Streaming API
            print("You are now connected to the streaming API.")

        def on_error(self, status_code):
            # On error - if an error occurs, display the error / status code
            print('An Error has occured: ' + repr(status_code))
            return False

        def on_data(self, data):
            # This is the meat of the script...it connects to your mongoDB and stores the tweet
            try:
                client = MongoClient(MONGO_HOST)

                # Use twitterdb database. If it doesn't exist, it will be created.
                db = client.twitterdb

                # Decode the JSON from Twitter
                datajson = json.loads(data)

                # grab the 'created_at' data from the Tweet to use for display
                created_at = datajson['created_at']

                # print out a message to the screen that we have collected a tweet
                print("Tweet collected at " + str(created_at))

                # insert the data into the mongoDB into a collection called twitter_search
                # if twitter_search doesn't exist, it will be created.
                db.twitter_search.insert(datajson)
            except Exception as e:
                print(e)


STEP4. SAVE TWEETS FROM MONGODB TO THE PYTHON DATAFRAME


    load_tweets_into_mongo()
    conn = pymongo.MongoClient(MONGO_HOST)

    exclude_data = {'_id': False}
    raw_data = list(conn.twitterdb.twitter_search.find({}))

    # retrieve text field that contains tweets from the mongoDB
    tweeter_df = pd.DataFrame(raw_data)
    tweeter_df = tweeter_df[['text', '_id']]
    print(tweeter_df.dtypes)

    # take a look at messages
    print(tweeter_df['text'])
    print(tweeter_df[34])




STEP5. USE NAIVE BAYES CLASSIFIER TO PERFORM SENTIMENT ANALYSIS (TEXTBLOB
LIBRARY) AND SAVE RESULTS OF THE ANALYSIS TO CSV FILE

def naive_bayes_classifier():
    import nltk.classify.util
    from nltk.classify import NaiveBayesClassifier
    from nltk.corpus import movie_reviews

    def word_feats(words):
        return dict([(word, True) for word in words])

    negids = movie_reviews.fileids('neg')
    posids = movie_reviews.fileids('pos')

    negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
    posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

    negcutoff = len(negfeats) * 3 / 4
    poscutoff = len(posfeats) * 3 / 4

    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
    print
    'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))

    classifier = NaiveBayesClassifier.train(trainfeats)
    print
    'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
    classifier.show_most_informative_features()

    blob = TextBlob("I love this library", analyzer=NaiveBayesAnalyzer() )
    blob.sentiment
    
    
    
    tweeter_df['polarity'] = tweeter_df['text'].apply(classify)
    print(tweeter_df.head(10))

    # save results to csv
    tweeter_df.to_csv('classified_tweets.csv')
    
    
STEP6. CREATE WORD CLOUD OUT OF RESULTS

df = pd.read_csv('classified_tweets.csv')
neg_string = pd.Series(pd.Series(df['text'])).str.cat( sep=' ' )

def wordCloud(df):
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    wordcloud = WordCloud( width=1600, height=800, max_font_size=200 ).generate(df)
    plt.figure( figsize=(12, 10) )
    plt.imshow( wordcloud, interpolation="bilinear" )
    plt.axis("off")
    plt.show()

    
    
