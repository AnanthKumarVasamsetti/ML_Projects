import tweepy
from tweepy import OAuthHandler
import json
import timeit
from tweepy import Stream
from tweepy.streaming import StreamListener
import sentiment_mod as s

#Live streaming class starts
class MyListener(StreamListener):
    def on_data(self,data):
        try:
            all_data = json.loads(data)
            tweet = all_data['text']
            sentiment_value, confidence = s.sentiment(tweet)
            print(tweet, sentiment_value, confidence)
            if(confidence * 100 >= 80):
                with open('tweet.txt','a') as fp:
                    fp.write(sentiment_value+" "+str(confidence))
                    fp.write('\n')
                    fp.close()

                with open('tweet_text.txt','a') as fp:
                    fp.write(tweet)
                    fp.write('\n')
                    fp.close()
            return True
        except BaseException as error:
            print(str(error))
        return True
    
    def on_error(self,status):
        print(status)
        return True
#Live streaming class ends

def process_or_store(tweet):    
    return json.dumps(tweet,indent = 4)

consumer_key = "DyoradoFFgvUeLV2D1cuf5LhF"
consumer_secret = "MiAYDYSj5QQbs3WkZc3lWSpxk7RrtXUxFnqZfczWYfS3zSuQq4"

access_key = "228366541-6KK4oTN89YpgWAIPd6dWYjhZGequ57AszT2Pf9GX"
access_secret = "bSieAmeE4uXFFwIF8jXUAORI2pOwQKWJ2vKtaMmNK4OO4"

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)

api = tweepy.API(auth)
fp = open('tweet.txt','w')
fp.close()
fp = open('tweet_text.txt','w')
fp.close()
# start_time = timeit.default_timer() #To calculate the time taken by the complete execution of for loop
# for tweet in tweepy.Cursor(api.user_timeline).items(10):
#     tweets_dump = process_or_store(tweet._json)
#      fp.write(tweets_dump)
# print("--- %0.3s seconds ---"%(timeit.default_timer() - start_time))

#For live streaming
twitter_stream = Stream(auth,MyListener())
twitter_stream.filter(track=['#FridayFeeling'])
