from preprocessing import TweetPreProcessor
import json
import os
from collections import Counter
from nltk.corpus import stopwords
from nltk import bigrams
import string

fname = 'tweet.json'
all_terms = []
os.system("cls") # Clears screen
# To eliminate punctuations also
punctuation = list(string.punctuation)
stop_words = stopwords.words('english') + punctuation + ['rt', 'via','…']
try:
    with open(fname,'r') as fp:
        count = Counter()
        for line in fp:
            tweet = json.loads(line)
            # Create a list with all the terms in the tweet(by converting into lowercase to compare)
            all_terms = [term for term in TweetPreProcessor(tweet['text']) if term.lower() not in stop_words]
            # Update the count of the terms
            count.update(all_terms)
        # Print the first 5 most common words
        print(count.most_common(5))
except Exception as error:
    print(str(error))

# Count terms only once equivalent to document frequency
single_terms = set(all_terms)
# Filter terms with hashtags
hashtag_terms = [term for term in all_terms if term.startswith('#')]
# Frequent terms without hashtags
only_terms = [term for term in all_terms if not term.startswith(('#','@'))]
# To identify bigrams
bigram_terms = bigrams(all_terms)
print(hashtag_terms)
print(only_terms)
print(list(bigram_terms))
