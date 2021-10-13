'''
Made specifically to preprocess tweets stored in csv format
'''
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
import csv
import pickle

def preprocessing(file):
    data = []
    with open(file, encoding = "ISO-8859-1") as f:
        reader = csv.reader(f)
        try:
            for line in reader:
                tweet = []
                if line[0] == '4': line[0] = '1'
                tweet.append(line[0])
                tweet.append(line[5])
                data.append(tweet)
        except Exception as e:
            print(f'\n\tError: {e}')
            print(f'\t{line}\n')
    return data

def train(text, sentiment):
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(text, sentiment)
    return model

print('1. Cleaning tweets')
clean_tweets = preprocessing('data/tweets.csv')
with open('data/clean_tweets.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(clean_tweets)

sentiment, text = [], []

print('2. Splitting text and target')
for i in range(len(clean_tweets)):
    sentiment.append(clean_tweets[i][0])
    text.append(clean_tweets[i][1])

print('3. Model training')
model = train(text, sentiment)

print('4. Saving model')
with open('model_naivebayes', 'wb') as f:
    pickle.dump(model, f)
