import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
import numpy as np

# Завантаження даних
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return data

# Обробка твіту
def process_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)  # Видалення URL
    tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)  # Видалення згадок
    tweet = re.sub(r'#[A-Za-z0-9]+', '', tweet)  # Видалення хештегів
    tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)  # Видалення спеціальних символів
    tweet = tweet.lower()
    tokens = word_tokenize(tweet)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens

# Завантаження файлів
positive_tweets = load_data('plot.tok.gt9.5000')
negative_tweets = load_data('quote.tok.gt9.5000')

# Позначення класів
labels = [1] * len(positive_tweets) + [0] * len(negative_tweets)
tweets = positive_tweets + negative_tweets

# Обробка текстів
processed_tweets = [process_tweet(tweet) for tweet in tweets]

def build_frequency_dict(tweets, labels):
    freq_dict = {}
    for label, tweet in zip(labels, tweets):
        for word in tweet:
            pair = (word, label)
            freq_dict[pair] = freq_dict.get(pair, 0) + 1
    return freq_dict

freq_dict = build_frequency_dict(processed_tweets, labels)

def compute_log_prior(labels):
    num_positive = sum(labels)
    num_negative = len(labels) - num_positive
    total = len(labels)
    log_prior_positive = np.log(num_positive / total)
    log_prior_negative = np.log(num_negative / total)
    return log_prior_positive, log_prior_negative

log_prior_positive, log_prior_negative = compute_log_prior(labels)

def compute_log_likelihood(freq_dict, tweets, labels):
    vocab = set([pair[0] for pair in freq_dict.keys()])
    V = len(vocab)

    positive_total = sum([freq for (word, label), freq in freq_dict.items() if label == 1])
    negative_total = sum([freq for (word, label), freq in freq_dict.items() if label == 0])

    log_likelihood = {}
    for word in vocab:
        freq_positive = freq_dict.get((word, 1), 0)
        freq_negative = freq_dict.get((word, 0), 0)

        log_likelihood[word] = {
            1: np.log((freq_positive + 1) / (positive_total + V)),
            0: np.log((freq_negative + 1) / (negative_total + V))
        }
    return log_likelihood

log_likelihood = compute_log_likelihood(freq_dict, processed_tweets, labels)

def naive_bayes_predict(tweet, log_prior_positive, log_prior_negative, log_likelihood):
    tweet = process_tweet(tweet)
    positive_score = log_prior_positive
    negative_score = log_prior_negative

    for word in tweet:
        if word in log_likelihood:
            positive_score += log_likelihood[word][1]
            negative_score += log_likelihood[word][0]

    return 1 if positive_score > negative_score else 0

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(processed_tweets, labels, test_size=0.2, random_state=42)

# Передбачення на тестовому наборі
y_pred = [naive_bayes_predict(' '.join(tweet), log_prior_positive, log_prior_negative, log_likelihood) for tweet in X_test]

# Обчислення точності
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

def analyze_words(log_likelihood):
    positive_words = sorted(log_likelihood.items(), key=lambda x: x[1][1], reverse=True)[:10]
    negative_words = sorted(log_likelihood.items(), key=lambda x: x[1][0], reverse=True)[:10]

    print("Most positive words:", positive_words)
    print("Most negative words:", negative_words)

analyze_words(log_likelihood)

def analyze_errors(X_test, y_test, y_pred):
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            print(f"Error: True={y_test[i]}, Predicted={y_pred[i]}, Tweet={' '.join(X_test[i])}")

analyze_errors(X_test, y_test, y_pred)

my_tweet = "This movie was absolutely fantastic!"
prediction = naive_bayes_predict(my_tweet, log_prior_positive, log_prior_negative, log_likelihood)
print(f'My tweet sentiment: {"Positive" if prediction == 1 else "Negative"}')
