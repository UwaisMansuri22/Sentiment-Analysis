from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import nltk
from nltk.corpus import twitter_samples, stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import random
import numpy as np
from transformers import *
import en_vectors_web_lg
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')

nlp = en_vectors_web_lg.load()

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased')

set(stopwords.words('english'))
app = Flask(__name__)

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

def construct_sentence_vector(sentence):
    return np.array([token.vector for token in nlp(sentence)]).mean(axis=0)

def construct_dim_vector(descriptive_words):
    return np.array([construct_sentence_vector(sentence) for sentence in descriptive_words]).mean(axis=0)

def euclidian_distance(X, Y):
    return np.sqrt(np.sum(np.power(X-Y, 2)))

def cosine_sim(X, Y):
    return np.dot(X, Y)/(np.linalg.norm(X) * np.linalg.norm(Y))


classes = ['delivery', 'service', 'quality', 'price', 'misc']
lexicon = {
    'delivery': ['time', 'schedule', 'punctual', 'delayed', 'packaging damaged', 'quick delivery', 'despatched'],
    'service': ['cancel', 'cancellation', 'credit', 'refund', 'excellent service', 'fantastic service', 'fast service',
                'awesome', 'disguisting service', 'customer service', 'staff terrible'],
    'quality': ['poor quality', 'damaged', 'excellent product', 'good quality', 'nice', 'love', 'good product',
                'great value', 'good materials', 'fantastic', 'awesome product', 'happy quality', 'great design'],
    'price': ['great value money', 'moneys worth']
}

lexicon_vectors = {}
for aspect, descriptive_words in lexicon.items():
    lexicon_vectors[aspect] = construct_dim_vector(descriptive_words)

def select_aspect(review):
    vec = construct_sentence_vector(review)
    sims = np.array(list(map(lambda lex_v: cosine_sim(vec, lex_v), lexicon_vectors.values())))
    max_cat = np.argmax(sims) if sims.max() > 0.40 else -1
    return max_cat


@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/templates/<path:path>')
def send_js(path):
    return send_from_directory('templates', path)
    
@app.route('/',methods=['POST'])
def my_form_post():
    text1 = request.form['text1'].lower()
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    text = twitter_samples.strings('tweets.20150430-223406.json')
    tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]

    stop_words = stopwords.words('english')

    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(tokens)

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(tokens)

    all_pos_words = get_all_words(positive_cleaned_tokens_list)

    freq_dist_pos = FreqDist(all_pos_words)
    print(freq_dist_pos.most_common(10))

    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

    positive_dataset = [(tweet_dict, "Positive")
                        for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, "Negative")
                        for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset

    random.shuffle(dataset)

    train_data = dataset[:8000]
    test_data = dataset[8000:]

    classifier = NaiveBayesClassifier.train(train_data)

    print("Accuracy is:", classify.accuracy(classifier, test_data))

    print(classifier.show_most_informative_features(10))

    print(text1.split('\n'))
    result = []

    class result_obj:
        def __init__(self, review, result_review, aspect_review):
            self.review = review
            self.result = result_review
            self.aspect = aspect_review
    for x in text1.split('\n'):
        if x != "" and x != '\r':
            custom_tokens = word_tokenize(x)
            review_result = classifier.classify(dict([token, True] for token in custom_tokens))
            predicted_aspects = select_aspect(x)
            aspect_output = (lambda ix: classes[ix])(predicted_aspects)
            result.append(result_obj(x, review_result, aspect_output))

    y_axis = []
    x_axis = []
    for i in result:
        if i.aspect not in x_axis:
            x_axis.append(i.aspect)

    for j in x_axis:
        count = 0
        for i in result:
            if i.aspect == j:
                count = count + 1

        y_axis.append(count)

    x = np.array(x_axis)
    y = np.array(y_axis)
    plt.bar(x, y)
    plt.savefig("templates/graphs.png")
    return render_template('form.html', result=result)

if __name__ == '__main__':
  app.run(debug=True, host ="127.0.0.1",port=4545, threaded = True)
    