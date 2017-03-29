# from __future__ import print_function
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import pickle
import datetime as dt


def get_meeting_statements(filename):
    with open(filename) as f:
        return pickle.load(f)


def get_labels(filename):
    prices = dict()
    col_names = ['date', 'open', 'high', 'low', 'close', 'volume', 'count', 'WAP']
    for file in glob.glob(filename):
        this_file = file.split('/')[-1].split('.')[0]
        prices[this_file] = pd.read_csv(file, parse_dates=['date'], infer_datetime_format=True,names=col_names).drop_duplicates()
        prices[this_file].set_index('date', inplace=True,)
        prices[this_file].index = prices[this_file].index.tz_localize('America/Los_Angeles').tz_convert('America/New_York').tz_localize(None)
        prices[this_file]['close-MA-4'] = fit_moving_average_trend(prices[this_file]['close'], window=4)

    pre_post_FOMC_time_selector = []
    for date in minutes_df.index:
        pre_post_FOMC_time_selector.extend(pd.date_range(date.replace(hour=13, minute=30), periods=2, freq='2 H'))
    
    # This is prices that is only 
    prices_FOMC = dict()
    for key in prices.keys():
        prices_FOMC[key] = prices[key].loc[pre_post_FOMC_time_selector][['close-MA-4']].dropna()

def lemmatize_descriptions(meeting_statements):
    lem = WordNetLemmatizer()
    lemmatize = lambda d: " ".join(lem.lemmatize(word) for word in d.split())
    return [lemmatize(desc) for desc in meeting_statements]


def get_vectorizer(meeting_statements, num_features=5000):
    vect = TfidfVectorizer(max_features=num_features, stop_words='english')
    return vect.fit(meeting_statements)


def run_model(Model, X_train, X_test, y_train, y_test):
    m = Model()
    m.fit(X_train, y_train)
    y_predict = m.predict(X_test)
    return accuracy_score(y_test, y_predict), \
        f1_score(y_test, y_predict), \
        precision_score(y_test, y_predict), \
        recall_score(y_test, y_predict)

def fit_moving_average_trend(series, window=6):
    return series.rolling(window=window,center=False).mean()

def compare_models(meeting_statements, labels, models):
    desc_train, desc_test, y_train, y_test = \
        train_test_split(meeting_statements, labels)

    print "-----------------------------"
    print "Without Lemmatization:"
    run_test(models, desc_train, desc_test, y_train, y_test)

    print "-----------------------------"
    print "With Lemmatization:"
    run_test(models, lemmatize_descriptions(desc_train),
             lemmatize_descriptions(desc_test), y_train, y_test)

    print "-----------------------------"


def run_test(models, desc_train, desc_test, y_train, y_test):
    vect = get_vectorizer(desc_train)
    X_train = vect.transform(desc_train).toarray()
    X_test = vect.transform(desc_test).toarray()

    print "acc\tf1\tprec\trecall"
    for Model in models:
        name = Model.__name__
        acc, f1, prec, rec = run_model(Model, X_train, X_test, y_train, y_test)
        print "%.4f\t%.4f\t%.4f\t%.4f\t%s" % (acc, f1, prec, rec, name)


if __name__ == '__main__':
    meeting_statements = get_meeting_statements('../data/minutes_df.pickle')
    labels = get_labels('data/labels.txt')
    print "distribution of labels:"
    for i, count in enumerate(np.bincount(labels)):
        print "%d: %d" % (i, count)
    models = [LogisticRegression, KNeighborsClassifier, MultinomialNB,
              RandomForestClassifier]
    compare_models(meeting_statements, labels, models)
