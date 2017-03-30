# from __future__ import print_function
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from dumb_predictors import MeanRegressor, ModeClassifier
import pandas as pd
import numpy as np
import pickle
import datetime as dt
import glob


class Fab(object):
	'''
	Fab - FOMC Analyzer Bot

	A class to create predictions on the FOMC meeting minutes
	'''
	def __init__(self):
	    self.meeting_statements = self.get_meeting_statements('../data/minutes_df.pickle')
	    self.labels = self.get_labels('../data/*.csv', self.meeting_statements.index)

	    for ticker in self.labels.keys():
	        print ticker
	        print "distribution of labels:"
	        for i, count in enumerate(np.bincount(self.labels[ticker]['binary'].values)):
	            print "%d: %d" % (i, count)
	        models = [LogisticRegression, KNeighborsClassifier, MultinomialNB,
	                  RandomForestClassifier, ModeClassifier]
	        self.compare_models(self.meeting_statements.loc[self.labels[ticker].index]['statements'].values.tolist(), 
	        			   self.labels[ticker]['binary'].values, models)
	        print "" 
	        print "" 
	        print "" 


	def get_meeting_statements(self, filename):
	    with open(filename) as f:
	        return pickle.load(f)


	def get_labels(self, filename, index):

	    # A dictionary of dataframes.  One for each ticker
	    prices = dict()
	    col_names = ['date', 'open', 'high', 'low', 'close', 'volume', 'count', 'WAP']

	    # First get all the prices into respective DF's
	    for file in glob.glob(filename):
	        ticker = file.split('/')[-1].split('.')[0]
	        prices[ticker] = pd.read_csv(file, parse_dates=['date'], infer_datetime_format=True,names=col_names).drop_duplicates()
	        prices[ticker].set_index('date', inplace=True)
	        prices[ticker].index = prices[ticker].index.tz_localize('America/Los_Angeles').tz_convert('America/New_York').tz_localize(None)
	        prices[ticker]['close-MA-4'] = self.fit_moving_average_trend(prices[ticker]['close'], window=4)

	    # Selector to get only prices right before and right after the statement release
	    pre_post_FOMC_time_selector = []
	    for date in index:
	        pre_post_FOMC_time_selector.extend(pd.date_range(date.replace(hour=13, minute=30), periods=2, freq='2 H'))

	    prices_FOMC = dict()
	    for key in prices.keys():
	        prices_FOMC[key] = prices[key].loc[pre_post_FOMC_time_selector][['close-MA-4']].dropna()

	    # each value in this dictionary is a two columns of values.  
	    y_dfs = dict()
	    for key in prices_FOMC:
	        y_dfs[key] = prices_FOMC[key].groupby(prices_FOMC[key].index.date).diff().dropna() 
	        y_dfs[key]['fomc-close-MA-4-pct'] = y_dfs[key]['close-MA-4'] / prices[key].loc[y_dfs[key].index]['close']

	        # Removes the time from the index since now we are left with one prediction a day
	        y_dfs[key].index = y_dfs[key].index.normalize()

	        # Binary column stores 1 if market went up, 0 otherwise
	        y_dfs[key]['binary'] = (y_dfs[key]['close-MA-4'] > 0) * 1
	        y_dfs[key].columns = ['abs-delta', 'pct-delta', 'binary']
	    return y_dfs



	def lemmatize_descriptions(self, meeting_statement):
	    lem = WordNetLemmatizer()
	    lemmatize = lambda d: " ".join(lem.lemmatize(word) for word in d.split())
	    return [lemmatize(desc) for desc in meeting_statement]


	def get_vectorizer(self, meeting_statement, num_features=50000):
	    vect = TfidfVectorizer(max_features=num_features, stop_words='english')
	    return vect.fit(meeting_statement)


	def run_model(self, Model, X_train, X_test, y_train, y_test):
	    m = Model()
	    m.fit(X_train, y_train)
	    y_predict = m.predict(X_test)
	    return accuracy_score(y_test, y_predict), \
	           f1_score(y_test, y_predict), \
	           precision_score(y_test, y_predict), \
	           recall_score(y_test, y_predict)


	def fit_moving_average_trend(self, series, window=6):
	    return series.rolling(window=window,center=False).mean()


	def compare_models(self, X, labels, models):
	    desc_train, desc_test, y_train, y_test = train_test_split(X, labels)

	    print "-----------------------------"
	    print "Without Lemmatization:"
	    self.run_test(models, desc_train, desc_test, y_train, y_test)

	    print "-----------------------------"
	    print "With Lemmatization:"
	    self.run_test(models, self.lemmatize_descriptions(desc_train),
	             	  self.lemmatize_descriptions(desc_test), y_train, y_test)

	    print "-----------------------------"


	def run_test(self, models, desc_train, desc_test, y_train, y_test):
	    vect = self.get_vectorizer(desc_train)
	    X_train = vect.transform(desc_train).toarray()
	    X_test = vect.transform(desc_test).toarray()

	    print "acc\tf1\tprec\trecall"
	    for Model in models:
	        name = Model.__name__
	        acc, f1, prec, rec = self.run_model(Model, X_train, X_test, y_train, y_test)
	        print "%.4f\t%.4f\t%.4f\t%.4f\t%s" % (acc, f1, prec, rec, name)


if __name__ == '__main__':


    meeting_statements = get_meeting_statements('../data/minutes_df.pickle')
    labels = get_labels('../data/*.csv', meeting_statements.index)


    for ticker in labels.keys():
        print ticker
        print "distribution of labels:"
        for i, count in enumerate(np.bincount(labels[ticker]['binary'].values)):
            print "%d: %d" % (i, count)
        models = [LogisticRegression, KNeighborsClassifier, MultinomialNB,
                  RandomForestClassifier, ModeClassifier]
        compare_models(meeting_statements.loc[labels[ticker].index]['statements'].values.tolist(), labels[ticker]['binary'].values, models)
                
        print "" 
        print "" 
        print "" 
