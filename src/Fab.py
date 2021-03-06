# baseline models
from dumb_predictors import MeanRegressor, ModeClassifier

# NLP
import textacy
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Classification
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# Regression Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

# Testing and optimization
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics.regression import mean_absolute_error, mean_squared_error, r2_score

import pandas as pd
import numpy as np
import pickle
import datetime as dt
import glob
import sys
import os.path

# from keras.models import Sequential
# from keras.layers import Dense, Activation

np.set_printoptions(formatter={'float_kind':lambda x: "%.4fdf" % x})

class Fab(object):
    """
    Fab - FOMC Analysis Bot - 0.1 - A work in progress

    A class to create predictions on the FOMC meeting minutes

    """
    def __init__(self, regression=True):

        self.regression = regression
        self.labels = None
        self.last_test = None
        self.last_predict = None
        self.meeting_statements = self.get_meeting_statements('../data/minutes.pickle')
        self.prices = self.get_prices()
        self.set_labels()

        self.all_data_df = None  # a data frame that holds all the X and y's 
        self.X = None
        self.y = None

        print("Available tickers:")
        print(", ".join(list(self.labels.keys())))


    def get_meeting_statements(self, filename='../data/minutes_df.pickle'):
        with open(filename, 'rb') as f:
            print("Loading saved statements")
            return pickle.load(f)


    def get_prices(self, filename='../data/*.csv', pickle_path='../data/prices.pickle', refresh=False):

        """
        Gets prices from the 16 instruments.  It loads from the saved pickle file unless refresh is True

        Input: filename, pickle_path, and refresh.  If refresh is True, it will skip loading of the prices
               from the pickle path and instead read from CSVs in the data directory specified in filename.
        """

        if refresh is False and os.path.exists(pickle_path):
            print("Loading saved prices")
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)

        # A dictionary of dataframes.  One for each 
        # note the timezone issues need to be rechecked prior to running live
        prices = dict()
        col_names = ['date', 'open', 'high', 'low', 'close', 'volume', 'count', 'WAP']
        for file in glob.glob(filename):
            ticker = file.split('/')[-1].split('.')[0]
            prices[ticker] = pd.read_csv(file, parse_dates=['date'], infer_datetime_format=True,names=col_names).drop_duplicates()
            prices[ticker].set_index('date', inplace=True)
            prices[ticker].index = prices[ticker].index.tz_localize('America/Los_Angeles').tz_convert('America/New_York').tz_localize(None)
            prices[ticker]['close-MA-4'] = self.fit_moving_average_trend(prices[ticker]['close'], window=4)
        self.write_prices()
        return prices


    def write_prices(self, pickle_path='../data/prices.pickle'):
        with open(pickle_path, "wb") as f:
            pickle.dump(self.prices, f)


    def set_labels(self, index=None, pickle_path='../data/labels.pickle', refresh=False):
        """
        Gets the labels from the 16 instruments.  It loads from the saved pickle file unless refresh is True

        Input: filename, pickle_path, and refresh.  If refresh is True, it will skip loading of the prices
               from the pickle path and instead read from CSVs in the data directory specified in filename.

        """

        if refresh is False and os.path.exists(pickle_path):
            print("Loading saved labels")
            with open(pickle_path, 'rb') as f:
                self.labels = pickle.load(f)
                return self.labels

        if index is None:
            index = self.meeting_statements.index

        # Selector to get only prices right before and right after the statement release
        pre_post_FOMC_time_selector = []
        for date in index:
            pre_post_FOMC_time_selector.extend(pd.date_range(date.replace(hour=13, minute=30), periods=2, freq='2 H'))

        prices_FOMC = dict()
        for key in self.prices.keys():
            prices_FOMC[key] = self.prices[key].loc[pre_post_FOMC_time_selector][['close-MA-4']].dropna()

        # each value in this dictionary is a two columns of values.  
        y_dfs = dict()
        for key in prices_FOMC:
            y_dfs[key] = prices_FOMC[key].groupby(prices_FOMC[key].index.date).diff().dropna() 
            y_dfs[key]['fomc-close-MA-4-pct'] = (y_dfs[key]['close-MA-4'] / self.prices[key].loc[y_dfs[key].index]['close'])

            # Removes the time from the index since now we are left with one prediction a day
            y_dfs[key].index = y_dfs[key].index.normalize()

            # Binary column stores 1 if market went up, 0 otherwise
            y_dfs[key]['binary'] = (y_dfs[key]['close-MA-4'] > 0) * 1
            y_dfs[key].columns = ['abs-delta', 'pct-delta', 'binary']
        self.write_labels()
        self.labels = y_dfs


    def write_labels(self, pickle_path='../data/labels.pickle'):
        with open(pickle_path, "wb") as f:
            pickle.dump(self.labels, f)


#     def lemmatize_descriptions(self, meeting_statements):
#         lemmatized = []
#         for statement in meeting_statements:
#             doc = nlp(statement)
#             lemmatized.append(" ".join([token.lemma_ for token in doc]))
#         return lemmatized
            

#     def get_vectorizer(self, meeting_statement, num_features=5000):
#         vect = TfidfVectorizer(max_features=num_features, stop_words='english')
#         return vect.fit(meeting_statement)


    def fit_moving_average_trend(self, series, window=6):
        return series.rolling(window=window,center=False).mean()


    def run_model(self, model, X_train, X_test, y_train, y_test):

        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)

        self.last_test = y_test
        self.last_predict = y_predict

        # fpr, tpr, thresholds = roc_curve(y_test, y_predict)
        if self.regression:
            return r2_score(y_test, y_predict), \
                   mean_squared_error(y_test, y_predict), \
                   mean_absolute_error(y_test, y_predict)
        else:
            return roc_auc_score(y_test, y_predict), \
                   accuracy_score(y_test, y_predict), \
                   f1_score(y_test, y_predict), \
                   precision_score(y_test, y_predict), \
                   recall_score(y_test, y_predict)


    def run_test(self, models, desc_train, desc_test, y_train, y_test):
#         vect = self.get_vectorizer(desc_train)
#         X_train = vect.transform(desc_train).toarray()
#         X_test = vect.transform(desc_test).toarray()

        X_train = desc_train
        X_test = desc_test
        
        if self.regression:
            print("r2\tmse\tmae")
            for model in models:
                name = model.__class__.__name__
                r2, mse, mae = self.run_model(model, X_train, X_test, y_train, y_test)
                print("%.4f\t%.4f\t%.4f\t%s" % (r2, mse, mae, name))
            return mse
        else:
            print("auc\tacc\tf1\tprec\trecall")
            for model in models:
                name = model.__class__.__name__
                auc_score, acc, f1, prec, rec = self.run_model(model, X_train, X_test, y_train, y_test)
                print("%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%s" % (auc_score, acc, f1, prec, rec, name))
            return acc


    def compare_models(self, desc, y, models, splits=6):

        # desc_train, desc_test, y_train, y_test = train_test_split(desc, labels)

        tscv = TimeSeriesSplit(n_splits=splits)

        for train_index, test_index in tscv.split(desc):
            desc_train, desc_test = desc[train_index], desc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print("")
            print("Length: train {}, test {}".format(len(train_index), len(test_index)))
            print("Balance: train {}, test {}".format(np.sum(y_train)/float(len(y_train)), 
                                                      np.sum(y_test)/float(len(y_test))) )
            print ("-----------------------------")
#             print "Without Lemmatization:"
            self.run_test(models, desc_train, desc_test, y_train, y_test)

#             print("-----------------------------")
#             print("With Lemmatization:")
#             self.run_test(models, self.lemmatize_descriptions(desc_train),
#                            self.lemmatize_descriptions(desc_test), y_train, y_test)
#             print("-----------------------------")


    def predict(self, meeting_statement, timestamp, tickers):
        """
        Predicts the output of the given meeting_statement to the tickers in tickers.  The prerequisite
        of this method is that prices of the instruments are already preloaded.

        Designed for live prediction during FOMC meeting days 

        meeting_statement -- a new meeting statement
        """

        # Wait until 1:55 PM
        # keep updating prices until 2 PM.
        # While refreshing FOMC site for minute:
        #   statement = get statement
        # if prices have already been updated
        #   X = self.get_features(meeting_statement, timestamp, tickers) #one row matrix
        #   return self.model.predict(X)
        pass

    def anticipate_and_predict(self, tickers=['SHY-USD-TRADES']):
        """

        """
        # Wait until 1:45 PM
        # keep updating prices until 2 PM.
        # While refreshing FOMC site for minute:
        #   statement = get statement
        #   timestamp = current time
        # set closing prices as current time
        # return predict(statement, timestamp, tickers)
        pass


    def run(self, tickers=['SHY-USD-TRADES']):
        """
        Adds the features into X for prediction and then runs several different models to compare
        their performance.  Sets the best model as the instance's model.

        Implementation needed 

        """

        # build NLP df with features
        corpus = textacy.Corpus(lang='en')
        corpus.add_texts(self.meeting_statements['statements'].values, n_threads=4)
        self.doc_term_matrix, self.id2term = textacy.vsm.doc_term_matrix((doc.to_terms_list(ngrams=3, 
                                                                                  named_entities=True, 
                                                                                  as_strings=True) for doc in corpus),
                                                               weighting='tfidf', 
                                                               normalize=False, 
                                                               smooth_idf=True, 
                                                               # min_df=2, 
                                                               # max_df=0.95
                                                              )
        # take index from self.labels and build features based on prices

        # merge the NLP features with the prices features on the index of each ticker

        self.X = self.doc_term_matrix

        # Parameter tuning on RF

        if self.regression:
            models = [MeanRegressor(),
                      GradientBoostingRegressor(), 
                      RandomForestRegressor(n_estimators=300)]
            
        else:
            models = [ModeClassifier(),
                      RandomForestClassifier(n_estimators=300)]

        for ticker in tickers:
            if ticker in self.labels.keys():
                print(ticker)
                print("distribution of labels:")
                for i, count in enumerate(np.bincount(self.labels[ticker]['binary'].values)):
                    print("%d: %d" % (i, count))
                if self.regression:
                    # min = self.labels[ticker]['pct-delta'].values.min()
                    # max = self.labels[ticker]['pct-delta'].values.max()
                    self.compare_models(self.meeting_statements.loc[self.labels[ticker].index]['statements'], 
                                    self.labels[ticker]['abs-delta'].values, models)
                else:
                    selector = np.array([True if i in self.labels[ticker].index else False for i in self.meeting_statements.index])
                    self.compare_models(self.X.toarray()[selector], 
                                        self.labels[ticker]['binary'].values, models)
