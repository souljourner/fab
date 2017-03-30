import numpy as np

class ModeClassifier(object):
    """
    A dumb classification model that simply predicts the most highest occurence outcome.
    """

    def __init__(self):
        self.mode = None

    def fit(self, X_train, y):
        self.mode = np.argmax(np.bincount(y))

    def predict(self, X_test):
        return np.array([self.mode] * len(X_test))


class MeanRegressor(object):
    """
    A dumb classification model that simply predicts the mean outcome.
    """

    def __init__(self):
        self.mean = None

    def fit(self, X_train, y):
        self.mean = np.mean(y)

    def predict(self, X_test):
        return np.array([self.mean] * len(X_test))
