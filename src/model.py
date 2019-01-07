import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
import sys
path = '/home/alexandr/Desktop/study/boosterspro'
sys.path.append(path) if not path in sys.path else None
from src.utils import rmse


class AveragingModel(BaseEstimator, ClassifierMixin):
    def __init__(self, models, weights=None, pretrained=False):
        super().__init__()
        self.models = models
        self.weights = weights if not weights is None else np.ones(len(models))*1./len(models)
        self.trained = pretrained


    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        for model in self.models:
            model.fit(X_train, y_train)
            score = rmse(y_test, model.predict(X_test))
            print('model: ', model, '\n', 'score: ', score, '\n')
        self.trained = True

        return self


    def predict(self, X):
        assert self.trained, 'Train your model first'
        pred = 0
        for model, weight in zip(self.models, self.weights):
            pred += weight*model.predict(X)

        return pred

    def score(self, X, y):
        return rmse(self.predict(X), y)
