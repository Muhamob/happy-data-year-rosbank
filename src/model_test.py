import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV


class Model():
    def __init__(self, model, X, y):
        self.model_ = model
        self.X = X
        self.y = y

    def train(self, X, y, conf):
        '''
        function for training model 
        '''
        self.model = self.model_(**conf)
        try:
            self.model.predict(X, y)
        except:
            self.model.fit(X, y)
        
    def tuning(self, parameters, kind='grid search'):
        if kind=='grid search':
            clf = GridSearchCV(self.model, parameters, cv=5)
            clf.fit(self.X, self.y)
