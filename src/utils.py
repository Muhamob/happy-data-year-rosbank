import numpy as np
from math import sqrt
from sklearn.model_selection import cross_validate


def rmse(y_true, y_pred):
    return sqrt( np.mean((y_true-y_pred)**2) )

def cv(clf, X, y, **params):
    scores = cross_validate(clf, X, y, **params)
    for key in scores:
        print(key, scores[key])


