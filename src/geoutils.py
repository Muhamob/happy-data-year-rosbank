import numpy as np
from math import cos, sin, sqrt, atan2, radians
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


def distance(x, y):
    '''
    input: floats
    output: float, distance in kilometers
    '''
    R = 6373.0
    lat_a, long_a, lat_b, long_b = map(radians, (*x, *y))
    dlon = long_b - long_a
    dlat = lat_b - lat_a
    a = sin(dlat/2)**2 + cos(lat_a) * cos(lat_b) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def get_nn_distances(coords, obj_coords, name, n_neighbors=1, 
        return_ind=False):
    knc = KNeighborsClassifier(metric=distance)
    knc.fit(obj_coords, np.ones(obj_coords.shape[0]))
    distances, indices = knc.kneighbors(X=coords, n_neighbors=n_neighbors)
    output = {'{0}_{1}'.format(name, i): distances[:, i] for i in range(n_neighbors)}
    for i in range(1, n_neighbors):
        output['{0}_{1}'.format(name, i)] = distances[:, i]
        if return_ind:
            output['{0}_ind_{1}'.format(name, i)] = indices[:, i]
    
    return pd.DataFrame(output)
