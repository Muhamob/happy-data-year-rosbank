import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import time 
import pickle


# add path to another folders
path = '/home/alexandr/Desktop/study/boosterspro/happy_data_year_rosbank'
sys.path.append(path) if not path in sys.path else None

# from src.osmhandler import Handler
import src.geoutils as geoutils

def to_float(x):
    try:
        return float(x)
    except:
        return np.nan

def concat(a, b):
    '''
    save index of a
    '''
    a_index = a.index.values
    a_ = a.copy()
    b_ = b.copy()
    a_.index = list(range(a_.shape[0]))
    b_.index = list(range(b_.shape[0]))
    res = pd.concat([a_, b_], axis=1)
    res.index = a_index
    return res

def facilities(x_coords, *args, **params):
    with open(path+'/data/working/locations.pickle', 'rb') as f:
        locations = pickle.load(f)
    distances = geoutils.get_nn_distances(x_coords, locations, *args, **params)
    return distances

def cities(X):
    #X['city'] = X[~X.address_rus.isnull()].address_rus.apply(lambda x: x.split(',')[2]) 
    cities = pd.read_csv(path+'/data/working/cities.csv')
    cities = cities[['Город', 'Население', 'Широта', 'Долгота']]
    cities['Население'] = cities.apply(lambda s: to_float(s['Население']), axis = 1)
    cities.columns = ['city', 'city_people', 'city_lat', 'city_long']
    
    X['city'] = X[~X.address_rus.isnull()].address_rus.apply(lambda x: str(x) +',,').apply(lambda x: x.split(',')[2])
    X['city'] = X[~X.city.isnull()].city.apply(lambda s: s[1:])
    X = X.merge(cities.drop_duplicates(subset=['city']),how='left',on='city',)
    X['city_dist'] = X[~X.city_lat.isnull()].apply(
            lambda s: geoutils.distance((s.lat, s.long), (s.city_lat, s.city_long)), axis=1)

    rare_cities = X.city.value_counts()[(X.city.value_counts() < 20) == True].index
    X.city = X.city.apply(lambda x: 'RARE' if x in rare_cities else x)
    X.city = X.city.rank().fillna(-1)
    #rare_cities = X.city.value_counts()[(X.city.value_counts() < 20) ==True].index
    #X.city = X.city.apply(lambda x: 'RARE' if x in rare_cities else x)
    #X.city= X.city.rank().fillna(-1)
    return X

def atms(X):
    dots = X[['lat', 'long']]
    dists = geoutils.get_nn_distances(dots, dots, 'atm_distance', 5, True)
    X = concat(X, dists)
    for i in range(1, 5):
        X['atm_distance_{0}'.format(i)] = np.exp(X['atm_distance_{0}'.format(i)])-1
    return X

def apply(X):
    fac = facilities(X[['lat', 'long']], 'loc', 10)
    result = concat(X, fac)
    result = cities(result)
    result = atms(result)
    return result


if __name__ == '__main__':
    x_coords = np.array( ( (0, 0), (10, 10), (20, 20)))
    start = time.clock()
    dists = facilities(x_coords, 'loc', 10)
    stop = time.clock()
    print(dists.shape, stop-start, 'sec.')

