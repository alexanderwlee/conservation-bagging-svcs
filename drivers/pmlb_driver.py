import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from experiment import experiment

from pmlb import fetch_data


def run():
    # cloud
    X, y = fetch_data('cloud', return_X_y=True, local_cache_dir='./datasets/PMLB/')
    sc = StandardScaler()
    X = sc.fit_transform(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    experiment(X, y, save_data=True, data_file_name='cloud_results.csv')

    # biomed
    X, y = fetch_data('biomed', return_X_y=True, local_cache_dir='./datasets/PMLB/')
    sc = StandardScaler()
    X = sc.fit_transform(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    experiment(X, y, save_data=True, data_file_name='biomed_results.csv')

    # car_evaluation
    X, y = fetch_data('car_evaluation', return_X_y=True, local_cache_dir='./datasets/PMLB/')
    sc = StandardScaler()
    X = sc.fit_transform(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    experiment(X, y, save_data=True, data_file_name='car_evaluation_results.csv')

    # allrep
    X, y = fetch_data('allrep', return_X_y=True, local_cache_dir='./datasets/PMLB/')
    sc = StandardScaler()
    X = sc.fit_transform(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    experiment(X, y, save_data=True, data_file_name='allrep_results.csv')


if __name__ == '__main__':
    run()