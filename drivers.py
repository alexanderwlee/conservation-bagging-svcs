import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from experiment import experiment

from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.datasets import make_classification
from pmlb import fetch_data


def easy_driver():
    # iris
    print('iris')
    X, y = load_iris(return_X_y=True)
    sc = StandardScaler()
    X = sc.fit_transform(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    experiment(X, y, save_data=True, data_file_name='iris_results')
    print()

    # wine
    print('wine')
    X, y = load_wine(return_X_y=True)
    sc = StandardScaler()
    X = sc.fit_transform(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    experiment(X, y, save_data=True, data_file_name='wine_results')
    print()

    # breats cancer
    print('breast cancer')
    X, y = load_breast_cancer(return_X_y=True)
    sc = StandardScaler()
    X = sc.fit_transform(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    experiment(X, y, save_data=True, data_file_name='breast_cancer_results')
    print()

    # digits
    print('digits')
    X, y = load_digits(return_X_y=True)
    sc = StandardScaler()
    X = sc.fit_transform(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    experiment(X, y, save_data=True, data_file_name='digits_results')
    print()


def clf_driver():
    # 500, 400, 200, 4
    print('500, 400, 200, 4')
    X, y = make_classification(n_samples=500, n_features=400, n_informative=200, n_classes=4, random_state=0)
    experiment(X, y, save_data=True, data_file_name='500_400_200_4_results')
    print()

    # 1000, 100, 90, 2
    print('1000, 100, 90, 2')
    X, y = make_classification(n_samples=1000, n_features=100, n_informative=90, n_classes=2, random_state=0)
    experiment(X, y, save_data=True, data_file_name='1000_100_90_2_results')
    print()

    # 1000, 300, 200, 4
    print('1000, 300, 200, 4')
    X, y = make_classification(n_samples=1000, n_features=300, n_informative=200, n_classes=4, random_state=0)
    experiment(X, y, save_data=True, data_file_name='1000_300_200_4_results')
    print()


def openml_driver():
    # teachingAssistant
    print('teachingAssistant')
    df = pd.read_csv('datasets/openML/teachingAssistant.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    sc = StandardScaler()
    X = sc.fit_transform(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    experiment(X, y, save_data=True, data_file_name='teachingAssistant_results')
    print()

    # monks-problems-2
    print('monks-problems-2')
    df = pd.read_csv('datasets/openML/monks-problems-2.csv')
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    sc = StandardScaler()
    X = sc.fit_transform(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    experiment(X, y, save_data=True, data_file_name='monks-problems-2_results')
    print()

    # one-hundred-plants-margin
    print('one-hundred-plants-margin')
    df = pd.read_csv('datasets/openML/one-hundred-plants-margin.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    sc = StandardScaler()
    X = sc.fit_transform(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    experiment(X, y, save_data=True, data_file_name='one-hundred-plants-margin_results')
    print()


def pmlb_driver():
    # cloud
    print('cloud')
    X, y = fetch_data('cloud', return_X_y=True, local_cache_dir='./datasets/PMLB/')
    sc = StandardScaler()
    X = sc.fit_transform(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    experiment(X, y, save_data=True, data_file_name='cloud_results')
    print()

    # biomed
    print('biomed')
    X, y = fetch_data('biomed', return_X_y=True, local_cache_dir='./datasets/PMLB/')
    sc = StandardScaler()
    X = sc.fit_transform(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    experiment(X, y, save_data=True, data_file_name='biomed_results')
    print()

    # car_evaluation
    print('car_evaluation')
    X, y = fetch_data('car_evaluation', return_X_y=True, local_cache_dir='./datasets/PMLB/')
    sc = StandardScaler()
    X = sc.fit_transform(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    experiment(X, y, save_data=True, data_file_name='car_evaluation_results')
    print()

    # allrep
    print('allrep')
    X, y = fetch_data('allrep', return_X_y=True, local_cache_dir='./datasets/PMLB/')
    sc = StandardScaler()
    X = sc.fit_transform(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    experiment(X, y, save_data=True, data_file_name='allrep_results')
    print()
