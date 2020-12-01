from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from experiment import experiment

from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits


def run():
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
    experiment(X, y, save_data=True, data_file_name='breat_cancer_results')
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


if __name__ == '__main__':
    run()