import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from experiment import experiment


def run():
    # teachingAssistant
    df = pd.read_csv('datasets/openML/teachingAssistant.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    sc = StandardScaler()
    X = sc.fit_transform(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    experiment(X, y, save_data=True, data_file_name='teachingAssistant_results')

    # monks-problems-2
    df = pd.read_csv('datasets/openML/monks-problems-2.csv')
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    sc = StandardScaler()
    X = sc.fit_transform(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    experiment(X, y, save_data=True, data_file_name='monks-problems-2_results')

    # madelon
    df = pd.read_csv('datasets/openML/madelon.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    sc = StandardScaler()
    X = sc.fit_transform(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    experiment(X, y, save_data=True, data_file_name='madelon_results')


if __name__ == '__main__':
    run()