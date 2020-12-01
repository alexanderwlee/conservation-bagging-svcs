import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from experiment import experiment


def run():
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

    # madelon
    print('madelon')
    df = pd.read_csv('datasets/openML/madelon.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    sc = StandardScaler()
    X = sc.fit_transform(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    experiment(X, y, save_data=True, data_file_name='madelon_results')
    print()


if __name__ == '__main__':
    run()