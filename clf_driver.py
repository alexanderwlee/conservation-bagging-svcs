from experiment import experiment

from sklearn.datasets import make_classification


def run():
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


if __name__ == '__main__':
    run()