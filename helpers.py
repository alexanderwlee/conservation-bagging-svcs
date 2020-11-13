import random
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


def lexicase(estimator_pred_dict, X_test, y_test, shuffle_random_seed):
    random.seed(0)
    np.random.seed(shuffle_random_seed)

    candidates = estimator_pred_dict.copy()

    shuffle_indices = np.arange(len(y_test))
    np.random.shuffle(shuffle_indices)
    for est, y_pred in candidates.items():
        candidates[est] = y_pred[shuffle_indices]
    y_test = y_test[shuffle_indices]

    for example_index in range(len(y_test)):
        best_candidates = {}
        for est, y_pred in candidates.items():
            if y_pred[example_index] == y_test[example_index]:
                best_candidates[est] = y_pred
        if len(best_candidates) > 0:
            candidates = best_candidates
        if len(candidates) == 1:
            return list(candidates.keys())[0]
    return random.choice(list(candidates.keys()))


def lexigarage(estimator_pred_dict, X_test, y_test, n_estimators):
    garage = []
    for i in range(n_estimators):
        estimator = lexicase(estimator_pred_dict, X_test, y_test, i)
        garage.append(estimator)
    return garage


def get_factory_pred_dict(factory, X_test):
    factory_pred_dict = {}
    for svc in factory:
        y_pred = svc.predict(X_test)
        factory_pred_dict[svc] = y_pred
    return factory_pred_dict


def maj_vote(estimators, X_test):
    predictions = np.asarray([est.predict(X_test) for est in estimators]).T # matrix of examples x estimator predictions
    maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=predictions)
    return maj_vote


def maj_vote_score(estimators, X_test, y_test):
    y_pred = maj_vote(estimators, X_test)
    return accuracy_score(y_test, y_pred)


def experiment(X, y, n_repl=30, n_folds=5, n_runs=20, n_svcs=100):
    for rep in range(n_repl):
        print('rep:', rep + 1)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=rep)
        fold = 0
        for train_index, test_index in kf.split(X):
            print('\tfold:', fold + 1)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            factory = []
            super_ensemble = []
            for run in range(n_runs):
                print('\t\trun:', run + 1)
                svc_bag = BaggingClassifier(base_estimator=SVC(), n_estimators=n_svcs, n_jobs=-1, random_state=0)
                svc_bag.fit(X_train, y_train)
                svc_bag_score = svc_bag.score(X_test, y_test)
                print('\t\tsvc_bag_score:', svc_bag_score)
                factory += svc_bag.estimators_
                super_ensemble.append(svc_bag)

            factory_score = maj_vote_score(factory, X_test, y_test)
            print('\tfactory_score:', factory_score)

            super_ensemble_score = maj_vote_score(super_ensemble, X_test, y_test)
            print('\tsuper_ensemble_score:', super_ensemble_score)

            factory_pred_dict = get_factory_pred_dict(factory, X_test)

            garage_100 = lexigarage(factory_pred_dict, X_test, y_test, 100)
            garage_100_score = maj_vote_score(garage_100, X_test, y_test)
            print('\tgarage_100_score:', garage_100_score)

            garage_300 = lexigarage(factory_pred_dict, X_test, y_test, 300)
            garage_300_score = maj_vote_score(garage_300, X_test, y_test)
            print('\tgarage_300_score:', garage_300_score)

            garage_1000 = lexigarage(factory_pred_dict, X_test, y_test, 1000)
            garage_1000_score = maj_vote_score(garage_1000, X_test, y_test)
            print('\tgarage_1000_score:', garage_1000_score)

            fold += 1
