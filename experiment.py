import random
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


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


def lexiworkshop(estimator_pred_dict, X_test, y_test, n_estimators):
    workshop = []
    for i in range(n_estimators):
        estimator = lexicase(estimator_pred_dict, X_test, y_test, i)
        workshop.append(estimator)
    return workshop


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


def experiment(X, y, n_repl=10, n_folds=5, n_runs=10, n_svcs=100, save_data=False, data_file_name='test_data_file'):
    df_dict = {
        'rep': [], 'bag_svc': [], 'factory': [], 'super_ensemble': [], 
        'workshop_100': [], 'workshop_300': [], 'workshop_500': []
    } 
    for rep in range(n_repl):
        print('rep:', rep)
        df_dict['rep'].append(rep)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=rep)
        fold = 0
        sum_fold_bag_svc_score = 0
        sum_fold_factory_score = 0
        sum_fold_super_ensemble_score = 0
        sum_fold_workshop_100_score = 0
        sum_fold_workshop_300_score = 0
        sum_fold_workshop_500_score = 0
        for train_index, test_index in kf.split(X):
            print('\tfold:', fold)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            sum_run_bag_svc_score = 0
            factory = []
            super_ensemble = []
            for run in range(n_runs):
                print('\t\trun:', run)
                bag_svc = BaggingClassifier(base_estimator=SVC(), n_estimators=n_svcs, n_jobs=-1, random_state=0)
                bag_svc.fit(X_train, y_train)
                bag_svc_score = bag_svc.score(X_test, y_test)
                print('\t\t\tbag_svc_score:', bag_svc_score)
                sum_run_bag_svc_score += bag_svc_score
                factory += bag_svc.estimators_
                super_ensemble.append(bag_svc)
            mean_run_bag_svc_score = sum_run_bag_svc_score / n_runs
            sum_fold_bag_svc_score += mean_run_bag_svc_score

            factory_score = maj_vote_score(factory, X_test, y_test)
            print('\t\tfactory_score:', factory_score)
            sum_fold_factory_score += factory_score

            super_ensemble_score = maj_vote_score(super_ensemble, X_test, y_test)
            print('\t\tsuper_ensemble_score:', super_ensemble_score)
            sum_fold_super_ensemble_score += super_ensemble_score

            factory_pred_dict = get_factory_pred_dict(factory, X_test)

            workshop_100 = lexiworkshop(factory_pred_dict, X_test, y_test, 100)
            workshop_100_score = maj_vote_score(workshop_100, X_test, y_test)
            print('\t\tworkshop_100_score:', workshop_100_score)
            sum_fold_workshop_100_score += workshop_100_score

            workshop_300 = lexiworkshop(factory_pred_dict, X_test, y_test, 300)
            workshop_300_score = maj_vote_score(workshop_300, X_test, y_test)
            print('\t\tworkshop_300_score:', workshop_300_score)
            sum_fold_workshop_300_score += workshop_300_score

            workshop_500 = lexiworkshop(factory_pred_dict, X_test, y_test, 500)
            workshop_500_score = maj_vote_score(workshop_500, X_test, y_test)
            print('\t\tworkshop_500_score:', workshop_500_score)
            sum_fold_workshop_500_score += workshop_500_score

            fold += 1
        
        mean_fold_bag_svc_score = sum_fold_bag_svc_score / n_folds
        mean_fold_factory_score = sum_fold_factory_score / n_folds
        mean_fold_super_ensemble_score = sum_fold_super_ensemble_score / n_folds
        mean_fold_workshop_100_score = sum_fold_workshop_100_score / n_folds
        mean_fold_workshop_300_score = sum_fold_workshop_300_score / n_folds
        mean_fold_workshop_500_score = sum_fold_workshop_500_score / n_folds
        df_dict['bag_svc'].append(mean_fold_bag_svc_score)
        df_dict['factory'].append(mean_fold_factory_score)
        df_dict['super_ensemble'].append(mean_fold_super_ensemble_score)
        df_dict['workshop_100'].append(mean_fold_workshop_100_score)
        df_dict['workshop_300'].append(mean_fold_workshop_300_score)
        df_dict['workshop_500'].append(mean_fold_workshop_500_score)
        
        if save_data:
            df = pd.DataFrame(df_dict)
            df.to_csv(f'results/{data_file_name}.csv', index=False)


if __name__ == '__main__':
    # use for debugging purposes
    pass