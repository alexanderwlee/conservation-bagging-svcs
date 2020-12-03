import pandas as pd


def get_combined_results():
    dfs = []
    df_source_names = []
    df_names = []
    
    # Easy
    df_source_names += ['Easy'] * 4

    iris_results = pd.read_csv('results/iris_results.csv')
    dfs.append(iris_results)
    df_names.append('iris')

    wine_results = pd.read_csv('results/wine_results.csv')
    dfs.append(wine_results)
    df_names.append('wine')
    
    breast_cancer_results = pd.read_csv('results/breast_cancer_results.csv')
    dfs.append(breast_cancer_results)
    df_names.append('breast cancer')
    
    digits_results = pd.read_csv('results/digits_results.csv')
    dfs.append(digits_results)
    df_names.append('digits')
    
    # Clf
    df_source_names += ['Clf'] * 3

    clf_500_400_200_4_results = pd.read_csv('results/500_400_200_4_results.csv')
    dfs.append(clf_500_400_200_4_results)
    df_names.append('500_400_200_4')
    
    clf_1000_100_90_2_results = pd.read_csv('results/1000_100_90_2_results.csv')
    dfs.append(clf_1000_100_90_2_results)
    df_names.append('1000_100_90_2')
    
    clf_1000_300_200_4_results = pd.read_csv('results/1000_300_200_4_results.csv')
    dfs.append(clf_1000_300_200_4_results)
    df_names.append('1000_300_200_4')

    # openML
    df_source_names += ['OpenML'] * 3

    teaching_assistant_results = pd.read_csv('results/teachingAssistant_results.csv')
    dfs.append(teaching_assistant_results)
    df_names.append('teachingAssistant')

    monk_problems_2_results = pd.read_csv('results/monks-problems-2_results.csv')
    dfs.append(monk_problems_2_results)
    df_names.append('monk-problems-2')

    one_hundred_plants_margin_results = pd.read_csv('results/one-hundred-plants-margin_results.csv')
    dfs.append(one_hundred_plants_margin_results)
    df_names.append('one-hundred-plants-margin')

    # PMLB
    df_source_names += ['PMLB'] * 4

    cloud_results = pd.read_csv('results/cloud_results.csv')
    dfs.append(cloud_results)
    df_names.append('cloud')

    biomed_results = pd.read_csv('results/biomed_results.csv')
    dfs.append(biomed_results)
    df_names.append('biomed')

    car_evaluation_results = pd.read_csv('results/car_evaluation_results.csv')
    dfs.append(car_evaluation_results)
    df_names.append('car_evaluation')

    allrep_results = pd.read_csv('results/allrep_results.csv')
    dfs.append(allrep_results)
    df_names.append('allrep')

    mean_dfs = [df.describe().iloc[[1], 1:] for df in dfs]
    combined_results = pd.concat(mean_dfs, ignore_index=True)
    combined_results = combined_results.multiply(100)
    combined_results.insert(0, 'source', df_source_names)
    combined_results.insert(1, 'dataset', df_names)
    combined_results = combined_results.round(1)

    return combined_results

