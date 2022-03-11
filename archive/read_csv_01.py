# https://www.tensorflow.org/tutorials/load_data/csv?hl=en
# https://towardsdatascience.com/build-better-pipelines-with-tensorflow-dataset-328932b16d56
# https://stackoverflow.com/questions/47091726/difference-between-tf-data-dataset-map-and-tf-data-dataset-apply/47096355

# SETUP
# import pandas as pd
import ray

# import modin.pandas as pd
import pandas as pd
import modin.pandas as pdm


pd.set_option('display.max_columns', None)
# Show all columns in pandas

import numpy as np
from helper_functions import FEATURES, FEATURES_NUMERICAL, FEATURES_CATEGORICAL, TARGET
from helper_functions import dataframe_analysis, create_small_csv,csv_files_from_dir_to_df

# FEature engine imports
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.transformation import YeoJohnsonTransformer
from feature_engine.discretisation import EqualFrequencyDiscretiser
from feature_engine.selection import SmartCorrelatedSelection, DropConstantFeatures, DropCorrelatedFeatures

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

FEATURES_TO_REMOVE_SPACE = ['Generation_type', 'Version_type_name', 'Commune']





if __name__ == '__main__':
    ray.init()
    filepath = '/ai-data/estimates-data-2021_1_2022_1/estimates-data-encoded-ic-hashed-2021_2022-1_1_2022_1_31.csv'

    data_folder = '/ai-data/estimates-data-2021_1_2022_1/*.csv'

    # All files and directories ending with .txt and that don't begin with a dot:

    df = csv_files_from_dir_to_df('/ai-data/estimates-data-2021_1_2022_1', use_modin_pd=True, output_file_name='text_data.csv')




    # create_small_csv(original_file_path=filepath, n_rows=10000, small_file_path='small_file.csv', sep=';')

    #df = pd.read_csv('small_file.csv', sep=';', encoding='utf-8', on_bad_lines='skip')
    df = pd.read_csv('text_data.csv', sep=';', encoding='utf-8', on_bad_lines='skip')

    #print(df.info(verbose=True, null_counts=True))
    #print(df.head())


    # get only useful columns

    # REDUCE number of rows
    df = df[FEATURES + TARGET]

    # Drop amount values 0 or negative
    df = df[df['Amount'] > 0.0]

    # Take only  365 days period insurances
    df = df[df['Insured_days'] == 365]  #

    # split df to X and y
    y = df.pop(TARGET[0])
    X = df.copy()
    del df

    #print(X.info(verbose=True, null_counts=True))
    #print(X.head())
    
    # numerical values nan impute
    mmi = MeanMedianImputer(imputation_method='median', variables=FEATURES_NUMERICAL)
    # categorical values nan impute
    ci = CategoricalImputer(imputation_method='missing', fill_value='Missing', variables=FEATURES_CATEGORICAL,
                            return_object=False, ignore_format=False)
    # change distribution shape Yeo-Johnson
    yeo = YeoJohnsonTransformer(variables=FEATURES_NUMERICAL)

    # numerical values discretization
    efd = EqualFrequencyDiscretiser(variables=None, q=5, return_object=True,
                                    return_boundaries=False)

    dcf = DropConstantFeatures(variables=None, tol=0.98, missing_values='ignore')
    dcorrf = DropCorrelatedFeatures(variables=None, method='pearson', threshold=0.8, missing_values='ignore')

    scs = SmartCorrelatedSelection(variables=FEATURES_NUMERICAL, method='pearson', threshold=0.8,
                                   missing_values='ignore', selection_method='missing_values',
                                   estimator=None, scoring='roc_auc', cv=3)

    # Pipeline
    X = mmi.fit_transform(X)
    X = ci.fit_transform(X)
    X = dcf.fit_transform(X)
    X = dcorrf.fit_transform(X)

    for feature in FEATURES_TO_REMOVE_SPACE:
        X[feature] = X[feature].map(lambda x: x.replace(' ', ''))

    X = efd.fit_transform(X)
    X = X.astype(str).apply(lambda x: x.name + '_' + x)

    # print(X.info(verbose=True, null_counts=True))
    print(X.head(10))
    # print(X['Load_capacity'])

    X.to_csv(path_or_buf='text_data.csv', sep=';', encoding='utf-8', index=False)

    
    X = efd.fit_transform(X)
    #X = yeo.fit_transform(X)
    

    print(X.shape)
    print(X.info(verbose=True, null_counts=True))
    print(X.head())

    


