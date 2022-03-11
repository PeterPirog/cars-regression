# https://www.tensorflow.org/tutorials/load_data/csv?hl=en
# https://towardsdatascience.com/build-better-pipelines-with-tensorflow-dataset-328932b16d56
# https://stackoverflow.com/questions/47091726/difference-between-tf-data-dataset-map-and-tf-data-dataset-apply/47096355

# SETUP
# import pandas as pd
import ray
import glob
import os

# import modin.pandas as pd
import pandas as pd
import modin.pandas as pdm
from sklearn.pipeline import Pipeline
import joblib

pd.set_option('display.max_columns', None)
pdm.set_option('display.max_columns', None)
# Show all columns in pandas

import numpy as np
from helper_functions import FEATURES, FEATURES_NUMERICAL, FEATURES_CATEGORICAL, TARGET
from helper_functions import dataframe_analysis, create_small_csv, csv_files_from_dir_to_df
from helper_functions import filter_directory_with_csv

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

    filepath = '/home/ppirog/projects/cars-regression/filtered_file.csv'
    preprocessed_filename = 'preprocessed_file.csv'
    sep = ';'
    encoding = 'utf-8'
    number_of_bins_for_numeric_features = 5
    drop_constant_threshold = 0.98

    df = pd.read_csv(filepath, sep=sep, encoding=encoding, on_bad_lines='skip', low_memory=False)

    # split df to X and y
    y = df.pop(TARGET[0])
    X = df.copy()
    del df

    # numerical values nan impute
    mmi = MeanMedianImputer(imputation_method='median', variables=FEATURES_NUMERICAL)
    # categorical values nan impute
    ci = CategoricalImputer(imputation_method='missing', fill_value='Missing', variables=FEATURES_CATEGORICAL,
                            return_object=False, ignore_format=False)
    #Remove constant feature to prevent Nan values after EqualFrequencyDiscretiser
    dcf1 = DropConstantFeatures(variables=None, tol=1.0, missing_values='ignore')
    # numerical values discretization
    efd = EqualFrequencyDiscretiser(variables=None, q=number_of_bins_for_numeric_features, return_object=True,
                                    return_boundaries=False)

    dcf2 = DropConstantFeatures(variables=None, tol=drop_constant_threshold, missing_values='ignore')

    pipe = Pipeline([
        ('MeanMedianImputer', mmi),
        ('CategoricalImputer', ci),
        ('DropConstantFeatures1', dcf1), #Prevent from standard deviation equal 0
        ('EqualFrequencyDiscretiser', efd),
        ('DropConstantFeatures', dcf2),
    ])

    out = pipe.fit_transform(X, y)
    out = pd.concat([out, y], axis=1)

    print(out.info(verbose=True, null_counts=True))
    print(out.head())

    # Save files

    joblib.dump(pipe, 'preprocessing_pipeline.pkl')
    out.to_csv(path_or_buf=preprocessed_filename, sep=sep, encoding=encoding, index=False)

"""




    # numerical values discretization
    efd = EqualFrequencyDiscretiser(variables=None, q=5, return_object=True,
                                    return_boundaries=False)

    dcf = DropConstantFeatures(variables=None, tol=0.98, missing_values='ignore')
    dcorrf = DropCorrelatedFeatures(variables=None, method='pearson', threshold=0.8, missing_values='ignore')


    # Pipeline
    X = mmi.fit_transform(X)
    X = ci.fit_transform(X)
    X = dcf.fit_transform(X)
    X = dcorrf.fit_transform(X)

    for feature in FEATURES_TO_REMOVE_SPACE:
        X[feature] = X[feature].map(lambda x: x.replace(' ', ''))

    # Discretize numerical values
    X = efd.fit_transform(X)
    # Convert all categories to labels
    X = X.astype(str).apply(lambda x: x.name + '_' + x)

    # print(X.info(verbose=True, null_counts=True))
    # Add amount log10 value
    X = X.merge(y_log.rename('Amount_log10'), left_index=True, right_index=True)
    print(X.head())

    X.to_csv(path_or_buf='text2_data.csv', sep=';', encoding='utf-8', index=False)

"""
