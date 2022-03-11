"""
This script:
1. read filtered_file.csv
2. initialize sklearn pipeline, target ='Amount' is separated
3. impute numerical values with median
4.  Impute categorical values with missing label
5. split numerical data to 5 bins with equal quantiles
6. drop quasi constant features, operation is made 3 times to reduce size of dataframe before each step
7. combine transformed features with target value
8. save dataframe to preprocessed_file.csv
"""
# TODO Gears should be int not range value


# SETUP
import ray
import pandas as pd
import modin.pandas as pdm
from sklearn.pipeline import Pipeline
import joblib

# Show all columns in pandas
pd.set_option('display.max_columns', None)
pdm.set_option('display.max_columns', None)


import numpy as np
from helper_functions import TARGET

# Feature engine imports
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.discretisation import EqualFrequencyDiscretiser
from feature_engine.selection import DropConstantFeatures

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

FEATURES_TO_REMOVE_SPACE = ['Generation_type', 'Version_type_name', 'Commune']
# These features are originally numerical but converted to categories
FEATURES_INT_TO_OBJ = ['Gears', 'Doors', 'Key_pairs']


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

    # Convert individually some int features to categories
    X[FEATURES_INT_TO_OBJ] = X[FEATURES_INT_TO_OBJ].astype(object)

    # numerical values nan impute
    mmi = MeanMedianImputer(imputation_method='median', variables=None)
    # categorical values nan impute
    ci = CategoricalImputer(imputation_method='missing', fill_value='Missing', variables=None,
                            return_object=False, ignore_format=False)

    # Remove constant feature to prevent Nan values after EqualFrequencyDiscretiser
    # numerical values discretize
    efd = EqualFrequencyDiscretiser(variables=None, q=number_of_bins_for_numeric_features, return_object=True,
                                    return_boundaries=False)

    dcf98 = DropConstantFeatures(variables=None, tol=drop_constant_threshold, missing_values='ignore')

    pipe = Pipeline([
        ('DropConstantFeatures1', dcf98),
        ('MeanMedianImputer', mmi),
        ('CategoricalImputer', ci),
        ('DropConstantFeatures2', dcf98),  # Prevent from standard deviation equal 0
        ('EqualFrequencyDiscretiser', efd),
        ('DropConstantFeatures3', dcf98),
    ])

    out= pipe.fit_transform(X, y)
    out = pd.concat([out, y], axis=1)

    print(out.info(verbose=True, show_counts=True))
    print(out.head())

    # Save files
    joblib.dump(pipe, 'preprocessing_pipeline.pkl')
    out.to_csv(path_or_buf=preprocessed_filename, sep=sep, encoding=encoding, index=False)

