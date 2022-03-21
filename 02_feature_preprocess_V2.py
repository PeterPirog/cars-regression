#https://towardsdatascience.com/a-simple-trick-to-avoid-running-out-of-memory-d45b0a014ceb
# https://www.youtube.com/watch?v=L7ZmnFV8fzc
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
import psutil

# Show all columns in pandas
pd.set_option('display.max_columns', None)
pdm.set_option('display.max_columns', None)

import numpy as np
from tools.domain_settings import TARGET, FEATURES_CATEGORICAL, FEATURES_NUMERICAL

# Feature engine imports
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.discretisation import EqualFrequencyDiscretiser
from feature_engine.selection import DropConstantFeatures

from sklearn.ensemble import HistGradientBoostingClassifier

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

FEATURES_TO_REMOVE_SPACE = ['Generation_type', 'Version_type_name', 'Commune']
# These features are originally numerical but converted to categories
#FEATURES_INT_TO_OBJ = ['Gears', 'Doors', 'Key_pairs']

if __name__ == '__main__':
    ray.init()

    #filepath = '/home/ppirog/projects/cars-regression/filtered_file_10000.csv'
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
    #X[FEATURES_INT_TO_OBJ] = X[FEATURES_INT_TO_OBJ].astype(object)

    # Get column names only with object features
    object_features = list(X.select_dtypes([object]).columns)

    # Remove spaces in object features to create new category
    X[object_features] = X[object_features].applymap(lambda x: str(x).replace(" ", "_"))

    # Convert numeric
    X[FEATURES_NUMERICAL] = X[FEATURES_NUMERICAL].astype(np.float32)

    #newdf = X.select_dtypes(include=np.number).columns.tolist()
    #print(newdf)


    #FEATURES_NUMERICAL_SELECTED=['Capacity_ccm', 'Mileage', 'Power_kW', 'Younges_driver_age']
    # numerical values nan impute
    mmi = MeanMedianImputer(imputation_method='mean',variables=FEATURES_NUMERICAL)  # , variables=FEATURES_NUMERICAL
    # categorical values nan impute
    ci = CategoricalImputer(imputation_method='missing', fill_value='Missing', variables=None,
                            return_object=False, ignore_format=False)

    # Remove constant feature to prevent Nan values after EqualFrequencyDiscretiser
    # numerical values discretize
    efd = EqualFrequencyDiscretiser(variables=None, q=number_of_bins_for_numeric_features, return_object=True,
                                    return_boundaries=False)

    dcf98 = DropConstantFeatures(variables=None, tol=drop_constant_threshold, missing_values='ignore')

    pipe = Pipeline([
        # ('DropConstantFeatures1', dcf98),
        ('MeanMedianImputer', mmi),
        ('CategoricalImputer', ci),
        ('DropConstantFeatures2', dcf98),  # Prevent from standard deviation equal 0
        ('EqualFrequencyDiscretiser', efd),
        ('DropConstantFeatures3', dcf98),
    ])
    print(f'X type is: {type(X)}')

    X = dcf98.fit_transform(X, y).copy()

    # Impute numeric features
    medians = {}
    for i,feature in enumerate(FEATURES_NUMERICAL):
        if feature in X.columns:
            print(f'{i+1}/{len(FEATURES_NUMERICAL)}: Median imputer for the feature:{feature}')
            mmi = MeanMedianImputer(imputation_method='median', variables=[feature])
            X = mmi.fit_transform(X, y)
        else:
            print(f'Feature:{feature} not exist in dataframe !')

    # Impute categorical features by 'missing' label

    for i,feature in enumerate(FEATURES_CATEGORICAL):
        if feature in X.columns:
            print(f'{i+1}/{len(FEATURES_NUMERICAL)}: Categorical missing imputer for the feature:{feature}')
            ci = CategoricalImputer(imputation_method='missing', fill_value='Missing', variables=[feature],
                                    return_object=False, ignore_format=False)
            X = ci.fit_transform(X, y)
        else:
            print(f'Feature:{feature} not exist in dataframe !')

    out=X

    out = dcf98.fit_transform(out, y)
    out = efd.fit_transform(out, y)
    out = dcf98.fit_transform(out, y)
    """

    ŻLE WYLICZANA MEDIANA !!! bo liczy zera
    
    out = mmi.fit_transform(X, y)
    out = imp_mean.fit_transform(X, y)
import joblib
from ray.util.joblib import register_ray
register_ray()
with joblib.parallel_backend('ray'):
    search.fit(digits.data, digits.target)
    
    
    out=mmi.fit_transform(X,y)
    out=ci.fit_transform(out,y)
    out = dcf98.fit_transform(out, y)
    out = efd.fit_transform(out, y)
    out = dcf98.fit_transform(out, y) 
    
    
    
    print('1:',psutil.virtual_memory().percent)
    #print('2:', psutil.virtual_memory().percent)
    #X1 = dcf98.fit_transform(X, y).copy()
    #print(X1.info(verbose=True, show_counts=True))
    #del X
    print('3:', psutil.virtual_memory().percent)
    X2 = mmi.fit_transform(X, y).copy()
    print('4:', psutil.virtual_memory().percent)
    #del X1
    print('5:', psutil.virtual_memory().percent)
    X3 = ci.fit_transform(X2, y).copy()
    print('6:', psutil.virtual_memory().percent)
    del X2
    print('7:', psutil.virtual_memory().percent)
    X4 = dcf98.fit_transform(X3, y).copy()
    print('8:', psutil.virtual_memory().percent)
    del X3
    print('9:', psutil.virtual_memory().percent)
    X5 = efd.fit_transform(X4, y).copy()
    print('10:', psutil.virtual_memory().percent)
    del X4
    print('11:', psutil.virtual_memory().percent)
    out = dcf98.fit_transform(X5, y).copy()
    print('12:', psutil.virtual_memory().percent)
    del X5
    print('13:', psutil.virtual_memory().percent)

    #out = pipe.fit_transform(X, y)

    out = pd.concat([out, y], axis=1)
    print('14:', psutil.virtual_memory().percent)
    """
    print(out.info(verbose=True, show_counts=True))
    print(out.head())

    # Save files
    # joblib.dump(pipe, 'preprocessing_pipeline.pkl')
    out.to_csv(path_or_buf=preprocessed_filename, sep=sep, encoding=encoding, index=False)
