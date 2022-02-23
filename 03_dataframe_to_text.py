# SETUP
import ray
import pandas as pd
import modin.pandas as pdm
from sklearn.pipeline import Pipeline
import joblib

# Show all columns in pandas
pd.set_option('display.max_columns', None)
pdm.set_option('display.max_columns', None)

# Make numpy values easier to read.
import numpy as np
from helper_functions import TARGET

np.set_printoptions(precision=3, suppress=True)

if __name__ == '__main__':
    ray.init()
    filepath = '/home/ppirog/projects/cars-regression/preprocessed_file.csv'
    sep = ';'
    encoding = 'utf-8'
    df = pdm.read_csv(filepath, sep=sep, encoding=encoding, on_bad_lines='skip', low_memory=False)

    df = df.iloc[:5000]

    # Get features column names without the last one - Amount because this is target
    features = list(df.columns)[:-1]

    # Get column names only with int value features
    int_features = list(df.select_dtypes([int]).columns)

    # Get column names only with object features
    object_features = list(df.select_dtypes([object]).columns)

    # Remove spaces in object features to create new category
    df[object_features] = df[object_features].applymap(lambda x: str(x).replace(" ", "_"))

    df[int_features] = df[int_features].applymap(lambda x: str(x).replace("0", "very_low"))
    df[int_features] = df[int_features].applymap(lambda x: str(x).replace("1", "low"))
    df[int_features] = df[int_features].applymap(lambda x: str(x).replace("2", "medium"))
    df[int_features] = df[int_features].applymap(lambda x: str(x).replace("3", "high"))
    df[int_features] = df[int_features].applymap(lambda x: str(x).replace("4", "very_high"))

    # For feature columns add column name and string (categorical) value
    df[features] = df[features].astype(str).apply(lambda x: x.name + '_' + x)

    df['Text'] = ''
    for feature in features:
        df['Text'] = df['Text'].map(str) + ' ' + df[feature].map(str)

    # Delete feature columns
    df=df[['Text','Amount']].copy()

    print(df.info(verbose=True, null_counts=True))
    print(df.head())
    df.to_csv(path_or_buf='text_dataset.csv', sep=';', encoding='utf-8', index=False)
