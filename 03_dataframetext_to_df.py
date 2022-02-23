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
    filepath = '/home/ppirog/projects/cars-regression/text_df_file.csv'
    sep = ';'
    encoding = 'utf-8'
    df = pd.read_csv(filepath, sep=sep, encoding=encoding, on_bad_lines='skip', low_memory=False)
    print(df.info(verbose=True, null_counts=True))
    print(df.head())

    """
    filepath = '/home/ppirog/projects/cars-regression/preprocessed_file.csv'
    sep = ';'
    encoding = 'utf-8'
    df = pdm.read_csv(filepath, sep=sep, encoding=encoding, on_bad_lines='skip', low_memory=False)

    int_features = ['Doors', 'Gears', 'Mileage', 'Power_kW', 'Younges_driver_age', 'Estimated_Vehicle_value',
                    'Maximum_mass', 'Load_capacity']

    df = df.applymap(lambda x: str(x).replace(" ", "_"))
    df[int_features] = df[int_features].applymap(lambda x: str(x).replace("0", "very_low"))
    df[int_features] = df[int_features].applymap(lambda x: str(x).replace("1", "low"))
    df[int_features] = df[int_features].applymap(lambda x: str(x).replace("2", "medium"))
    df[int_features] = df[int_features].applymap(lambda x: str(x).replace("3", "high"))
    df[int_features] = df[int_features].applymap(lambda x: str(x).replace("4", "very_high"))

    #X_train = df.drop('Amount')

    df = df.astype(str).apply(lambda x: x.name + '_' + x)

    print(df.info(verbose=True, null_counts=True))
    print(df.head())

    df.to_csv(path_or_buf='text_df_file.csv', sep=';', encoding='utf-8', index=False)


    int_features=[]
    for col in df.columns:
        colType = str(df[col].dtype)
        if isinstance(colType,int):
            int_features.append(col)

    print(int_features)
    """

    ""
