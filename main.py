# https://www.tensorflow.org/tutorials/load_data/csv?hl=en
# https://towardsdatascience.com/build-better-pipelines-with-tensorflow-dataset-328932b16d56
# https://stackoverflow.com/questions/47091726/difference-between-tf-data-dataset-map-and-tf-data-dataset-apply/47096355

# SETUP
import pandas as pd

# Show all columns in pandas
pd.set_option('display.max_columns', None)
import numpy as np
import time

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

# https://www.tensorflow.org/api_docs/python/tf/data/experimental/OptimizationOptions?version=stable
"""

#titanic_file_path = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")

file_path="/home/ppirog/projects/cars-regression/datasets/estimates-data-encoded-ic-hashed_2021-10_1_2021_10_31.csv"

df = pd.read_csv(file_path,delimiter=';',encoding='latin1',encoding_errors='strict',on_bad_lines='skip')
print(df.head())

df.to_csv('/home/ppirog/projects/cars-regression/datasets/output2.csv')


df2 = pd.read_csv('/home/ppirog/projects/cars-regression/datasets/output.csv')

"""

tf.config.threading.set_inter_op_parallelism_threads(64)

dataset = tf.data.experimental.make_csv_dataset(
    '/home/ppirog/projects/cars-regression/datasets/out*.csv',
    batch_size=5,  # Artificially small to make examples easier to show.
    label_name='Amount',
    field_delim=',',
    # Shuffling
    shuffle=True,
    shuffle_buffer_size=1000,
    shuffle_seed=42,
    num_epochs=None,  # number of times this dataset is repeated
    ignore_errors=True,
    # Additional options
    column_names=None,  # optional field names if there is no headers
    column_defaults=None,
    select_columns=None, use_quote_delim=True, na_value='', header=True,
    prefetch_buffer_size=None,
    num_parallel_reads=tf.data.AUTOTUNE,  # or None tf.data.AUTOTUNE
    sloppy=False,
    num_rows_for_inference=100, compression_type=None)


# dataset=dataset.filter(lambda x: x[1] == 'label_comes_here')
def func1(x, y):
    print(type(x), type(y))
    print(f'y={y}')
    print(f'x={x}')
    print(f'x[]={x["Vehicle value"]}')
    print('')
    print('test result:', x["Vehicle value"] > 0.0)
    if x["Vehicle value"] == 0.0:
        return True
    else:
        return False
    # return True




# dataset=dataset.filter(func1)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
# Filtering
dataset = dataset.unbatch().filter(lambda x, y: True if x["Vehicle value"] > 0.0 else False)
dataset = dataset.filter(lambda x, y: True if x["Status"] == 'success' else False)
# dataset = dataset.map(lambda x, y: -x["Vehicle value"])


def func2(x):
    print(x.items())

    return x

dataset = dataset.apply(lambda x: -1* x['ID'])




dataset = dataset.batch(5).prefetch(tf.data.AUTOTUNE)  # tf.data.AUTOTUNE

# dataset = dataset.filter(lambda x,y: y>0)
# check types of input features
input_types = {}
# Get feature columns


start = time.time()
for batch, label in dataset.take(1):
    for key, value in batch.items():
        print(f"{key:20s}: {value}")
        input_types[key] = value.dtype
    print()
    print(f"{'label':20s}: {label}")
    print()
end = time.time()
print(f"Runtime of the program is {end - start}")

feature_columns = list(batch.keys())
# print(f'feature columns: {feature_columns}')
print(f'feature columns: {feature_columns}')
for key in input_types:
    print(f'{key}: {input_types[key]}')


# Dataset is created
###################################################
