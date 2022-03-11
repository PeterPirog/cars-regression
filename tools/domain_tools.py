import glob
import os
import numpy as np
import tensorflow as tf
import modin.pandas as pd

pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split

from tools.domain_settings import sep, encoding

try:
    import ray

    ray.init()
    # import modin.pandas as pdm
    pd.set_option('display.max_columns', None)
except:
    pass


def filter_single_csv(input_filename_path, output_filename_path, features, target, log10_target=True,
                      sep=';', encoding='utf-8', shuffle=True):
    """

    :param input_filename_path:  string,path to input file to filter
    :param output_filename_path: string, path to file or filename to save output filtered file
    :param features: list of strings, dafault=None, list of column in original dataframe to copy to output dataframe,
    if None, all columns are copied to the input dataframe
    :param target: string or list with single string, default = None, name of column with target value
    :param log10_target: bool, default=True, if True target value is converted to log10(x), it reduces long tails in the
    input data distributions
    :param sep: string, default=';', separator used in input csv files
    :param encoding: string,default='utf-8', methos of string encoding
    :param shuffle: bool,default=True, if true rows inside the CSV file are shuffled
    :return: dataframe with file
    """

    df = pd.read_csv(input_filename_path, sep=sep, encoding=encoding, on_bad_lines='skip', low_memory=False)

    # FILTER FILE
    # Remove unusefull columns if are defined
    if isinstance(target, str):
        target = [target]

    if (target is not None) & (features is not None):
        df = df[features + target].copy()

    # Drop amount values 0 or negative
    df = df[df['Amount'] > 0.0]

    # Take only  365 days period insurances
    df = df[df['Insured_days'] == 365]  #

    # Shuffle rows
    if shuffle:
        df = df.sample(frac=1)

    # Transform output values by log10(x)
    if log10_target:
        df['Amount'] = np.log10(df['Amount'])

    df.to_csv(path_or_buf=output_filename_path, sep=sep, encoding=encoding, index=False)
    return df


def loadCarData(source_file, number_rows=None,
                batch_size=32, return_form='nparray', random_state=42):
    """
                Function creates train and validation data as datasets or np arrays
    :param source_file: 
    :param number_rows: 
    :param batch_size: 
    :param return_form: 'nparray' or 'datasets' or 'full'
    :param random_state: 
    :return: 
    """

    df = pd.read_csv(source_file, sep=';', encoding='utf-8', on_bad_lines='skip', low_memory=False)

    if number_rows is not None:
        df = df.iloc[:number_rows]

    train_sentences, val_sentences, train_labels, val_labels = train_test_split(df["Text"].to_numpy(),
                                                                                df["Amount"].to_numpy(),
                                                                                test_size=0.05,
                                                                                random_state=random_state)
    if return_form == 'nparray':
        # Return np arrays
        print(
            f'Function returns numpy arrays in form: train_sentences, val_sentences, train_labels, val_labels =loadCarData(...)')
        return train_sentences, val_sentences, train_labels, val_labels
    else:
        # Return datasets
        print(f'Function returns datasets in form: train_ds,val_ds=loadCarData(...)')
        print(f'train_sentences shape: {train_sentences.shape}')
        print(f'train_sentences type: {type(train_sentences)}')

        # train X & y
        train_sentences_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_sentences, tf.string))
        train_labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_labels, tf.dtypes.bfloat16))

        # test X & y
        val_sentences_ds = tf.data.Dataset.from_tensor_slices(tf.cast(val_sentences, tf.string))
        val_labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(val_labels, tf.dtypes.bfloat16))

        # combine features qith targets into train and validation datasets
        train_ds = tf.data.Dataset.zip(
            (
                train_sentences_ds,
                train_labels_ds
            ))

        val_ds = tf.data.Dataset.zip(
            (
                val_sentences_ds,
                val_labels_ds
            ))

        # Optimize datasets
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        buffer_size = train_ds.cardinality().numpy()

        train_ds = train_ds.shuffle(buffer_size=buffer_size) \
            .batch(batch_size=batch_size, drop_remainder=True) \
            .cache() \
            .prefetch(AUTOTUNE)

        val_ds = val_ds.shuffle(buffer_size=buffer_size) \
            .batch(batch_size=batch_size, drop_remainder=True) \
            .cache() \
            .prefetch(AUTOTUNE)
        if return_form == 'datasets':
            return train_ds, val_ds
        else:
            return train_ds, val_ds, train_sentences, val_sentences, train_labels, val_labels


def filter_directory_with_csv(input_dir_folder, output_dir_folder, features, target, sep=';', encoding='utf-8',
                              log10_target=True):
    input_dir_folder = input_dir_folder + '/*.csv'

    glob_files = glob.glob(input_dir_folder)
    N_files = len(glob_files)

    for i, path in enumerate(glob_files):
        filename_filtered = 'filtered_' + os.path.basename(path)
        path_filtered = os.path.join(output_dir_folder, filename_filtered)

        print(f'Reading from file {i + 1}/{N_files}: {path}')
        filter_single_csv(input_filename_path=path,
                          output_filename_path=path_filtered,
                          features=features, target=target,
                          sep=sep, encoding=encoding, log10_target=log10_target)


def csv_files_from_dir_to_df(dir_folder, output_file_name='big_text_file.csv', sep=';', encoding='utf-8'):
    dir_folder = dir_folder + '/*.csv'
    glob_files = glob.glob(dir_folder)
    N_files = len(glob_files)

    frames = []
    for i, path in enumerate(glob_files):
        print(f'Reading from file {i + 1}/{N_files}: {path}')

        frames.append(pd.read_csv(path, sep=sep, encoding=encoding, on_bad_lines='skip', low_memory=False))

    df = pd.concat(frames, ignore_index=True)

    print('Shuffling output file ....')
    df = df.sample(frac=1)

    print('Saving output file ....')
    df.to_csv(path_or_buf=output_file_name, sep=sep, encoding=encoding, index=False)

    print(f'Created file: {output_file_name}')
    return df


##### TEMPORARY IMPORTS
from tools.domain_settings import FEATURES, TARGET
from tools.general_tools import dataframe_analysis

if __name__ == '__main__':
    input_file = '/ai-data/estimates-data-2021_1_2022_1/estimates-data-encoded-ic-hashed-2021_2022-1_1_2022_1_31.csv'
    output_file = '/home/ppirog/projects/cars-regression/filtered_dataset2/output.csv'

    df = filter_single_csv(input_filename_path=input_file,
                           output_filename_path=output_file,
                           features=FEATURES,
                           target=TARGET, log10_target=True,
                           sep=sep, encoding=encoding, shuffle=True)

    print(df.head())
    print(df.info(verbose=True, null_counts=True))
    print(dataframe_analysis(df, xls_filename='small_analysis.xlsx'))
