import numpy as np
import tensorflow as tf
import random

import glob
import os
import pandas as pd
import modin.pandas as pdm
from sklearn.model_selection import train_test_split

pd.pandas.set_option('display.max_rows', None, 'display.max_columns', None)

FEATURES_CATEGORICAL = ['IC_slug',
                        'Package',
                        'Body',
                        'Brand',
                        'Fuel_type',
                        'Gearbox_type',
                        'Generation_type',
                        'Import_status',
                        'LPG_system',
                        'Model_type',
                        'Ownership_mode',
                        'Steering_wheel_side',
                        'Subject_type',
                        'Voivodeship',
                        'Marital_status',
                        'Version_type_name',
                        'AC_PREVIOUS_INSURANCE_COMPANY_PREVIOUS_YEAR',
                        'TPL_PREVIOUS_INSURANCE_COMPANY_PREVIOUS_YEAR',
                        'Commune',
                        'County',
                        'Customer_Contact_ZIP',
                        'Place',
                        'Parking_zip_code']

FEATURES_NUMERICAL = ['Capacity_ccm',
                      'Doors',
                      'Gears',
                      'Key_pairs',
                      'Mileage',
                      'Power_kW',
                      'Younges_driver_age',
                      'AC_CLAIMS_COUNT__1_YEAR',
                      'AC_CLAIMS_COUNT__10_YEAR',
                      'AC_CLAIMS_COUNT__11_YEAR',
                      'AC_CLAIMS_COUNT__12_YEAR',
                      'AC_CLAIMS_COUNT__13_YEAR',
                      'AC_CLAIMS_COUNT__14_YEAR',
                      'AC_CLAIMS_COUNT__15_YEAR',
                      'AC_CLAIMS_COUNT__2_YEAR',
                      'AC_CLAIMS_COUNT__3_YEAR',
                      'AC_CLAIMS_COUNT__4_YEAR',
                      'AC_CLAIMS_COUNT__5_YEAR',
                      'AC_CLAIMS_COUNT__6_YEAR',
                      'AC_CLAIMS_COUNT__7_YEAR',
                      'AC_CLAIMS_COUNT__8_YEAR',
                      'AC_CLAIMS_COUNT__9_YEAR',
                      'AC_CLAIMS_COUNT_PREVIOUS_YEAR',
                      'AC_CLAIMS_WITHOUT_REFUSAL_COUNT__1_YEAR',
                      'AC_CLAIMS_WITHOUT_REFUSAL_COUNT__10_YEAR',
                      'AC_CLAIMS_WITHOUT_REFUSAL_COUNT__11_YEAR',
                      'AC_CLAIMS_WITHOUT_REFUSAL_COUNT__12_YEAR',
                      'AC_CLAIMS_WITHOUT_REFUSAL_COUNT__13_YEAR',
                      'AC_CLAIMS_WITHOUT_REFUSAL_COUNT__14_YEAR',
                      'AC_CLAIMS_WITHOUT_REFUSAL_COUNT__15_YEAR',
                      'AC_CLAIMS_WITHOUT_REFUSAL_COUNT__2_YEAR',
                      'AC_CLAIMS_WITHOUT_REFUSAL_COUNT__3_YEAR',
                      'AC_CLAIMS_WITHOUT_REFUSAL_COUNT__4_YEAR',
                      'AC_CLAIMS_WITHOUT_REFUSAL_COUNT__5_YEAR',
                      'AC_CLAIMS_WITHOUT_REFUSAL_COUNT__6_YEAR',
                      'AC_CLAIMS_WITHOUT_REFUSAL_COUNT__7_YEAR',
                      'AC_CLAIMS_WITHOUT_REFUSAL_COUNT__8_YEAR',
                      'AC_CLAIMS_WITHOUT_REFUSAL_COUNT__9_YEAR',
                      'AC_CLAIMS_WITHOUT_REFUSAL_COUNT_PREVIOUS_YEAR',
                      'AC_EVENTS_COUNT__1_YEAR',
                      'AC_EVENTS_COUNT__10_YEAR',
                      'AC_EVENTS_COUNT__11_YEAR',
                      'AC_EVENTS_COUNT__12_YEAR',
                      'AC_EVENTS_COUNT__13_YEAR',
                      'AC_EVENTS_COUNT__14_YEAR',
                      'AC_EVENTS_COUNT__15_YEAR',
                      'AC_EVENTS_COUNT__2_YEAR',
                      'AC_EVENTS_COUNT__3_YEAR',
                      'AC_EVENTS_COUNT__4_YEAR',
                      'AC_EVENTS_COUNT__5_YEAR',
                      'AC_EVENTS_COUNT__6_YEAR',
                      'AC_EVENTS_COUNT__7_YEAR',
                      'AC_EVENTS_COUNT__8_YEAR',
                      'AC_EVENTS_COUNT__9_YEAR',
                      'AC_EVENTS_COUNT_PREVIOUS_YEAR',
                      'AC_EVENTS_WHERE_CULPRIT_COUNT__1_YEAR',
                      'AC_EVENTS_WHERE_CULPRIT_COUNT__10_YEAR',
                      'AC_EVENTS_WHERE_CULPRIT_COUNT__11_YEAR',
                      'AC_EVENTS_WHERE_CULPRIT_COUNT__12_YEAR',
                      'AC_EVENTS_WHERE_CULPRIT_COUNT__13_YEAR',
                      'AC_EVENTS_WHERE_CULPRIT_COUNT__14_YEAR',
                      'AC_EVENTS_WHERE_CULPRIT_COUNT__15_YEAR',
                      'AC_EVENTS_WHERE_CULPRIT_COUNT__2_YEAR',
                      'AC_EVENTS_WHERE_CULPRIT_COUNT__3_YEAR',
                      'AC_EVENTS_WHERE_CULPRIT_COUNT__4_YEAR',
                      'AC_EVENTS_WHERE_CULPRIT_COUNT__5_YEAR',
                      'AC_EVENTS_WHERE_CULPRIT_COUNT__6_YEAR',
                      'AC_EVENTS_WHERE_CULPRIT_COUNT__7_YEAR',
                      'AC_EVENTS_WHERE_CULPRIT_COUNT__8_YEAR',
                      'AC_EVENTS_WHERE_CULPRIT_COUNT__9_YEAR',
                      'AC_EVENTS_WHERE_CULPRIT_COUNT_PREVIOUS_YEAR',
                      'AC_INSURANCE_COMPANIES_COUNT__10_YEAR',
                      'AC_INSURANCE_COMPANIES_COUNT__15_YEAR',
                      'AC_INSURANCE_COMPANIES_COUNT__5_YEAR',
                      'AC_INSURANCE_PERIOD__5_YEAR',
                      'AC_POLICY_COUNT__7_YEAR',
                      'TPL_CLAIMS_COUNT__1_YEAR',
                      'TPL_CLAIMS_COUNT__10_YEAR',
                      'TPL_CLAIMS_COUNT__11_YEAR',
                      'TPL_CLAIMS_COUNT__12_YEAR',
                      'TPL_CLAIMS_COUNT__13_YEAR',
                      'TPL_CLAIMS_COUNT__14_YEAR',
                      'TPL_CLAIMS_COUNT__15_YEAR',
                      'TPL_CLAIMS_COUNT__2_YEAR',
                      'TPL_CLAIMS_COUNT__3_YEAR',
                      'TPL_CLAIMS_COUNT__4_YEAR',
                      'TPL_CLAIMS_COUNT__5_YEAR',
                      'TPL_CLAIMS_COUNT__6_YEAR',
                      'TPL_CLAIMS_COUNT__7_YEAR',
                      'TPL_CLAIMS_COUNT__8_YEAR',
                      'TPL_CLAIMS_COUNT__9_YEAR',
                      'TPL_CLAIMS_COUNT_PREVIOUS_YEAR',
                      'TPL_CLAIMS_WITHOUT_REFUSAL_COUNT__1_YEAR',
                      'TPL_CLAIMS_WITHOUT_REFUSAL_COUNT__10_YEAR',
                      'TPL_CLAIMS_WITHOUT_REFUSAL_COUNT__11_YEAR',
                      'TPL_CLAIMS_WITHOUT_REFUSAL_COUNT__12_YEAR',
                      'TPL_CLAIMS_WITHOUT_REFUSAL_COUNT__13_YEAR',
                      'TPL_CLAIMS_WITHOUT_REFUSAL_COUNT__14_YEAR',
                      'TPL_CLAIMS_WITHOUT_REFUSAL_COUNT__15_YEAR',
                      'TPL_CLAIMS_WITHOUT_REFUSAL_COUNT__2_YEAR',
                      'TPL_CLAIMS_WITHOUT_REFUSAL_COUNT__3_YEAR',
                      'TPL_CLAIMS_WITHOUT_REFUSAL_COUNT__4_YEAR',
                      'TPL_CLAIMS_WITHOUT_REFUSAL_COUNT__5_YEAR',
                      'TPL_CLAIMS_WITHOUT_REFUSAL_COUNT__6_YEAR',
                      'TPL_CLAIMS_WITHOUT_REFUSAL_COUNT__7_YEAR',
                      'TPL_CLAIMS_WITHOUT_REFUSAL_COUNT__8_YEAR',
                      'TPL_CLAIMS_WITHOUT_REFUSAL_COUNT__9_YEAR',
                      'TPL_CLAIMS_WITHOUT_REFUSAL_COUNT_PREVIOUS_YEAR',
                      'TPL_EVENTS_COUNT__1_YEAR',
                      'TPL_EVENTS_COUNT__10_YEAR',
                      'TPL_EVENTS_COUNT__11_YEAR',
                      'TPL_EVENTS_COUNT__12_YEAR',
                      'TPL_EVENTS_COUNT__13_YEAR',
                      'TPL_EVENTS_COUNT__14_YEAR',
                      'TPL_EVENTS_COUNT__15_YEAR',
                      'TPL_EVENTS_COUNT__2_YEAR',
                      'TPL_EVENTS_COUNT__3_YEAR',
                      'TPL_EVENTS_COUNT__4_YEAR',
                      'TPL_EVENTS_COUNT__5_YEAR',
                      'TPL_EVENTS_COUNT__6_YEAR',
                      'TPL_EVENTS_COUNT__7_YEAR',
                      'TPL_EVENTS_COUNT__8_YEAR',
                      'TPL_EVENTS_COUNT__9_YEAR',
                      'TPL_EVENTS_COUNT_PREVIOUS_YEAR',
                      'TPL_INSURANCE_COMPANIES_COUNT__10_YEAR',
                      'TPL_INSURANCE_COMPANIES_COUNT__15_YEAR',
                      'TPL_INSURANCE_COMPANIES_COUNT__5_YEAR',
                      'TPL_INSURANCE_PERIOD__5_YEAR',
                      'TPL_POLICY_COUNT__7_YEAR',
                      'Estimated_Vehicle_value',
                      'Maximum_mass',
                      'Load_capacity',
                      'Insured_days']

FEATURES = FEATURES_CATEGORICAL + FEATURES_NUMERICAL


def test_function():
    print('Test function correctly imported')
    return 'Test function correctly imported'


TARGET = ['Amount']


# define features


def dataframe_analysis(df, xls_filename='Columns_analysis.xlsx'):
    # Delete old analysis file if exist
    if os.path.exists(xls_filename):
        os.remove(xls_filename)

    # Analysis of  unique values
    output = []

    for col in df.columns:
        nonNull = len(df) - np.sum(pd.isna(df[col]))
        unique = df[col].nunique()
        colType = str(df[col].dtype)

        output.append([col, nonNull, unique, colType])

    output = pd.DataFrame(output)
    output.columns = ['colName', 'non-null values', 'unique', 'dtype']
    output = output.sort_values(by='unique', ascending=False)
    output.to_excel(xls_filename, sheet_name='Columns')

    # Return categorical columns
    # get all categorical columns in the dataframe
    catCols = [col for col in df.columns if df[col].dtype == "O"]
    numCols = [col for col in df.columns if not df[col].dtype == "O"]
    # print(output)
    return output, catCols, numCols


def create_small_csv(original_file_path, n_rows=1000, small_file_path='small_file.csv', sep=';'):
    df = pd.read_csv(original_file_path, sep=sep, encoding='utf-8', on_bad_lines='skip')
    df = df.iloc[:n_rows]
    df.to_csv(path_or_buf=small_file_path, sep=sep, encoding='utf-8', index=False)


def csv_files_from_dir_to_df(dir_folder, output_file_name='big_text_file.csv', sep=';', encoding='utf-8',
                             use_modin_pd=False):
    dir_folder = dir_folder + '/*.csv'
    glob_files = glob.glob(dir_folder)
    N_files = len(glob_files)

    frames = []
    for i, path in enumerate(glob_files):
        print(f'Reading from file {i + 1}/{N_files}: {path}')
        if use_modin_pd:
            frames.append(pdm.read_csv(path, sep=sep, encoding=encoding, on_bad_lines='skip', low_memory=False))
        else:
            frames.append(pd.read_csv(path, sep=sep, encoding=encoding, on_bad_lines='skip', low_memory=False))
    if use_modin_pd:
        df = pdm.concat(frames, ignore_index=True)
    else:
        df = pd.concat(frames, ignore_index=True)

    print('Shuffling output file ....')
    df = df.sample(frac=1)

    print('Saving output file ....')
    df.to_csv(path_or_buf=output_file_name, sep=sep, encoding=encoding, index=False)

    print(f'Created file: {output_file_name}')
    return df


def filter_single_csv(input_filename_path, output_filename_path, features, target, use_modin_pd=True, log10_target=True,
                      sep=';', encoding='utf-8'):
    if use_modin_pd:
        df = pdm.read_csv(input_filename_path, sep=sep, encoding=encoding, on_bad_lines='skip', low_memory=False)
    else:
        df = pd.read_csv(input_filename_path, sep=sep, encoding=encoding, on_bad_lines='skip', low_memory=False)

    # FILTER FILE
    # Remove unusefull columns
    df = df[features + target].copy()

    # Drop amount values 0 or negative
    df = df[df['Amount'] > 0.0]

    # Take only  365 days period insurances
    df = df[df['Insured_days'] == 365]  #

    # Shuffle rows
    df = df.sample(frac=1)

    # Transform output values by log10(x)
    if log10_target:
        df['Amount'] = np.log10(df['Amount'])

    df.to_csv(path_or_buf=output_filename_path, sep=sep, encoding=encoding, index=False)
    # print(df.head())
    # print(df.info(verbose=True, null_counts=True))


def filter_directory_with_csv(input_dir_folder, output_dir_folder, features, target, sep=';', encoding='utf-8',
                              use_modin_pd=True, log10_target=True):
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
                          sep=sep, encoding=encoding,
                          use_modin_pd=use_modin_pd, log10_target=log10_target)


def loadCarData(source_file='/home/ppirog/projects/cars-regression/text_dataset.csv',number_rows=None):
    df = pdm.read_csv(source_file, sep=';', encoding='utf-8', on_bad_lines='skip', low_memory=False)

    if number_rows is not None:
        df = df.iloc[:number_rows]

    train_sentences, val_sentences, train_labels, val_labels = train_test_split(df["Text"].to_numpy(),
                                                                                df["Amount"].to_numpy(),
                                                                                test_size=0.05,
                                                                                random_state=42)


    with open('/home/ppirog/projects/cars-regression/npdata/train_sentences.npy', 'wb') as f:
        np.save(f, train_sentences)

    with open('/home/ppirog/projects/cars-regression/npdata/val_sentences.npy', 'wb') as f:
        np.save(f, val_sentences)

    with open('/home/ppirog/projects/cars-regression/npdata/train_labels.npy', 'wb') as f:
        np.save(f, train_labels)

    with open('/home/ppirog/projects/cars-regression/npdata/val_labels.npy', 'wb') as f:
        np.save(f, val_labels)

    return train_sentences, val_sentences, train_labels, val_labels

def setSeed(seed,threads=1):
    os.environ['PYTHONHASHSEED'] = '0'
    tf.config.set_soft_device_placement(True)

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ["OMP_NUM_THREADS"] = str(threads)

    tf.config.threading.set_inter_op_parallelism_threads(threads)
    tf.config.threading.set_intra_op_parallelism_threads(threads)

def load_trained_vectorizer(dirpath_vect_model):
    #filepath_vect_model = "tokenizer_model_10k"
    loaded_model = tf.keras.models.load_model(dirpath_vect_model)
    loaded_vectorizer = loaded_model.layers[0]
    return loaded_model,loaded_vectorizer


def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True,seed=None,
                              shuffle_size=10000,batch_size=1):
    assert (train_split + test_split + val_split) == 1
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    #ds=ds.cache()
    #ds=ds.prefetch(AUTOTUNE)

    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=seed)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)   #.batch(batch_size).cache().prefetch(AUTOTUNE)
    val_ds = ds.skip(train_size).take(val_size) #.batch(batch_size) #.cache().prefetch(AUTOTUNE)
    test_ds = ds.skip(train_size).skip(val_size) #.batch(batch_size) #.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds


if __name__ == '__main__':
    train_sentences, val_sentences, train_labels, val_labels = loadCarData()
    pass
