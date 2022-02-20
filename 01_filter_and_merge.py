"""
This script:
  read csv files from input_dir_folder
  filter each file by drop rows without amount value
  drop rows with insurance period different than 365 days
  save each filtered file in output_dir_folder
  each files are shuffled and combined in

"""

# SETUP
import ray

# import modin.pandas as pd
import pandas as pd
import modin.pandas as pdm

# Show all columns in pandas and modin pandas
pd.set_option('display.max_columns', None)
pdm.set_option('display.max_columns', None)

import numpy as np
from helper_functions import FEATURES, TARGET
from helper_functions import csv_files_from_dir_to_df
from helper_functions import filter_directory_with_csv

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

if __name__ == '__main__':
    ray.init()

    # Filtering csv files
    input_dir_folder = '/ai-data/estimates-data-2021_1_2022_1'
    output_dir_folder = '/home/ppirog/projects/cars-regression/filtered_dataset'
    output_file_name = 'filtered_file.csv'

    filter_directory_with_csv(input_dir_folder, output_dir_folder=output_dir_folder,
                              features=FEATURES, target=TARGET, sep=';', encoding='utf-8',
                              use_modin_pd=True, log10_target=True)

    csv_files_from_dir_to_df(dir_folder=output_dir_folder, output_file_name=output_file_name,
                             sep=';', encoding='utf-8', use_modin_pd=True)
