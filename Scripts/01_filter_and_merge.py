"""
This script:
  read csv files from input_dir_folder
  filter each file by drop rows without amount value
  drop rows with insurance period different than 365 days
  save each filtered file in output_dir_folder
  each files are shuffled and combined in

"""
import argparse
#import modin.pandas as pdm
# import modin.pandas as pd
#import pandas as pd
# SETUP
import ray

# Show all columns in pandas and modin pandas
#pd.set_option('display.max_columns', None)
#pdm.set_option('display.max_columns', None)

#import numpy as np

# Make numpy values easier to read.
#np.set_printoptions(precision=3, suppress=True)
from helper_functions import FEATURES, TARGET
from helper_functions import filter_directory_with_csv, csv_files_from_dir_to_df
# import modin.pandas as pdm
# import modin.pandas as pd
# import pandas as pd
# SETUP
import ray

# Make numpy values easier to read.
# np.set_printoptions(precision=3, suppress=True)
from helper_functions import FEATURES, TARGET
from helper_functions import filter_directory_with_csv, csv_files_from_dir_to_df

# CUSTOM IMPORTS OF OWN FUNCTIONS AND CLASSES
from tools.domain_settings import FEATURES,TARGET,sep,encoding
from tools.general_tools import dir_path,dataframe_analysis





# Show all columns in pandas and modin pandas
# pd.set_option('display.max_columns', None)
# pdm.set_option('display.max_columns', None)
# import numpy as np

parser=argparse.ArgumentParser(description='Script to rectify CSV files by:'
                                           '1. Remove rows without target value'
                                           '2. Remove rows with insurance period different than 365 days')


parser.add_argument('--source_dir',type=dir_path, help='Path to directory with csv files',
                    default='/ai-data/estimates-data-2021_1_2022_1')

parser.add_argument('--sep',type=str,help='Separator used in csv files typically , or ;',default=';')
parser.add_argument('--encoding',type=str,help='Type of encoding from csv file, default value utf-8',default='utf-8')


if __name__ == '__main__':
    ray.init()

    # Filtering csv files
    input_dir_folder = '/ai-data/estimates-data-2021_1_2022_1'
    output_dir_folder = '/home/ppirog/projects/cars-regression/filtered_dataset3'
    output_file_name = 'filtered_file.csv'

    sep = ';'
    encoding = 'utf-8'







    filter_directory_with_csv(input_dir_folder, output_dir_folder=output_dir_folder,
                              features=FEATURES, target=TARGET, sep=sep, encoding=encoding,
                              use_modin_pd=True, log10_target=True)

    csv_files_from_dir_to_df(dir_folder=output_dir_folder, output_file_name=output_file_name,
                             sep=sep, encoding=encoding, use_modin_pd=True)

