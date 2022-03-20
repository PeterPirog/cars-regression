"""
This script:
  read csv files from input_dir_folder
  filter each file by drop rows without amount value
  drop rows with insurance period different than 365 days
  save each filtered file in output_dir_folder
  each files are shuffled and combined in

"""

# pip-autoremove somepackage -y
# SETUP
import ray

from tools.domain_settings import FEATURES, TARGET
from tools.domain_settings import sep, encoding, log10_target
from tools.domain_settings import input_source_dir_folder, output_filtered_dir_folder, output_filtered_file_name

from tools.domain_tools import filter_directory_with_csv, csv_files_from_dir_to_df

if __name__ == '__main__':
    try:
        ray.shutdown()
    except:
        pass
    ray.init()
    select_N_rows = None

    # Filter files from unuseful records
    filter_directory_with_csv(input_source_dir_folder, output_dir_folder=output_filtered_dir_folder,
                              features=FEATURES, target=TARGET, sep=sep, encoding=encoding, log10_target=log10_target)
    # Merge filtered files into single file
    df = csv_files_from_dir_to_df(dir_folder=output_filtered_dir_folder, output_file_name=output_filtered_file_name,
                                  sep=sep, encoding=encoding, select_N_rows=select_N_rows)
    print(df.info(verbose=True, show_counts=True))
    ray.shutdown()

# ray start --head --dashboard-host 12302
