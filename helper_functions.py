import numpy as np
import os
import pandas as pd
pd.pandas.set_option('display.max_rows', None,'display.max_columns', None)

def test_function():
    print('Test function correctly imported')
    return 'Test function correctly imported'



def dataframe_analysis(df,xls_filename='Columns_analysis.xlsx'):
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
    print(output)
    return output, catCols, numCols

if __name__ == '__main__':


    FILE_PATH = 'dataset.csv'
    DIR_DATA = '/content/drive/MyDrive/datasets/cars/'

    # load dataset
    df = pd.read_csv(FILE_PATH, delimiter=';')
    #print(df.head())
    dataframe_analysis(df)
