import numpy as np
import pandas as pd

def test_function():
    print('Test function correctly imported')
    return 'Test function correctly imported'



def dataframe_analysis(df: pd.dataframe) -> pd.dataframe:
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
    output.to_excel("Columns_analysis.xlsx", sheet_name='Columns')
    return output