import numpy as np
import os
import pandas as pd
pd.pandas.set_option('display.max_rows', None,'display.max_columns', None)

def test_function():
    print('Test function correctly imported')
    return 'Test function correctly imported'

TARGET='Amount'
# define features
FEATURES=['Body',
          'Brand',
          'Fuel-type',
          'Gearbox-type',
          'Generation-type',
          'Import-status',
          'LPG-system',
          'Model-type',
          'Ownership-mode',
          'Package',
          'Steering-wheel-side',
          'Subject-type',
          'Voivodeship',
          'Marital-status',
          'Version-type-name',
          'AC_PREVIOUS_INSURANCE_COMPANY_PREVIOUS_YEAR',
          'TPL_PREVIOUS_INSURANCE_COMPANY_PREVIOUS_YEAR',
          'Commune',
          'County',
          'Customer-Contact-ZIP',
          'Place',
          'Parking-zip-code',
          'Capacity-ccm',
          'Doors',
          'Gears',
          'Key-pairs',
          'Mileage',
          'Power-kW',
          'Younges-driver-age',
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
          'Estimated-Vehicle-value',
          'Maximum-mass',
          'Load capacity',
          'Production year',
          'Publication datetime',
          'Registration-date',
          'Purchase-date',
          'PESEL']

def dataframe_analysis(df,xls_filename='Columns_analysis.xlsx'):
    #Delete old analysis file if exist
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
    #print(output)
    return output, catCols, numCols

if __name__ == '__main__':


    FILE_PATH = 'dataset.csv'
    DIR_DATA = '/content/drive/MyDrive/datasets/cars/'

    # load dataset
    df = pd.read_csv(FILE_PATH, delimiter=';')
    #print(df.head())
    dataframe_analysis(df)
