import pandas as pd
import numpy as np
import requests
import os

def load_data(url):
    response = requests.get(url)
    data = pd.read_csv(pd.compat.StringIO(response.text))
    return data

def clean_data(data):
    # (1) Convert 'MonthlyIncome' and 'MonthlyRate' columns to float
    data[['MonthlyIncome', 'MonthlyRate']] = data[['MonthlyIncome', 'MonthlyRate']].astype(float)

    # (2) Convert specified columns to object data type
    columns_to_convert = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'JobSatisfaction',
                           'NumCompaniesWorked', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel',
                           'TrainingTimesLastYear', 'WorkLifeBalance']
    data[columns_to_convert] = data[columns_to_convert].astype(object)

    # (3) Set 'EmployeeNumber' as the index
    data.set_index('EmployeeNumber', inplace=True)

    # (4) Remove numerical features with absolute z-score > 3
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    z_scores = np.abs(data[numerical_cols].apply(lambda x: (x - x.mean()) / x.std(), axis=0))
    outlier_cols = z_scores[z_scores > 3].any().index
    data = data.drop(outlier_cols, axis=1)

    return data

if __name__ == "__main__":
    url = "https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset"
    raw_data = load_data(url)

    # Create the 'data/raw' directory if it doesn't exist
    raw_dir = os.path.join('..', 'data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)

    # Save the raw data to 'data/raw/raw_data.csv'
    raw_data_path = os.path.join(raw_dir, 'raw_data.csv')
    raw_data.to_csv(raw_data_path, index=False)

    clean_data = clean_data(raw_data)

    # Create the 'data/processed' directory if it doesn't exist
    processed_dir = os.path.join('..', 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    # Save the cleaned data to 'data/processed/clean_data.csv'
    clean_data_path = os.path.join(processed_dir, 'clean_data.csv')
    clean_data.to_csv(clean_data_path, index=False)