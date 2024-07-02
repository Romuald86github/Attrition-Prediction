import os
import pandas as pd
import numpy as np
import requests
from mage_ai import data_loader, data_exporter

@data_loader
def load_data(url):
    """Load the raw data from the provided URL and save it to the 'data/raw/' directory."""
    response = requests.get(url)
    data = pd.read_csv(pd.compat.StringIO(response.text))

    # Create the 'data/raw/' directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)

    # Save the raw data to 'data/raw/raw_data.csv'
    data.to_csv('data/raw/raw_data.csv', index=False)

    return data

@data_exporter
def clean_data(data):
    """Clean the raw data."""
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

    # (5) Save the cleaned data
    os.makedirs('data/processed', exist_ok=True)
    data.to_csv('data/processed/clean_data.csv', index=False)

    return data

if __name__ == "__main__":
    url = "https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset"
    raw_data = load_data(url)
    clean_data(raw_data)