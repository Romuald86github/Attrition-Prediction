import os
import pandas as pd
import numpy as np
import requests
from scipy.stats import zscore
from io import StringIO
from mage_ai.data_preparation.decorators import data_loader, data_exporter

@data_loader
def load_data(url):
    """Load the raw data from the provided URL and save it to the 'data_loaders/raw_data.csv' file."""
    response = requests.get(url)

    # Save the raw data to 'data_loaders/raw_data.csv'
    file_path = os.path.join('my-attrition-prediction-project', 'data_loaders', 'raw_data.csv')
    with open(file_path, 'wb') as f:
        f.write(response.content)

    # Read the CSV data
    data = pd.read_csv(file_path)

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

    # (4) Remove rows with numerical features having absolute z-score > 3
    numerical_cols = [col for col in data.columns if data[col].dtype in ['float64', 'int64']]
    feat = data[numerical_cols]

    # Calculate z-scores using scipy.stats.zscore
    z_scores = zscore(feat)

    # Convert the result to a DataFrame
    z_score_df = pd.DataFrame(z_scores, columns=feat.columns, index=feat.index)

    # Filter rows where all z-scores are less than 3
    data = data[(z_score_df < 3).all(axis=1)]

    # (5) Save the cleaned data
    file_path = os.path.join('my-attrition-prediction-project', 'transformers', 'clean_data.csv')
    data.to_csv(file_path)

    return data

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/Romuald86github/Internship/main/employee_attrition.csv"
    raw_data = load_data(url)
    cleaned_data = clean_data(raw_data)
