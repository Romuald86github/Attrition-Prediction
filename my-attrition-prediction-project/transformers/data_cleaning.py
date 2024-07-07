import os
import pandas as pd
import numpy as np
import requests
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
    """Clean the raw data by dropping the 'EmployeeNumber' column."""
    # Drop the 'EmployeeNumber' column
    data = data.drop('EmployeeNumber', axis=1)

    # Save the cleaned data
    file_path = os.path.join('my-attrition-prediction-project', 'transformers', 'clean_data.csv')
    data.to_csv(file_path)

    return data

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/Romuald86github/Internship/main/employee_attrition.csv"
    raw_data = load_data(url)
    cleaned_data = clean_data(raw_data)