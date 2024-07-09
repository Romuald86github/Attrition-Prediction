import os
import pandas as pd
import requests
from mage_ai.data_preparation.decorators import data_loader

@data_loader
def load_data(*args, **kwargs):
    url = kwargs.get('url', 'https://raw.githubusercontent.com/Romuald86github/Internship/main/employee_attrition.csv')
    response = requests.get(url)

    # Save the raw data to 'data_loaders/raw_data.csv'
    file_path = os.path.join('my-attrition-prediction-project', 'data_loaders', 'raw_data.csv')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        f.write(response.content)

    # Read the CSV data
    data = pd.read_csv(file_path)
    return data

    # test 
