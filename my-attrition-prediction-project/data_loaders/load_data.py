import os
import pandas as pd
import requests
from pandas import DataFrame
from mage_ai.data_preparation.decorators import data_loader, test

@data_loader
def load_data(**kwargs) -> DataFrame:
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

@test
def test_output(df) -> None:
    assert df is not None, 'The output is undefined'
    assert isinstance(df, pd.DataFrame), 'The output is not a Pandas DataFrame'
    assert len(df) > 0, 'The DataFrame is empty'