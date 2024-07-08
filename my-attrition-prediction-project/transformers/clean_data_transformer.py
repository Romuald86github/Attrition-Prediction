import os
import pandas as pd
from mage_ai.data_preparation.decorators import transformer

@transformer
def clean_data(data, *args, **kwargs):
    data = data.drop('EmployeeNumber', axis=1)
    
    # Save the cleaned data
    file_path = os.path.join('my-attrition-prediction-project', 'transformers', 'clean_data.csv')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data.to_csv(file_path, index=False)
    return data
