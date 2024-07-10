import pandas as pd
from pandas import DataFrame
from mage_ai.data_preparation.decorators import transformer, test

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@transformer
def clean_data(data: DataFrame, *args, **kwargs) -> DataFrame:
    """
    Clean the input data by dropping the 'EmployeeNumber' column.
    """
    data = data.drop('EmployeeNumber', axis=1)
    return data

@test
def test_clean_data(data: DataFrame) -> None:
    """
    Test the output of the clean_data transformer.
    """
    assert data is not None, 'The output is undefined'
    assert isinstance(data, pd.DataFrame), 'The output is not a Pandas DataFrame'
    assert 'EmployeeNumber' not in data.columns, 'The EmployeeNumber column was not dropped'
