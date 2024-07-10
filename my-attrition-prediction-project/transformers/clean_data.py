import pandas as pd
from pandas import DataFrame


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

def select_number_columns(df: DataFrame) -> DataFrame:
    return df['EmployeeNumber']


def drop_employee_number_column(df: DataFrame) -> DataFrame:
    return df.drop('EmployeeNumber', axis=1)


@transformer
def clean_data(df: DataFrame, *args, **kwargs) -> DataFrame:
    return drop_employee_number_column(select_number_columns(df))


@test
def test_output(df) -> None:
    """
    Template code for testing the output of the block.
    """
    assert df is not None, 'The output is undefined'



