import pandas as pd
import DataFrame



if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

def select_number_columns(df: DataFrame) -> DataFrame:
    return df


def features (df: DataFrame) -> DataFrame:
    return df[['DailyRate', 'DistanceFromHome', 'EnvironmentSatisfaction',
                         'JobInvolvement', 'JobSatisfaction', 'MaritalStatus', 'NumCompaniesWorked',
                         'OverTime', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
                         'WorkLifeBalance', 'YearsAtCompany', 'Attrition']]

@transformer
def select_features(df: DataFrame, *args, **kwargs) -> DataFrame:
    return features(select_number_columns(df))



@test
def test_output(df: DataFrame) -> None:
    """
    Template code for testing the output of the block.
    """
    assert df is not None, 'The output is undefined'
    assert isinstance(df, DataFrame), 'The output is not a Pandas DataFrame'
    assert all(col in df.columns for col in ['DailyRate', 'DistanceFromHome', 'EnvironmentSatisfaction',
                                            'JobInvolvement', 'JobSatisfaction', 'MaritalStatus', 'NumCompaniesWorked',
                                            'OverTime', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
                                            'WorkLifeBalance', 'YearsAtCompany', 'Attrition']), 'Not all expected columns are present'
    
