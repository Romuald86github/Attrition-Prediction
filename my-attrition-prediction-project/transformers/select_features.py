import pandas as pd
from mage_ai.data_preparation.decorators import transformer

@transformer
def select_features(data, *args, **kwargs):
    selected_features = kwargs.get('selected_features', ['DailyRate', 'DistanceFromHome', 'EnvironmentSatisfaction',
                         'JobInvolvement', 'JobSatisfaction', 'MaritalStatus', 'NumCompaniesWorked',
                         'OverTime', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
                         'WorkLifeBalance', 'YearsAtCompany'])
    target_column = kwargs.get('target_column', 'Attrition')
    
    selected_data = data[selected_features + [target_column]]
    selected_data.to_csv("my-attrition-prediction-project/features/selected_data.csv", index=False)
    return selected_data
