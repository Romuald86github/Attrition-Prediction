import pandas as pd
from mage_ai.data_preparation.decorators import data_loader, data_exporter

@data_loader
def load_data():
    data = pd.read_csv("my-attrition-prediction-project/transformers/clean_data.csv")
    return data

@data_exporter
def select_features(data, selected_features):
    selected_data = data[selected_features]
    selected_data.to_csv("my-attrition-prediction-project/features/selected_data.csv", index=False)
    return selected_data

if __name__ == "__main__":
    data = load_data()
    target_column = "Attrition"
    selected_features = ['DailyRate', 'DistanceFromHome', 'EnvironmentSatisfaction',
                         'JobInvolvement', 'JobSatisfaction', 'MaritalStatus', 'NumCompaniesWorked',
                         'OverTime', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
                         'WorkLifeBalance', 'YearsAtCompany']
    select_features(data, selected_features)
