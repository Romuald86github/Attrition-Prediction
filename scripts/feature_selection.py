import pandas as pd

def select_features(data, selected_features):
    return data[selected_features]

if __name__ == "__main__":
    data = pd.read_csv("data/processed/clean_data.csv")
    target_column = "Attrition"
    selected_features = ['DailyRate', 'DistanceFromHome', 'EnvironmentSatisfaction',
                         'JobInvolvement', 'JobSatisfaction', 'MaritalStatus', 'NumCompaniesWorked',
                         'OverTime', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
                         'WorkLifeBalance', 'YearsAtCompany']
    selected_data = select_features(data, selected_features)
    selected_data.to_csv("data/processed/selected_data.csv", index=False)