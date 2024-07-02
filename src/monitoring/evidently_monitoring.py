import os
import mlflow
import mlflow.sklearn
import pandas as pd
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, ClassificationPerformanceTab
from evidently.pipeline.column_mapping import ColumnMapping

model_name = "RandomForestClassifier"
model_version = "1"
model_path = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_path)

# Load test data
X_test = pd.read_pickle('data/processed/X_test.pkl')
y_test = pd.read_pickle('data/processed/y_test.pkl')

# Set up column mapping
column_mapping = ColumnMapping(
    target="Attrition",
    numerical_features=list(X_test.columns),
    categorical_features=[],
    datetime_features=[]
)

# Create Evidently dashboard
dashboard = Dashboard(tabs=[DataDriftTab(), ClassificationPerformanceTab()])
dashboard.calculate(X_test, y_test, column_mapping=column_mapping)
dashboard.save("evidently_report.html")