from mage_ai.data_preparation.decorators import data_loader, data_exporter, transformer, pipeline
from mage_ai.pipelines.base import Pipeline
from mage_ai.shared.constants import DataType

@pipeline
def attrition_prediction_pipeline_definition():
    return AttritionPredictionPipeline()

class AttritionPredictionPipeline(Pipeline):
    @data_loader
    def load_raw_data(self):
        from data_loaders.data_cleaning import load_data
        return load_data("https://raw.githubusercontent.com/Romuald86github/Internship/main/employee_attrition.csv")

    @transformer
    def select_features(self, data):
        from transformers.feature_selection import select_features
        target_column = ["Attrition"]
        selected_features = ['DailyRate', 'DistanceFromHome', 'EnvironmentSatisfaction',
                             'JobInvolvement', 'JobSatisfaction', 'MaritalStatus', 'NumCompaniesWorked',
                             'OverTime', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
                             'WorkLifeBalance', 'YearsAtCompany']
        return select_features(data, selected_features, target_column)

    @transformer
    def preprocess_data(self, data, target_column):
        from transformers.preprocessing import preprocess_data
        return preprocess_data(data, target_column)

    @data_exporter
    def train_model(self, X_train, y_train, X_val, y_val, X_test, y_test):
        from models.training import train_and_evaluate_models
        return train_and_evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test)

    def run(self):
        raw_data = self.load_raw_data()
        selected_data = self.select_features(raw_data)
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_data(selected_data, "Attrition")
        best_model = self.train_model(X_train, y_train, X_val, y_val, X_test, y_test)

if __name__ == "__main__":
    pipeline = AttritionPredictionPipeline()
    pipeline.run()