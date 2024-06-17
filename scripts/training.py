import pickle
import mlflow
import mlflow.sklearn
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_log_model(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    val_accuracy = accuracy_score(y_val, y_pred_val)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    val_precision = precision_score(y_val, y_pred_val, pos_label=1)
    test_precision = precision_score(y_test, y_pred_test, pos_label=1)
    val_recall = recall_score(y_val, y_pred_val, pos_label=1)
    test_recall = recall_score(y_test, y_pred_test, pos_label=1)
    val_f1 = f1_score(y_val, y_pred_val, pos_label=1)
    test_f1 = f1_score(y_test, y_pred_test, pos_label=1)

    mlflow.log_param("model", model_name)
    mlflow.log_metric("val_accuracy", val_accuracy)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("val_precision", val_precision)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("val_recall", val_recall)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("val_f1", val_f1)
    mlflow.log_metric("test_f1", test_f1)
    mlflow.sklearn.log_model(model, model_name, registered_model_name=model_name)
    print(f"{model_name} validation accuracy: {val_accuracy}, test accuracy: {test_accuracy}")

    return model

def tune_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_val = best_model.predict(X_val)
    y_pred_test = best_model.predict(X_test)

    val_accuracy = accuracy_score(y_val, y_pred_val)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    val_precision = precision_score(y_val, y_pred_val, pos_label=1)
    test_precision = precision_score(y_test, y_pred_test, pos_label=1)
    val_recall = recall_score(y_val, y_pred_val, pos_label=1)
    test_recall = recall_score(y_test, y_pred_test, pos_label=1)
    val_f1 = f1_score(y_val, y_pred_val, pos_label=1)
    test_f1 = f1_score(y_test, y_pred_test, pos_label=1)

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("val_accuracy", val_accuracy)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("val_precision", val_precision)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("val_recall", val_recall)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("val_f1", val_f1)
    mlflow.log_metric("test_f1", test_f1)
    mlflow.sklearn.log_model(best_model, "RandomForestClassifier", registered_model_name="RandomForestClassifier")

    print(f"Tuned RandomForestClassifier validation accuracy: {val_accuracy}, test accuracy: {test_accuracy}")
    return best_model

if __name__ == "__main__":
    with open('data/preprocessed_data.pkl', 'rb') as f:
        X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(f)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("training_experiment")

    with mlflow.start_run():
        # Log the preprocessed data
        mlflow.log_artifact('data/preprocessed_data.pkl', 'preprocessed_data.pkl')

        # Log the metadata
        metadata = {
            "dataset": "HR Analytics",
            "version": "1.0",
            "description": "This is the HR Analytics dataset used for the training experiment.",
            "features": ["DailyRate", "DistanceFromHome", "EnvironmentSatisfaction", "JobInvolvement", "JobSatisfaction", "MaritalStatus", "NumCompaniesWorked", "OverTime", "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany"],
            "target": "Attrition"
        }
        with open("metadata.json", "w") as f:
            json.dump(metadata, f)
        mlflow.log_artifact("metadata.json", "metadata")

        best_model = tune_random_forest(X_train, y_train, X_val, y_val, X_test, y_test)
        best_model = train_and_log_model(best_model, "RandomForestClassifier", X_train, y_train, X_val, y_val, X_test, y_test)