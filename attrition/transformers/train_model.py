import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from mage_ai.data_preparation.decorators import transformer

@transformer
def train_model(X_train, y_train, X_val, y_val, X_test, y_test):
    best_model = None
    best_score = 0

    models = [
        RandomForestClassifier(random_state=42)
    ]

    for model in models:
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
        val_roc_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        test_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        if val_roc_auc > best_score:
            best_model = model
            best_score = val_roc_auc

    # Perform hyperparameter tuning on the best model
    if isinstance(best_model, RandomForestClassifier):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }

        grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='roc_auc')
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
        val_roc_auc = roc_auc_score(y_val, best_model.predict_proba(X_val)[:, 1])
        test_roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

        os.makedirs('my-attrition-prediction-project/models', exist_ok=True)
        model_path = 'my-attrition-prediction-project/models/best_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)

        print(f"Tuned RandomForestClassifier validation accuracy: {val_accuracy}, test accuracy: {test_accuracy}")

    return best_model