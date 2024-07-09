import os
import pickle
from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def export_data():
    # Load the preprocessed data from the file
    with open('my-attrition-prediction-project/features/preprocessed_data.pkl', 'rb') as f:
        X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(f)

    # Return the preprocessed data
    return X_train, X_val, X_test, y_train, y_val, y_test