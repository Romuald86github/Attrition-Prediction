import os
import pickle
from mage_ai.io.file import FileIO
from mage_ai.data_preparation.decorators import data_exporter
from pandas import DataFrame

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def export_data(X_train: DataFrame, X_val: DataFrame, X_test: DataFrame, y_train: DataFrame, y_val: DataFrame, y_test: DataFrame) -> None:
    """
    Template for exporting preprocessed data to filesystem as a pickle file.
    """
    filepath = 'preprocessed_data.pkl'
    preprocessed_data = (X_train, X_val, X_test, y_train, y_val, y_test)
    FileIO().export(preprocessed_data, filepath, file_format='pickle')