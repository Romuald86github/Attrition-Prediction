import os
import pickle
from mage_ai.io.file import FileIO
from mage_ai.data_preparation.decorators import data_exporter
from pandas import DataFrame

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def export_data(df: DataFrame, **kwargs) -> None:
    """
    Template for exporting preprocessed data to filesystem as a pickle file.
    """
    filepath = 'preprocessed_data.pkl'
    pFileIO().export(df, filepath)