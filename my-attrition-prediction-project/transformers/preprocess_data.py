# preprocess_data.py

import os
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from scipy.stats import yeojohnson, skew
import boto3
import cloudpickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

# Ensure AWS credentials are set in the environment
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_DEFAULT_REGION')

# Set the S3 bucket and path
bucket_name = "attritionproject"
artifact_path = "attrition/mlflow/artifacts"
artifact_uri = f"s3://{bucket_name}/{artifact_path}"

# Initialize boto3 client
s3_client = boto3.client(
    's3',
    region_name=aws_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

class PreprocessingPipeline:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def transform(self, model_input):
        return self.pipeline.transform(model_input)

def remove_skewness(X: DataFrame) -> DataFrame:
    skewed_cols = [col for col in X.select_dtypes(include=['float64', 'int64']).columns if skew(X[col]) > 0.5]
    X_transformed = X.copy()
    for col in skewed_cols:
        X_transformed[col], _ = yeojohnson(X_transformed[col])
    return X_transformed

@transformer
def preprocess_data(df: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']

    # Remove skewness from columns with skewness > 0.5
    X = remove_skewness(X)

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Encode or get_dummies for categorical columns
    categorical_cols = X.select_dtypes(include='object').columns
    one_hot_encoder = OneHotEncoder(drop='first')
    column_transformer = ColumnTransformer(
        transformers=[
            ('cat', one_hot_encoder, categorical_cols)
        ],
        remainder='passthrough'
    )
    X = column_transformer.fit_transform(X)
    X = pd.DataFrame(X, columns=column_transformer.get_feature_names_out())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    preprocessed_data = (X_train, X_val, X_test, y_train, y_val, y_test)

    # Create preprocessing pipeline
    preprocessing_pipeline = Pipeline(steps=[
        ('skewness_transformer', FunctionTransformer(remove_skewness, validate=False)),
        ('column_transformer', column_transformer),
        ('scaler', scaler)
    ])

    # Save preprocessing pipeline to S3
    pipeline_bytes = cloudpickle.dumps(preprocessing_pipeline)
    s3_client.put_object(Bucket=bucket_name, Key=f"{artifact_path}/preprocessing_pipeline.pkl", Body=pipeline_bytes)

    return preprocessed_data

@test
def test_preprocess_data(data: DataFrame) -> None:
    """
    Template code for testing the output of the block.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(data)
    assert X_train is not None, 'The X_train output is undefined'
    assert X_val is not None, 'The X_val output is undefined'
    assert X_test is not None, 'The X_test output is undefined'
    assert y_train is not None, 'The y_train output is undefined'
    assert y_val is not None, 'The y_val output is undefined'
    assert y_test is not None, 'The y_test output is undefined'
    assert isinstance(X_train, DataFrame), 'The X_train output is not a Pandas DataFrame'
    assert isinstance(X_val, DataFrame), 'The X_val output is not a Pandas DataFrame'
    assert isinstance(X_test, DataFrame), 'The X_test output is not a Pandas DataFrame'