import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
from scipy.stats import yeojohnson, skew
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5001")  # Update with the correct URI if running on a remote machine

class PreprocessingPipeline(mlflow.pyfunc.PythonModel):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, context, model_input):
        return self.pipeline.transform(model_input)

def preprocess_data(data, target_column='Attrition'):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Remove skewness from columns with skewness > 0.5
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    skewed_cols = [col for col in numerical_cols if skew(X[col]) > 0.5]
    
    def remove_skewness(X):
        X_transformed = X.copy()
        for col in skewed_cols:
            X_transformed[col], _ = yeojohnson(X_transformed[col])
        return X_transformed

    skewness_transformer = FunctionTransformer(remove_skewness, validate=False)
    X = skewness_transformer.transform(X)

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
    X = pd.DataFrame(X, columns=column_transformer.get_feature_names_out())  # Ensure output is a DataFrame

    # Balance target classes
    ros = RandomOverSampler(random_state=42)
    X, y = ros.fit_resample(X, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Save the preprocessed data
    os.makedirs('my-attrition-prediction-project/features', exist_ok=True)
    pd.to_pickle(X_train, 'my-attrition-prediction-project/features/X_train.pkl')
    pd.to_pickle(X_val, 'my-attrition-prediction-project/features/X_val.pkl')
    pd.to_pickle(X_test, 'my-attrition-prediction-project/features/X_test.pkl')
    pd.to_pickle(y_train, 'my-attrition-prediction-project/features/y_train.pkl')
    pd.to_pickle(y_val, 'my-attrition-prediction-project/features/y_val.pkl')
    pd.to_pickle(y_test, 'my-attrition-prediction-project/features/y_test.pkl')

    preprocessed_data = (X_train, X_val, X_test, y_train, y_val, y_test)
    pd.to_pickle(preprocessed_data, 'my-attrition-prediction-project/features/preprocessed_data.pkl')
    
    # Log preprocessing steps to MLflow
    preprocessing_pipeline = Pipeline(steps=[
        ('skewness_transformer', skewness_transformer),
        ('column_transformer', column_transformer),
        ('scaler', scaler)
    ])

    # Create the custom PythonModel
    custom_model = PreprocessingPipeline(preprocessing_pipeline)

    # Infer the signature of the pipeline
    signature = infer_signature(X_train)

    # Log the pipeline model to MLflow
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            artifact_path="preprocessing_pipeline",
            python_model=custom_model,
            signature=signature,
            registered_model_name="AttritionPreprocessingPipeline"
        )

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    # Load the data from the CSV file
    data = pd.read_csv("my-attrition-prediction-project/features/selected_data.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(data)
