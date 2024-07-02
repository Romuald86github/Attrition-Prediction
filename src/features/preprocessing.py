import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from scipy.stats import yeojohnson
from mage_ai import data_loader, data_exporter

@data_loader
def load_data():
    data = pd.read_csv("data/processed/selected_data.csv")
    target_column = "Attrition"
    return data, target_column

@data_exporter
def preprocess_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Remove skewness
    columns_to_transform = ['DistanceFromHome', 'TotalWorkingYears', 'YearsAtCompany']
    for col in columns_to_transform:
        X[col], _ = yeojohnson(X[col])

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Encode or get_dummies for categorical columns
    categorical_cols = X.select_dtypes(include='object').columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

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
    pd.to_pickle((X_train, X_val, X_test, y_train, y_val, y_test), 'data/processed/preprocessed_data.pkl')

    return X_train, X_val, X_test, y_train, y_val, y_test
    
