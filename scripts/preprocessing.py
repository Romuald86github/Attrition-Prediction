import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from scipy.stats import yeojohnson

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

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    data = pd.read_csv("data/processed/selected_data.csv")
    target_column = "Attrition"
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(data, target_column)

    X_train_df = pd.DataFrame(X_train)
    X_val_df = pd.DataFrame(X_val)
    X_test_df = pd.DataFrame(X_test)
    y_train_df = pd.DataFrame(y_train)
    y_val_df = pd.DataFrame(y_val)
    y_test_df = pd.DataFrame(y_test)

    X_train_df.to_csv("data/processed/X_train.csv", index=False)
    X_val_df.to_csv("data/processed/X_val.csv", index=False)
    X_test_df.to_csv("data/processed/X_test.csv", index=False)
    y_train_df.to_csv("data/processed/y_train.csv", index=False)
    y_val_df.to_csv("data/processed/y_val.csv", index=False)
    y_test_df.to_csv("data/processed/y_test.csv", index=False)