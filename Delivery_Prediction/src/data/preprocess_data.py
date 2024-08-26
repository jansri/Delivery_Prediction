import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()

    # Print column names and data types for debugging
    print("Columns in the DataFrame:")
    print(df.dtypes)

    # Handle missing values for numerical columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())

    return df

def preprocess_data(df):
    le = LabelEncoder()
    
    # Identify categorical and numerical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

    # Convert categorical variables to numerical
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])

    # Scale numerical features
    if len(numerical_columns) > 0:
        scaler = StandardScaler()
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    return df

if __name__ == "__main__":
    # Load the data
    df = pd.read_csv('/Users/jananisrinath/Desktop/Delivery_Prediction/data/raw_data.csv')
    
    # Print column names for debugging
    print("Columns in the DataFrame:", df.columns)

    df_cleaned = clean_data(df)
    df_preprocessed = preprocess_data(df_cleaned)
    
    # Save the preprocessed data
    df_preprocessed.to_csv('/Users/jananisrinath/Desktop/Delivery_Prediction/data/preprocessed_data.csv', index=False)