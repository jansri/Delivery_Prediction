import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import joblib

def train_model(X, y):
    # Convert integer columns to float64 if they may contain missing values
    X = X.astype({col: 'float64' for col in X.select_dtypes(include='int').columns})
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
    }

    best_model = None
    best_score = float('-inf')

    mlflow.set_experiment("package_delivery_prediction")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            mlflow.log_param("model", name)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            
            # Provide an input example for model signature
            input_example = X_train[:1]
            mlflow.sklearn.log_model(model, "model", input_example=input_example)

            if r2 > best_score:
                best_model = model
                best_score = r2

    return best_model

if __name__ == "__main__":
    # Load the featured data
    df = pd.read_csv('/Users/jananisrinath/Desktop/Delivery_Prediction/data/featured_data.csv')
    
    # Assume the target variable is named 'DELIVERY_TIME'
    target_column = 'DELIVERY_TIME'
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset")
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    print("Features used for training:", X.columns)
    
    best_model = train_model(X, y)
    
    # Save the best model
    joblib.dump(best_model, "/Users/jananisrinath/Desktop/Delivery_Prediction/models/best_model.joblib")
    print("Best model saved as 'best_model.joblib'")
