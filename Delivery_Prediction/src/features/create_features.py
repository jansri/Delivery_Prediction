import pandas as pd
import numpy as np

def create_features(df):
    # Create distance bins
    if 'DISTANCE' in df.columns:
        df['DISTANCE_BIN'] = pd.cut(df['DISTANCE'], bins=5, labels=False)

    # Create location pair feature
    if 'PICKUP_LOCATION' in df.columns and 'DELIVERY_LOCATION' in df.columns:
        df['LOCATION_PAIR'] = df['PICKUP_LOCATION'].astype(str) + '_' + df['DELIVERY_LOCATION'].astype(str)
        df['LOCATION_PAIR'] = pd.Categorical(df['LOCATION_PAIR']).codes

    # Create vehicle type feature
    if 'VEHICLE_TYPE' in df.columns:
        df['IS_TRUCK'] = (df['VEHICLE_TYPE'] == 'Truck').astype(int)

    # Create traffic conditions feature
    if 'TRAFFIC_CONDITIONS' in df.columns:
        df['IS_HEAVY_TRAFFIC'] = (df['TRAFFIC_CONDITIONS'] == 'Heavy').astype(int)

    # Create weather feature
    if 'WEATHER' in df.columns:
        df['IS_BAD_WEATHER'] = df['WEATHER'].isin(['Rainy', 'Snowy']).astype(int)

    # Create interaction features
    if 'DISTANCE' in df.columns and 'VEHICLE_TYPE' in df.columns:
        df['DISTANCE_VEHICLE'] = df['DISTANCE'] * pd.Categorical(df['VEHICLE_TYPE']).codes

    if 'TRAFFIC_CONDITIONS' in df.columns and 'WEATHER' in df.columns:
        df['TRAFFIC_WEATHER'] = pd.Categorical(df['TRAFFIC_CONDITIONS']).codes * pd.Categorical(df['WEATHER']).codes

    # Create boolean features
    if 'DISTANCE' in df.columns:
        df['IS_LONG_DISTANCE'] = (df['DISTANCE'] > 500).astype(int)

    return df

if __name__ == "__main__":
    # Load the preprocessed data
    df = pd.read_csv('/Users/jananisrinath/Desktop/Delivery_Prediction/data/preprocessed_data.csv')
    
    # Print column names for debugging
    print("Columns before feature engineering:", df.columns)

    # Apply feature engineering
    df_featured = create_features(df)
    
    # Print new column names for debugging
    print("Columns after feature engineering:", df_featured.columns)

    # Save the featured data
    df_featured.to_csv('/Users/jananisrinath/Desktop/Delivery_Prediction/data/featured_data.csv', index=False)