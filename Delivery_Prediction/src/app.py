from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("/Users/jananisrinath/Desktop/Delivery_Prediction/models/best_model.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.get_json()
        
        # Convert input data to DataFrame
        df = pd.DataFrame([data])
        
        # Perform the same feature engineering as in training
        df = create_features(df)
        
        # Make sure the DataFrame has all the columns the model expects
        for column in model.feature_names_in_:
            if column not in df.columns:
                df[column] = 0  # or any other appropriate default value
        
        # Reorder columns to match the order the model expects
        df = df[model.feature_names_in_]
        
        # Make prediction
        prediction = model.predict(df)
        
        # Return the prediction
        return jsonify({'predicted_delivery_time': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def create_features(df):
    # This function should mirror the feature engineering in create_features.py
    # Include all the feature engineering steps here
    
    if 'DISTANCE' in df.columns:
        df['DISTANCE_BIN'] = pd.cut(df['DISTANCE'], bins=5, labels=False)

    if 'PICKUP_LOCATION' in df.columns and 'DELIVERY_LOCATION' in df.columns:
        df['LOCATION_PAIR'] = df['PICKUP_LOCATION'].astype(str) + '_' + df['DELIVERY_LOCATION'].astype(str)
        df['LOCATION_PAIR'] = pd.Categorical(df['LOCATION_PAIR']).codes

    if 'VEHICLE_TYPE' in df.columns:
        df['IS_TRUCK'] = (df['VEHICLE_TYPE'] == 'Truck').astype(int)

    if 'TRAFFIC_CONDITIONS' in df.columns:
        df['IS_HEAVY_TRAFFIC'] = (df['TRAFFIC_CONDITIONS'] == 'Heavy').astype(int)

    if 'WEATHER' in df.columns:
        df['IS_BAD_WEATHER'] = df['WEATHER'].isin(['Rainy', 'Snowy']).astype(int)

    if 'DISTANCE' in df.columns and 'VEHICLE_TYPE' in df.columns:
        df['DISTANCE_VEHICLE'] = df['DISTANCE'] * pd.Categorical(df['VEHICLE_TYPE']).codes

    if 'TRAFFIC_CONDITIONS' in df.columns and 'WEATHER' in df.columns:
        df['TRAFFIC_WEATHER'] = pd.Categorical(df['TRAFFIC_CONDITIONS']).codes * pd.Categorical(df['WEATHER']).codes

    if 'DISTANCE' in df.columns:
        df['IS_LONG_DISTANCE'] = (df['DISTANCE'] > 500).astype(int)

    return df

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)