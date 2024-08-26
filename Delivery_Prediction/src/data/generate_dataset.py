import pandas as pd
import numpy as np
from datetime import timedelta, datetime

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 100000

# Generate data
data = {
    'pickup_location': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_samples),
    'delivery_location': np.random.choice(['Boston', 'San Francisco', 'Miami', 'Seattle', 'Denver'], n_samples),
    'distance': np.random.uniform(50, 1000, n_samples).round(2),  # in miles
    'vehicle_type': np.random.choice(['Van', 'Truck', 'Motorcycle'], n_samples),
    'traffic_conditions': np.random.choice(['Light', 'Moderate', 'Heavy'], n_samples),
    'weather': np.random.choice(['Clear', 'Cloudy', 'Rainy', 'Snowy'], n_samples),
    'previous_deliveries': np.random.randint(0, 50, n_samples)
}

# Create DataFrame
df = pd.DataFrame(data)

# Function to calculate synthetic delivery time
def calculate_delivery_time(row):
    base_time = row['distance'] / 50  # Base time: 50 miles per hour
    
    # Adjust for vehicle type
    vehicle_factor = {'Van': 1, 'Truck': 1.2, 'Motorcycle': 0.8}
    base_time *= vehicle_factor[row['vehicle_type']]
    
    # Adjust for traffic
    traffic_factor = {'Light': 1, 'Moderate': 1.3, 'Heavy': 1.6}
    base_time *= traffic_factor[row['traffic_conditions']]
    
    # Adjust for weather
    weather_factor = {'Clear': 1, 'Cloudy': 1.1, 'Rainy': 1.3, 'Snowy': 1.5}
    base_time *= weather_factor[row['weather']]
    
    # Slight random variation
    base_time *= np.random.uniform(0.9, 1.1)
    
    return round(base_time, 2)

# Calculate delivery time
df['delivery_time'] = df.apply(calculate_delivery_time, axis=1)

# Add some noise to previous_deliveries vs delivery_time correlation
df['delivery_time'] += np.random.normal(0, 0.5, n_samples)

# Ensure delivery_time is positive
df['delivery_time'] = df['delivery_time'].clip(lower=0)

# Display first few rows and data info
print(df.head())
print("\nDataset Info:")
print(df.info())

# Save to CSV
df.to_csv('logistics_dataset.csv', index=False)
print("\nDataset saved as 'logistics_dataset.csv'")