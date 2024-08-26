import snowflake.connector
import pandas as pd

def extract_data_from_snowflake():
    conn = snowflake.connector.connect(
        user='J********'
        password='***********',
        account='************',
        warehouse='COMPUTE_WH',
        database='LOGISTICS_DATA',
        schema='PUBLIC',
        login_timeout=60
    )

    query = """
    SELECT pickup_location, delivery_location, distance, vehicle_type, traffic_conditions, weather, delivery_time
    FROM LOGISTICS_DATA.PUBLIC.LOGISTICS_DELIVERY_DATA
    """
    df = pd.read_sql(query, conn)
    print(df.head())
    print(df.info())
    print(df.describe())
    conn.close()

    return df

if __name__ == "__main__":
    df = extract_data_from_snowflake()
    df.to_csv('/Users/jananisrinath/Desktop/Delivery_Prediction/data/raw_data.csv', index=False)