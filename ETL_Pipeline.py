import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
import psycopg2

#  Load environment variables from .env file
load_dotenv()

username = os.getenv("DB_USERNAME")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
database = os.getenv("DB_NAME")

# ✅ Define base directory relative to scripts folder
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
file_path = os.path.join(BASE_DIR, "raw", "coffee_shop_raw_data.csv")

# 1. Load raw CSV
df = pd.read_csv(file_path)

# 2. Clean & transform
df['transaction_datetime'] = pd.to_datetime(df['transaction_date'] + " " + df['transaction_time'])
df['transaction_qty'] = df['transaction_qty'].astype(int)
df['unit_price'] = df['unit_price'].astype(float)
df['total_amount'] = df['transaction_qty'] * df['unit_price']

# Extract time features
df['hour'] = df['transaction_datetime'].dt.hour
df['day'] = df['transaction_datetime'].dt.day
df['month'] = df['transaction_datetime'].dt.month
df['weekday'] = df['transaction_datetime'].dt.day_name()

# ✅ Create SQLAlchemy engine using env variables
engine = create_engine(f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}")

# ✅ Test psycopg2 connection
try:
    conn = psycopg2.connect(
        dbname=database,
        user=username,
        password=password,
        host=host,
        port=port
    )
    print("✅ Connection successful!")
    conn.close()
except Exception as e:
    print("❌ Connection failed:", e)

# 3. Upload to PostgreSQL
table_name = "coffee_shop_sales"
df.to_sql(table_name, engine, if_exists="replace", index=False)

print(f"ETL pipeline complete — data successfully loaded into table '{table_name}' in database '{database}'.")
