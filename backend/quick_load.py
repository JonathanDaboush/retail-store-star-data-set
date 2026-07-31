import os
import glob
import pandas as pd
from sqlalchemy import create_engine as sqlalchemy_create_engine

from dotenv import load_dotenv
from controller.producer import send_row_event
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)


load_dotenv()

MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_HOST = os.getenv("MYSQL_HOST", "mysql")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

DATABASE_URL = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
    f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
)

engine = sqlalchemy_create_engine(DATABASE_URL)

CSV_DIRECTORY = "/app/original_data"

def import_all_csvs():
    
    # Looks into /app/original_data/*.csv inside Docker
    csv_files = glob.glob(os.path.join(CSV_DIRECTORY, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {CSV_DIRECTORY}. Check your volume mapping!")
        return

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        # Drops the .csv extension and creates a safe database table name
        table_name = os.path.splitext(file_name)[0].lower().replace("-", "_").replace(" ", "_")
        
        try:
            print(f"Importing {file_name} into table '{table_name}'...")
            df = pd.read_csv(file_path)
            df.to_sql(table_name, con=engine, if_exists='replace', index=False)
            print(f"Successfully loaded {len(df)} rows.")
        except Exception as e:
            print(f"Error importing {file_name}: {e}")

if __name__ == "__main__":
    print("Starting initial data load")

    import_all_csvs()

    print("Data load complete")