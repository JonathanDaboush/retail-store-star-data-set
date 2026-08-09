"""Initialise the operational star schema without altering the source event bank."""
import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(
    f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}"
    f"@{os.getenv('MYSQL_HOST', 'mysql')}:{os.getenv('MYSQL_PORT', '3306')}/{os.getenv('MYSQL_DATABASE')}"
)
DATA_DIR = Path(os.getenv("SOURCE_DATA_DIR", "/app/original_data"))
DIMENSIONS = ["dim_campaigns", "dim_customers", "dim_dates", "dim_products", "dim_salespersons", "dim_stores"]

def import_all_csvs():
    """Load dimensions once and create an empty, indexed transaction table.

    Fact rows deliberately remain in the immutable CSV event bank until replayed.
    """
    with engine.begin() as connection:
        for table in DIMENSIONS:
            pd.read_csv(DATA_DIR / f"{table}.csv").to_sql(table, connection, if_exists="replace", index=False)
        # Use the source's columns/types, but never copy its transactions during bootstrap.
        # Read one row solely to retain numeric SQL types, then write no fact data.
        pd.read_csv(DATA_DIR / "fact_sales_normalized.csv", nrows=1).head(0).to_sql(
            "fact_sales_normalized", connection, if_exists="replace", index=False
        )
        connection.execute(text("ALTER TABLE fact_sales_normalized ADD PRIMARY KEY (sales_sk)"))
        connection.execute(text("CREATE INDEX ix_fact_sales_date ON fact_sales_normalized (sales_date)"))
        connection.execute(text("CREATE INDEX ix_fact_sales_customer ON fact_sales_normalized (customer_sk)"))

if __name__ == "__main__":
    import_all_csvs()
    print("Dimensions loaded; fact event bank is ready for replay.")
