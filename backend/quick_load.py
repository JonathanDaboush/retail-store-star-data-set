"""Initialise the operational star schema without altering the source event bank."""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import BigInteger, Date, DateTime, Integer, Numeric, String, create_engine, text
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()

engine = create_engine(
    f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}"
    f"@{os.getenv('MYSQL_HOST', 'mysql')}:{os.getenv('MYSQL_PORT', '3306')}/{os.getenv('MYSQL_DATABASE')}"
)

DATA_DIR = Path(os.getenv("SOURCE_DATA_DIR", "/app/original_data"))
DIMENSIONS = [
    "dim_campaigns",
    "dim_customers",
    "dim_dates",
    "dim_products",
    "dim_salespersons",
    "dim_stores",
]

TABLE_DTYPES = {
    "dim_campaigns": {
        "campaign_sk": Integer(),
        "campaign_id": String(32),
        "campaign_name": String(255),
        "start_date_sk": Integer(),
        "end_date_sk": Integer(),
        "campaign_budget": Numeric(14, 2),
    },
    "dim_customers": {
        "customer_sk": Integer(),
        "customer_id": String(32),
        "first_name": String(100),
        "last_name": String(100),
        "email": String(255),
        "residential_location": String(120),
        "customer_segment": String(80),
    },
    "dim_dates": {
        "full_date": Date(),
        "date_sk": Integer(),
        "year": Integer(),
        "month": Integer(),
        "day": Integer(),
        "weekday": Integer(),
        "quarter": Integer(),
    },
    "dim_products": {
        "product_sk": Integer(),
        "product_id": String(32),
        "product_name": String(255),
        "category": String(120),
        "brand": String(120),
        "origin_location": String(120),
    },
    "dim_salespersons": {
        "salesperson_sk": Integer(),
        "salesperson_id": String(32),
        "salesperson_name": String(150),
        "salesperson_role": String(80),
    },
    "dim_stores": {
        "store_sk": Integer(),
        "store_id": String(32),
        "store_name": String(150),
        "store_type": String(120),
        "store_location": String(120),
        "store_manager_sk": Integer(),
    },
    "fact_sales_normalized": {
        "sales_sk": BigInteger(),
        "sales_id": String(64),
        "customer_sk": Integer(),
        "product_sk": Integer(),
        "store_sk": Integer(),
        "salesperson_sk": Integer(),
        "campaign_sk": Integer(),
        "sales_date": DateTime(),
        "total_amount": Numeric(14, 2),
    },
}


def _safe_execute(connection, sql: str) -> None:
    try:
        connection.execute(text(sql))
    except SQLAlchemyError as exc:
        message = str(exc).lower()
        duplicate_markers = [
            "duplicate",
            "already exists",
            "multiple primary key defined",
        ]
        if any(marker in message for marker in duplicate_markers):
            return
        raise


def import_all_csvs() -> None:
    """Load dimensions and create an empty, constrained, indexed fact table."""
    with engine.begin() as connection:
        for table in DIMENSIONS:
            pd.read_csv(DATA_DIR / f"{table}.csv").to_sql(
                table,
                connection,
                if_exists="replace",
                index=False,
                dtype=TABLE_DTYPES[table],
            )

        pd.read_csv(DATA_DIR / "fact_sales_normalized.csv", nrows=1).head(0).to_sql(
            "fact_sales_normalized",
            connection,
            if_exists="replace",
            index=False,
            dtype=TABLE_DTYPES["fact_sales_normalized"],
        )

        _safe_execute(connection, "ALTER TABLE dim_campaigns ADD PRIMARY KEY (campaign_sk)")
        _safe_execute(connection, "ALTER TABLE dim_customers ADD PRIMARY KEY (customer_sk)")
        _safe_execute(connection, "ALTER TABLE dim_dates ADD PRIMARY KEY (date_sk)")
        _safe_execute(connection, "ALTER TABLE dim_products ADD PRIMARY KEY (product_sk)")
        _safe_execute(connection, "ALTER TABLE dim_salespersons ADD PRIMARY KEY (salesperson_sk)")
        _safe_execute(connection, "ALTER TABLE dim_stores ADD PRIMARY KEY (store_sk)")

        _safe_execute(connection, "ALTER TABLE fact_sales_normalized ADD PRIMARY KEY (sales_sk)")
        _safe_execute(connection, "CREATE UNIQUE INDEX ux_fact_sales_id ON fact_sales_normalized (sales_id)")
        _safe_execute(connection, "CREATE INDEX ix_fact_sales_date ON fact_sales_normalized (sales_date)")
        _safe_execute(connection, "CREATE INDEX ix_fact_sales_customer ON fact_sales_normalized (customer_sk)")
        _safe_execute(connection, "CREATE INDEX ix_fact_sales_product ON fact_sales_normalized (product_sk)")
        _safe_execute(connection, "CREATE INDEX ix_fact_sales_store ON fact_sales_normalized (store_sk)")

        _safe_execute(
            connection,
            """
            ALTER TABLE fact_sales_normalized
            ADD CONSTRAINT fk_fact_customer FOREIGN KEY (customer_sk) REFERENCES dim_customers (customer_sk)
            """,
        )
        _safe_execute(
            connection,
            """
            ALTER TABLE fact_sales_normalized
            ADD CONSTRAINT fk_fact_product FOREIGN KEY (product_sk) REFERENCES dim_products (product_sk)
            """,
        )
        _safe_execute(
            connection,
            """
            ALTER TABLE fact_sales_normalized
            ADD CONSTRAINT fk_fact_store FOREIGN KEY (store_sk) REFERENCES dim_stores (store_sk)
            """,
        )
        _safe_execute(
            connection,
            """
            ALTER TABLE fact_sales_normalized
            ADD CONSTRAINT fk_fact_salesperson FOREIGN KEY (salesperson_sk) REFERENCES dim_salespersons (salesperson_sk)
            """,
        )
        _safe_execute(
            connection,
            """
            ALTER TABLE fact_sales_normalized
            ADD CONSTRAINT fk_fact_campaign FOREIGN KEY (campaign_sk) REFERENCES dim_campaigns (campaign_sk)
            """,
        )


if __name__ == "__main__":
    import_all_csvs()
    print("Dimensions loaded; fact event bank is ready for replay.")
