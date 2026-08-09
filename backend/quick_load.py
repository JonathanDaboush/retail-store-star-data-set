"""Initialise the operational star schema without altering the source event bank."""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
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
            pd.read_csv(DATA_DIR / f"{table}.csv").to_sql(table, connection, if_exists="replace", index=False)

        pd.read_csv(DATA_DIR / "fact_sales_normalized.csv", nrows=1).head(0).to_sql(
            "fact_sales_normalized",
            connection,
            if_exists="replace",
            index=False,
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
