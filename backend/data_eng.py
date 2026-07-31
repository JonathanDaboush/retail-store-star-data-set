import numpy as np
import pandas as pd
import os
import json

# ============================================================
# OPTIONAL: DATA ENGINEERING / EVENT PIPELINE
# ============================================================

def create_historical_event_bank(df, output_folder="event_bank", batch_size=5000):

    os.makedirs(output_folder, exist_ok=True)

    df = df.copy()
    batch_files = []

    for start in range(0, len(df), batch_size):

        batch_number = (start // batch_size) + 1
        batch = df.iloc[start:start + batch_size]
        file_path = os.path.join(output_folder, f"batch_{batch_number}.json")

        batch.to_json(file_path, orient="records", date_format="iso")
        batch_files.append(file_path)

    print(f"Created {len(batch_files)} event batches in '{output_folder}'")

    return batch_files


def load_event_batch(file_path):

    with open(file_path, "r") as file:
        events = json.load(file)

    return pd.DataFrame(events)


def validate_event(event, required_columns):

    missing = [col for col in required_columns if col not in event]

    if missing:
        return False, missing

    return True, None


def initialize_metrics():

    return {
        "revenue": 0,
        "transactions": 0,
        "products": {},
        "customers": {},
        "daily_sales": {},
        "monthly_sales": {}
    }


def update_incremental_metrics(event, metrics):

    amount = float(event["total_amount"])

    metrics["revenue"] += amount
    metrics["transactions"] += 1

    product = event.get("product_sk")
    if product:
        metrics["products"].setdefault(product, 0)
        metrics["products"][product] += 1

    customer = event.get("customer_sk")
    if customer:
        metrics["customers"].setdefault(customer, 0)
        metrics["customers"][customer] += amount

    if "sales_date" in event:

        date = pd.to_datetime(event["sales_date"])
        day = str(date.date())
        month = str(date.to_period("M"))

        metrics["daily_sales"].setdefault(day, 0)
        metrics["monthly_sales"].setdefault(month, 0)

        metrics["daily_sales"][day] += amount
        metrics["monthly_sales"][month] += amount

    return metrics


def process_event_batch(batch_df, metrics, required_columns=None):

    required_columns = required_columns or ["sales_id", "customer_sk", "product_sk", "total_amount"]

    processed = 0
    failed = 0

    for _, row in batch_df.iterrows():

        event = row.to_dict()
        valid, error = validate_event(event, required_columns)

        if valid:
            metrics = update_incremental_metrics(event, metrics)
            processed += 1
        else:
            failed += 1

    print(f"  batch processed={processed} failed={failed}")

    return metrics


def generate_metric_summary(metrics):

    top_products = sorted(metrics["products"].items(), key=lambda x: x[1], reverse=True)[:10]
    top_customers = sorted(metrics["customers"].items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "total_revenue": round(metrics["revenue"], 2),
        "transactions": metrics["transactions"],
        "top_products": top_products,
        "top_customers": top_customers,
        "daily_sales": metrics["daily_sales"],
        "monthly_sales": metrics["monthly_sales"]
    }


def process_event_bank(batch_files):

    metrics = initialize_metrics()

    for batch_file in batch_files:
        print(f"Processing {batch_file}")
        batch = load_event_batch(batch_file)
        metrics = process_event_batch(batch, metrics)

    return generate_metric_summary(metrics)


def display_event_summary(summary):

    print("\n" + "=" * 80)
    print("EVENT PIPELINE SUMMARY")
    print("=" * 80)

    print(f"Total revenue:  {summary['total_revenue']:,.2f}")
    print(f"Transactions:   {summary['transactions']:,}")

    print("\nTop 10 products by transaction count:")
    for product_sk, count in summary["top_products"]:
        print(f"  product_sk={product_sk:<10} transactions={count}")

    print("\nTop 10 customers by spend:")
    for customer_sk, spend in summary["top_customers"]:
        print(f"  customer_sk={customer_sk:<10} spend={spend:,.2f}")

    months = sorted(summary["monthly_sales"].items())
    print(f"\nMonthly revenue ({len(months)} months tracked):")
    for month, revenue in months[:12]:
        print(f"  {month}: {revenue:,.2f}")
    if len(months) > 12:
        print(f"  ... and {len(months) - 12} more months")

    days = sorted(summary["daily_sales"].items())
    if days:
        print(f"\nDaily revenue tracked for {len(days)} days ({days[0][0]} to {days[-1][0]})")
    else:
        print("\nNo daily revenue tracked")


def run_event_bank_pipeline(df_sales_denormalized, output_folder="event_bank", batch_size=5000):
    """
    Optional data-engineering demo: chunks fact_sales into JSON batches
    and replays them through an incremental-metrics processor. Not run
    automatically - call explicitly (or flip EXPORT_EVENT_BANK on).
    """

    batch_files = create_historical_event_bank(
        df_sales_denormalized, output_folder=output_folder, batch_size=batch_size
    )

    summary = process_event_bank(batch_files)
    display_event_summary(summary)

    return summary


