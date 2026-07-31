import numpy as np
import pandas as pd
import os

TARGET_MAP = {
    "churn": ["churn_label"],
    "ltv": ["customer_ltv"],
    "demand": ["total_units_sold", "total_revenue"],
    "forecast": ["future_revenue"],
    "recommendation": [],
}

def create_customer_features(df_sales, df_customers):

    sales = df_sales.copy()
    sales["sales_date"] = pd.to_datetime(sales["sales_date"])

    customer_features = (
        sales
        .groupby("customer_sk")
        .agg(
            total_orders=("sales_id", "count"),
            total_spent=("total_amount", "sum"),
            average_order_value=("total_amount", "mean"),
            first_purchase_date=("sales_date", "min"),
            last_purchase_date=("sales_date", "max"),
            products_purchased=("product_sk", "nunique"),
            stores_used=("store_sk", "nunique"),
            campaigns_used=("campaign_sk", "nunique")
        )
        .reset_index()
    )

    reference_date = sales["sales_date"].max()

    customer_features["customer_lifetime_days"] = (
        reference_date - customer_features["first_purchase_date"]
    ).dt.days

    customer_features["days_since_last_purchase"] = (
        reference_date - customer_features["last_purchase_date"]
    ).dt.days

    customer_features = customer_features.merge(
        df_customers[["customer_sk", "customer_segment"]],
        on="customer_sk",
        how="left"
    )

    return customer_features


def create_product_features(df_sales, df_products):

    sales = df_sales.copy()

    product_features = (
        sales
        .groupby("product_sk")
        .agg(
            total_units_sold=("sales_id", "count"),
            total_revenue=("total_amount", "sum"),
            average_sales_value=("total_amount", "mean"),
            customers_purchased=("customer_sk", "nunique"),
            stores_sold_in=("store_sk", "nunique")
        )
        .reset_index()
    )

    product_features = product_features.merge(
        df_products[["product_sk", "category", "brand"]],
        on="product_sk",
        how="left"
    )

    return product_features


def create_sales_forecast_features(df_sales):

    sales = df_sales.copy()
    sales["sales_date"] = pd.to_datetime(sales["sales_date"])

    daily_sales = (
        sales
        .groupby(sales["sales_date"].dt.date)
        .agg(
            daily_revenue=("total_amount", "sum"),
            transaction_count=("sales_id", "count")
        )
        .reset_index()
    )

    daily_sales.rename(columns={"sales_date": "date"}, inplace=True)
    daily_sales["date"] = pd.to_datetime(daily_sales["date"])

    daily_sales["day_of_week"] = daily_sales["date"].dt.dayofweek
    daily_sales["month"] = daily_sales["date"].dt.month
    daily_sales["quarter"] = daily_sales["date"].dt.quarter

    daily_sales["rolling_7_day_average"] = daily_sales["daily_revenue"].rolling(7).mean()
    daily_sales["rolling_30_day_average"] = daily_sales["daily_revenue"].rolling(30).mean()

    return daily_sales


def create_customer_product_interactions(df_sales):

    interactions = (
        df_sales
        .groupby(["customer_sk", "product_sk"])
        .agg(
            purchase_count=("sales_id", "count"),
            total_spent=("total_amount", "sum"),
            average_spend=("total_amount", "mean")
        )
        .reset_index()
    )

    return interactions


def create_all_ml_feature_tables(df_sales, df_customers, df_products):

    return {
        "customer_features": create_customer_features(df_sales, df_customers),
        "product_features": create_product_features(df_sales, df_products),
        "sales_forecast_features": create_sales_forecast_features(df_sales),
        "customer_product_interactions": create_customer_product_interactions(df_sales)
    }


def clean_ml_dataframe(df, target_columns=None):

    df = df.copy()
    target_columns = target_columns or []

    df.drop_duplicates(inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    numeric_columns = df.select_dtypes(include=np.number).columns

    for col in numeric_columns:
        if col in target_columns:
            continue
        df[col] = df[col].fillna(df[col].median())

    categorical_columns = df.select_dtypes(include=["object", "category"]).columns

    for col in categorical_columns:
        df[col] = df[col].fillna("Unknown")

    return df.reset_index(drop=True)


def validate_ml_dataframe(df, name, target_columns=None):

    print("\n" + "=" * 60)
    print(f"ML DATASET CHECK: {name}")
    print("=" * 60)

    target_columns = target_columns or []

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nMissing Values:")
    missing = df.isna().sum().sort_values(ascending=False)
    print(missing[missing > 0])

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nConstant Columns:")
    print([col for col in df.columns if df[col].nunique(dropna=False) <= 1])

    print("\nTarget Distribution:")
    for target in target_columns:
        if target in df.columns:
            print(target)
            print(df[target].value_counts(dropna=False))

    print("\nNumeric Summary:")
    print(df.select_dtypes(include=np.number).describe().T)


def clean_all_ml_training_datasets(final_ml_datasets):

    cleaned_ml = {}

    for name, df in final_ml_datasets.items():
        print(f"\nCleaning {name}")
        cleaned = clean_ml_dataframe(df, target_columns=TARGET_MAP.get(name, []))
        validate_ml_dataframe(cleaned, name, target_columns=TARGET_MAP.get(name, []))
        cleaned_ml[name] = cleaned

    return cleaned_ml

