import pandas as pd


def create_final_churn_dataset(customer_features, df_customers):

    df = customer_features.copy()

    df = df.merge(
        df_customers[["customer_sk", "first_name", "last_name"]],
        on="customer_sk",
        how="left"
    )

    df["churn_label"] = (df["days_since_last_purchase"] > 90).astype(int)

    return df


def create_final_ltv_dataset(customer_features, df_customers):

    df = customer_features.copy()
    df["customer_ltv"] = df["total_spent"]

    return df


def create_final_demand_dataset(product_features, df_products, df_dates):

    df = product_features.copy()

    df = df.merge(
        df_products[["product_sk", "product_name", "category", "brand", "origin_location"]],
        on="product_sk",
        how="left"
    )

    # product_features has no date column today, so this branch is a
    # no-op for now. Left in place in case a date-aware demand feature
    # is added later.
    if "date" in df.columns:

        df["date"] = pd.to_datetime(df["date"])

        df = df.merge(
            df_dates[["full_date", "year", "month", "quarter", "weekday"]],
            left_on="date",
            right_on="full_date",
            how="left"
        )

    return df


def create_final_forecast_dataset(sales_forecast_features, df_dates):

    df = sales_forecast_features.copy()

    df["date"] = pd.to_datetime(df["date"])

    # sales_forecast_features already derives "month" and "quarter"
    # locally, so only the genuinely new calendar columns (year,
    # weekday) are merged in here to avoid a month_x/month_y collision.
    df = df.merge(
        df_dates[["full_date", "year", "weekday"]],
        left_on="date",
        right_on="full_date",
        how="left"
    )

    df["future_revenue"] = df["daily_revenue"].shift(-1)

    df.dropna(inplace=True)

    return df


def create_final_ml_datasets(ml_features, df_customers, df_products, df_dates):

    final_datasets = {}

    final_datasets["churn"] = create_final_churn_dataset(
        ml_features["customer_features"], df_customers
    )

    final_datasets["ltv"] = create_final_ltv_dataset(
        ml_features["customer_features"], df_customers
    )

    final_datasets["demand"] = create_final_demand_dataset(
        ml_features["product_features"], df_products, df_dates
    )

    final_datasets["forecast"] = create_final_forecast_dataset(
        ml_features["sales_forecast_features"], df_dates
    )

    # No target column - customer x product interaction matrix,
    # used for a recommendation model instead of a supervised label.
    final_datasets["recommendation"] = ml_features["customer_product_interactions"]

    return final_datasets
