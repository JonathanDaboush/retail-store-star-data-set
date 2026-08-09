"""
Retail Store Star-Schema ML Pipeline
------------------------------------
Extracts a Kaggle star-schema retail dataset, validates it, engineers
customer/product/forecast/interaction feature tables, builds model-ready
datasets for churn / LTV / demand / forecast / recommendation, trains a
model per task, and exposes a self-adjusting retraining entrypoint that
folds in new incoming sales data and only replaces a deployed model when
the retrained candidate actually performs better.

This version makes two changes on top of the previously-corrected script:

CHANGE 1 - ROW USAGE MANIFEST (replaces the old "learning_dataset.csv /
remaining_dataset.csv" export). Instead of dumping full copies of a
handful of "used" vs "unused" fact-table rows, the pipeline now writes
one CSV per star-schema table (fact_sales, customers, products, stores,
salespersons, campaigns, dates) listing every primary-key value in that
table together with a `used_in_training` flag. That flag is derived by
walking the foreign keys of every fact_sales row that was actually pulled
into the current learning sample. This gives you a complete, per-table
"what did this run touch" record you can filter (used_in_training ==
True) to safely delete/archive already-consumed rows from your own
staging tables. See build_row_usage_manifest() / export_row_usage_manifest().

CHANGE 2 - INDEPENDENT, IMPORTABLE FUNCTIONS. The script used to run as
one long sequence of top-level statements (download -> clean -> engineer
-> train -> ...), which fires immediately on `import` and relies on
module-level globals (see the old `ingest_new_data_and_retrain`, which
used `global df_sales_denormalized, ...`). Everything is now organized
into small, single-purpose functions that take explicit inputs and
return explicit outputs, plus a handful of orchestration functions:

    run_initial_pipeline()                     -> context dict
    ingest_new_data_and_retrain(new_df, ctx)    -> new context dict

`context` is a plain dict holding every intermediate artifact (cleaned
tables, feature tables, processed/split datasets, the model registry,
trained model packages, the row-usage manifest, etc). Your app can hold
onto that dict (in memory, in a session, pickled to disk - whatever fits)
and pass it straight back into ingest_new_data_and_retrain() whenever new
sales rows show up, with no reliance on module globals.

Nothing runs automatically on import. The `if __name__ == "__main__":`
block at the bottom reproduces the original "run everything once" script
behavior for standalone use (`python retail_pipeline.py`).
"""

import numpy as np
import pandas as pd
import os
import glob
import json
import shutil
import pickle
import time

try:
    import kagglehub
except ImportError:  # Local source files are the default runtime data source.
    kagglehub = None
import psutil

from typing import Set

from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder,
    MinMaxScaler
)
from sklearn.model_selection import (
    train_test_split,
    GroupShuffleSplit,
    RandomizedSearchCV,
    GridSearchCV,
    TimeSeriesSplit,
    StratifiedKFold,
    KFold
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    roc_auc_score
)
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor  # noqa: F401 (kept for future model swaps)

try:
    from xgboost import XGBRegressor, XGBClassifier
    from lightgbm import LGBMRegressor
except ImportError:
    XGBRegressor = XGBClassifier = LGBMRegressor = None


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

DEFAULT_N_JOBS = 1

# How many rows to actually use for feature engineering / model
# training. Everything else is set aside as "unused" rows.
LEARNING_SAMPLE_SIZE = 5000

DATASET_SLUG = "shrinivasv/retail-store-star-schema-dataset"
LOCAL_OUTPUT_DIRS = ("event_bank",)

MODEL_SAVE_DIR = "models"

# CHANGE 1: replaces the old EXPORT_REMAINING_DATA / REMAINING_DATA_FOLDER
# flags. Controls whether run_initial_pipeline() / ingest_new_data_and_retrain()
# write out the per-table row-usage manifest described above.
EXPORT_ROW_USAGE_MANIFEST = True
ROW_USAGE_FOLDER = "row_usage_manifest"

# Optional, off-by-default diagnostics (kept from the original script,
# just no longer run unconditionally at import time).
EXPORT_EVENT_BANK = False
RUN_RAW_FEATURE_DIAGNOSTICS = False

# Columns that represent monetary values. These are NEVER clipped
# or scaled - dollar figures must stay in their original units.
MONEY_COLUMNS = {
    "total_amount",
    "total_spent",
    "average_order_value",
    "total_revenue",
    "average_sales_value",
    "daily_revenue",
    "rolling_7_day_average",
    "rolling_30_day_average",
    "customer_ltv",
    "future_revenue",
    "average_spend",
}

# Per final-dataset: which columns are "target-ish" and must never
# be scaled/clipped either (this may include more than the single
# column actually used as y - e.g. demand has two candidate targets).
TARGET_MAP = {
    "churn": ["churn_label"],
    "ltv": ["customer_ltv"],
    "demand": ["total_units_sold", "total_revenue"],
    "forecast": ["future_revenue"],
    "recommendation": [],
}

# Per final-dataset: the single column actually used as y for
# train/test splitting. Datasets not listed here (recommendation)
# have no supervised target and are not split.
PRIMARY_TARGET = {
    "churn": "churn_label",
    "ltv": "customer_ltv",
    "demand": "total_units_sold",
    "forecast": "future_revenue",
}

# Per final-dataset: extra columns that must be dropped entirely
# (not just left unscaled) because they leak the target - they are
# either the exact source of the label or another target candidate
# that would let the model trivially back the answer out.
LEAKAGE_COLUMNS = {
    "churn": ["days_since_last_purchase"],   # churn_label = days_since_last_purchase > 90
    "ltv": ["total_spent"],                  # customer_ltv is literally a copy of total_spent
    "demand": ["total_revenue"],             # companion target to total_units_sold
    "forecast": [],
}

# Per final-dataset: a column that must survive preprocessing
# untouched (no scaling/clipping/removal) because it's needed to
# do a time-ordered train/test split. Dropped from X automatically
# once the split is done.
TIME_COLUMNS = {
    "forecast": "date",
}

HIGH_CARDINALITY_THRESHOLD = 0.95
LOW_VARIANCE_THRESHOLD = 0.99
CLIP_LOWER = 0.01
CLIP_UPPER = 0.99

# PERFORMANCE: how expensive tune_model()'s hyperparameter search is.
# RandomizedSearchCV does (roughly) SEARCH_ITERATIONS * CV_SPLITS model
# fits per supervised target, so with 4 supervised targets the old
# defaults (20 * 5 = 100 fits each) meant ~400 model fits total, on top
# of param grids that allow up to 300 trees / depth 7. Lower these (or
# override per-call via tune_model(..., iterations=.., cv_splits=..))
# if a run needs to be faster; raise them again once you have more time
# or more compute and want a more thorough search.
SEARCH_ITERATIONS = 10
CV_SPLITS = 3

# Tables required to be present in the downloaded dataset.
REQUIRED_TABLES = [
    "dim_campaigns",
    "dim_customers",
    "dim_dates",
    "dim_products",
    "dim_salespersons",
    "dim_stores",
    "fact_sales_denormalized",
    "fact_sales_normalized",
]


# ============================================================
# STAR SCHEMA METADATA
# ============================================================

PRIMARY_KEYS = {
    "campaigns": ["campaign_sk", "campaign_id"],
    "customers": ["customer_sk", "customer_id"],
    "dates": ["date_sk"],
    "products": ["product_sk", "product_id"],
    "salespersons": ["salesperson_sk", "salesperson_id"],
    "stores": ["store_sk", "store_id"],
    "fact_sales": ["sales_sk", "sales_id"]
}

FOREIGN_KEYS = {
    "fact_sales": {
        "customer_sk": ("customers", "customer_sk"),
        "product_sk": ("products", "product_sk"),
        "store_sk": ("stores", "store_sk"),
        "salesperson_sk": ("salespersons", "salesperson_sk"),
        "campaign_sk": ("campaigns", "campaign_sk"),
        "sales_date": ("dates", "full_date")
    }
}


# ============================================================
# 0. CLEAN UP OUTPUT FROM THE PREVIOUS RUN
# ============================================================

def cleanup_previous_run(dataset_slug=DATASET_SLUG, local_output_dirs=LOCAL_OUTPUT_DIRS):

    print("=" * 80)
    print("CLEANING UP OUTPUT FROM PREVIOUS RUN")
    print("=" * 80)

    for folder in local_output_dirs:
        if os.path.exists(folder):
            shutil.rmtree(folder, ignore_errors=True)
            print(f"Removed local output folder: {folder}")
        else:
            print(f"No local output folder to remove: {folder}")

    kagglehub_cache_root = os.path.expanduser("~/.cache/kagglehub/datasets")
    dataset_cache_path = os.path.join(kagglehub_cache_root, *dataset_slug.split("/"))

    if os.path.exists(dataset_cache_path):
        shutil.rmtree(dataset_cache_path, ignore_errors=True)
        print(f"Removed cached kaggle dataset: {dataset_cache_path}")
    else:
        print(f"No cached kaggle dataset found at: {dataset_cache_path}")

    print("Cleanup complete.\n")


# ============================================================
# 1. EXTRACT: DOWNLOAD AND LOAD DATASETS
# ============================================================

def download_and_load_datasets(dataset_slug=DATASET_SLUG, data_dir=None, max_fact_rows=None):
    """
    Loads the bundled immutable CSVs by default. Kaggle is only a fallback
    for users who intentionally do not have the project source data.
    """

    local_dir = data_dir or os.path.join(os.path.dirname(__file__), "original_data")
    if os.path.isdir(local_dir):
        csv_files = glob.glob(os.path.join(local_dir, "*.csv"))
    else:
        if kagglehub is None:
            raise RuntimeError("No local source data found and kagglehub is not installed.")
        path = kagglehub.dataset_download(dataset_slug)
        csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)

    datasets = {}
    for file in csv_files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        # The published file is named dim_fact_sales_denormalized although it
        # is the denormalized fact table used throughout this pipeline.
        if file_name == "dim_fact_sales_denormalized":
            file_name = "fact_sales_denormalized"
        datasets[file_name] = pd.read_csv(
            file,
            nrows=max_fact_rows if file_name in {"fact_sales_denormalized", "fact_sales_normalized"} else None,
        )

    print("LOADED DATASETS\n")
    for name, df in datasets.items():
        print(f"{name}: {df.shape}")

    return datasets


def extract_star_schema_frames(datasets, required_tables=REQUIRED_TABLES):
    """
    Validates that every required table is present and returns a dict of
    independent copies keyed by the same names (dim_customers, ...,
    fact_sales_denormalized, fact_sales_normalized).
    """

    for table in required_tables:
        if table not in datasets:
            raise ValueError(f"Missing dataset: {table}")

    frames = {table: datasets[table].copy() for table in required_tables}

    print("\nDATAFRAME HEADS\n")
    for label, df in frames.items():
        print(label)
        print(df.head(), "\n")

    return frames


# ============================================================
# STAR SCHEMA VALIDATION
# ============================================================

def validate_primary_keys(df, table_name):

    print("\n==============================")
    print(f"PRIMARY KEY CHECK: {table_name}")
    print("==============================")

    if table_name not in PRIMARY_KEYS:
        print("No primary key rules defined.")
        return True

    valid = True

    for col in PRIMARY_KEYS[table_name]:

        if col not in df.columns:
            print(f"Missing column: {col}")
            valid = False
            continue

        nulls = df[col].isna().sum()
        duplicates = df[col].duplicated().sum()

        print(f"{col}: Nulls={nulls}, Duplicates={duplicates}")

        if nulls > 0 or duplicates > 0:
            valid = False

    print("PRIMARY KEY STATUS:", "PASS" if valid else "FAIL")

    return valid


def validate_foreign_keys(tables):

    print("\n==============================")
    print("FOREIGN KEY CHECK")
    print("==============================")

    valid = True

    for child_table, relationships in FOREIGN_KEYS.items():

        child_df = tables[child_table]

        for fk_column, reference in relationships.items():

            parent_table, parent_column = reference
            parent_df = tables[parent_table]

            if fk_column not in child_df.columns:
                print(f"Missing FK column {child_table}.{fk_column}")
                valid = False
                continue

            child_values = child_df[fk_column]
            parent_values = parent_df[parent_column]
            # Facts carry timestamps while the date dimension is day-grain.
            if fk_column == "sales_date" and parent_column == "full_date":
                child_values = pd.to_datetime(child_values, errors="coerce").dt.normalize()
                parent_values = pd.to_datetime(parent_values, errors="coerce").dt.normalize()
            missing_count = (~child_values.isin(parent_values)).sum()

            print(f"{child_table}.{fk_column} -> {parent_table}.{parent_column} Missing={missing_count}")

            if missing_count > 0:
                valid = False

    print("FOREIGN KEY STATUS:", "PASS" if valid else "FAIL")

    return valid


def validate_business_rules(tables):

    print("\n==============================")
    print("BUSINESS RULE CHECK")
    print("==============================")

    valid = True

    if "campaigns" in tables:
        df = tables["campaigns"]
        if "start_date_sk" in df.columns and "end_date_sk" in df.columns:
            invalid_dates = (df["end_date_sk"] < df["start_date_sk"]).sum()
            print("Campaign invalid date ranges:", invalid_dates)
            if invalid_dates > 0:
                valid = False

    if "fact_sales" in tables:
        df = tables["fact_sales"]
        if "total_amount" in df.columns:
            negative_sales = (df["total_amount"] < 0).sum()
            print("Negative sales:", negative_sales)
            if negative_sales > 0:
                valid = False

    print("BUSINESS RULE STATUS:", "PASS" if valid else "FAIL")

    return valid


def validate_star_schema(tables):

    print("\n\n######## STAR SCHEMA VALIDATION ########")

    results = {}

    for table_name, df in tables.items():
        results[f"{table_name}_PK"] = validate_primary_keys(df, table_name)

    results["FOREIGN_KEYS"] = validate_foreign_keys(tables)
    results["BUSINESS_RULES"] = validate_business_rules(tables)

    print("\n========== FINAL RESULT ==========")
    for check, result in results.items():
        print(check, "PASS" if result else "FAIL")

    return results


def clean_dataframe(df):
    """
    General purpose cleaning for raw star-schema dimension/fact tables.
    """

    df = df.copy()

    df.drop_duplicates(inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    return df.reset_index(drop=True)


def clean_all_star_schema_tables(raw_frames):
    """
    Applies clean_dataframe() to every raw star-schema frame produced by
    extract_star_schema_frames(). Returns a dict with the same keys.
    """

    cleaned = {name: clean_dataframe(df) for name, df in raw_frames.items()}

    print("\nRAW STAR SCHEMA TABLES CLEANED")
    for name, df in cleaned.items():
        print(f"{name}: {df.shape}")

    return cleaned


def build_star_schema_tables(cleaned_frames):
    """
    Maps the raw table names (dim_customers, fact_sales_denormalized, ...)
    onto the short names used by PRIMARY_KEYS / FOREIGN_KEYS / validation
    (customers, fact_sales, ...). fact_sales_normalized is intentionally
    not part of this mapping - only the denormalized fact table is used
    downstream.
    """

    star_schema_tables = {
        "campaigns": cleaned_frames["dim_campaigns"],
        "customers": cleaned_frames["dim_customers"],
        "dates": cleaned_frames["dim_dates"],
        "products": cleaned_frames["dim_products"],
        "salespersons": cleaned_frames["dim_salespersons"],
        "stores": cleaned_frames["dim_stores"],
        "fact_sales": cleaned_frames["fact_sales_denormalized"],
    }

    return star_schema_tables


# ============================================================
# FEATURE ENGINEERING
# ============================================================

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


# ============================================================
# FINAL MODEL-READY DATASETS
# ============================================================

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


def create_temporal_ml_datasets(df_sales, df_customers, df_products):
    """Create labels strictly after a historical cutoff to avoid target leakage."""
    sales = df_sales.copy()
    sales["sales_date"] = pd.to_datetime(sales["sales_date"], errors="coerce")
    sales = sales.dropna(subset=["sales_date", "total_amount"])
    dates = np.sort(sales["sales_date"].dt.normalize().unique())
    if len(dates) < 10:
        raise ValueError("At least ten distinct sales dates are required for temporal ML datasets.")
    cutoff = pd.Timestamp(dates[max(1, int(len(dates) * .8) - 1)])
    history, future = sales[sales.sales_date.dt.normalize() <= cutoff], sales[sales.sales_date.dt.normalize() > cutoff]

    customer = create_customer_features(history, df_customers)
    future_value = future.groupby("customer_sk").total_amount.sum().rename("customer_ltv")
    customer = customer.merge(future_value, on="customer_sk", how="left")
    customer["customer_ltv"] = customer["customer_ltv"].fillna(0.0)
    customer["churn_label"] = (~customer.customer_sk.isin(future.customer_sk)).astype(int)
    churn = customer.drop(columns=["customer_ltv"])
    ltv = customer.drop(columns=["churn_label"])

    demand = create_product_features(history, df_products)
    future_units = future.groupby("product_sk").sales_id.count().rename("total_units_sold")
    demand = demand.drop(columns=["total_units_sold", "total_revenue"], errors="ignore").merge(future_units, on="product_sk", how="left")
    demand["total_units_sold"] = demand["total_units_sold"].fillna(0)

    daily = history.groupby(history.sales_date.dt.normalize()).agg(daily_revenue=("total_amount", "sum"), transaction_count=("sales_id", "count")).reset_index(names="date").sort_values("date")
    daily["future_revenue"] = daily.daily_revenue.shift(-1)
    daily["lag_1_revenue"] = daily.daily_revenue.shift(1)
    daily["rolling_7_day_average"] = daily.daily_revenue.shift(1).rolling(7, min_periods=1).mean()
    daily["day_of_week"] = daily.date.dt.dayofweek
    daily["month"] = daily.date.dt.month
    forecast = daily.dropna(subset=["future_revenue", "lag_1_revenue"])
    recommendation = create_customer_product_interactions(history)
    return {"churn": churn, "ltv": ltv, "demand": demand, "forecast": forecast, "recommendation": recommendation}

def create_final_ml_datasets(ml_features, df_customers, df_products, df_dates, df_sales=None):
    if df_sales is not None:
        return create_temporal_ml_datasets(df_sales, df_customers, df_products)

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


# ============================================================
# ML PREPROCESSING PIPELINE
# ============================================================

def remove_identifier_columns(df):

    remove = []

    for c in df.columns:
        lc = c.lower()
        if lc.endswith("_id") or lc.endswith("_sk") or lc in ("id", "sales_id"):
            remove.append(c)

    if remove:
        print("Removed identifier columns:", remove)

    return df.drop(columns=remove, errors="ignore")


def remove_leakage_columns(df, name):

    remove = [c for c in LEAKAGE_COLUMNS.get(name, []) if c in df.columns]

    if remove:
        print("Removed leakage-prone columns (directly define/duplicate the target):", remove)

    return df.drop(columns=remove, errors="ignore")


def remove_datetime_columns(df, protected):

    remove = [
        c for c in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[c]) and c not in protected
    ]

    if remove:
        print("Removed raw datetime columns (already captured as engineered features):", remove)

    return df.drop(columns=remove, errors="ignore")


def remove_constant_columns(df, protected):

    remove = []

    for c in df.columns:

        if c in protected:
            continue

        freq = df[c].value_counts(dropna=False, normalize=True)

        if len(freq) <= 1 or freq.iloc[0] >= LOW_VARIANCE_THRESHOLD:
            remove.append(c)

    if remove:
        print("Removed constant / near-constant columns:", remove)

    return df.drop(columns=remove, errors="ignore")


def remove_high_cardinality(df, protected):

    remove = []

    for c in df.columns:

        if c in protected:
            continue

        if pd.api.types.is_numeric_dtype(df[c]):
            continue

        ratio = df[c].nunique(dropna=False) / len(df)

        if ratio >= HIGH_CARDINALITY_THRESHOLD:
            remove.append(c)

    if remove:
        print("Removed high-cardinality columns:", remove)

    return df.drop(columns=remove, errors="ignore")


def encode_categories(df, max_one_hot_categories=10):
    """
    Encodes categorical variables so no raw string values remain.
    <= max_one_hot_categories unique values: one-hot encoded.
    > max_one_hot_categories unique values: label encoded.
    """

    df = df.copy()
    encoders = {}

    categorical_columns = df.select_dtypes(include=["object", "category"]).columns

    if len(categorical_columns) == 0:
        return df, encoders

    one_hot_columns = []
    label_columns = []

    for col in categorical_columns:
        if df[col].nunique() <= max_one_hot_categories:
            one_hot_columns.append(col)
        else:
            label_columns.append(col)

    if one_hot_columns:

        one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = one_hot_encoder.fit_transform(df[one_hot_columns])

        encoded_df = pd.DataFrame(
            encoded,
            columns=one_hot_encoder.get_feature_names_out(one_hot_columns),
            index=df.index
        )

        df = df.drop(columns=one_hot_columns)
        df = pd.concat([df, encoded_df], axis=1)

        encoders["one_hot"] = one_hot_encoder

    for col in label_columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder

    return df, encoders


def clip_numeric_columns(df, exclude):

    df = df.copy()

    for c in df.select_dtypes(include=np.number).columns:

        if c in exclude:
            continue

        lower = df[c].quantile(CLIP_LOWER)
        upper = df[c].quantile(CLIP_UPPER)
        df[c] = df[c].clip(lower, upper)

    return df


def scale_numeric_columns(df, exclude):

    df = df.copy()

    numeric = [c for c in df.select_dtypes(include=np.number).columns if c not in exclude]

    if numeric:
        scaler = MinMaxScaler()
        df[numeric] = scaler.fit_transform(df[numeric])

    return df


def get_protected_columns(name) -> Set[str]:
    """
    Columns that must never be removed for being constant/high-
    cardinality, and never clipped/scaled: money columns, this
    dataset's target column(s), and any protected time column.
    """

    protected = set(MONEY_COLUMNS) | set(TARGET_MAP.get(name, []))

    if name in TIME_COLUMNS:
        protected.add(TIME_COLUMNS[name])

    return protected


def preprocess_ml_dataset(name, df):

    print("\n" + "=" * 80)
    print(name.upper())
    print("=" * 80)

    df = df.copy()
    print("Original shape:", df.shape)

    protected = get_protected_columns(name)

    df = remove_identifier_columns(df)
    df = remove_leakage_columns(df, name)
    df = remove_datetime_columns(df, protected)
    df = remove_constant_columns(df, protected)
    df = remove_high_cardinality(df, protected)

    # Splitting happens before fitting encoders/scalers.  Fitting here used
    # test-set distributions and categories, which leaked evaluation data.
    if name == "recommendation":
        df, encoders = encode_categories(df)
        df = clip_numeric_columns(df, exclude=protected)
        df = scale_numeric_columns(df, exclude=protected)
    else:
        encoders = {}

    print("Final shape:", df.shape)
    print("Remaining object columns:", list(df.select_dtypes(include=["object", "category"]).columns))
    print("\nHead:")
    print(df.head())

    return df, encoders

def fit_transform_train_test(X_train, X_test):
    """Fit imputation, clipping, encoding, and scaling on training rows only."""
    train, test = X_train.copy(), X_test.copy()
    numeric = list(train.select_dtypes(include=np.number).columns)
    for col in numeric:
        median = train[col].median()
        train[col] = train[col].fillna(median)
        test[col] = test[col].fillna(median)
        low, high = train[col].quantile(CLIP_LOWER), train[col].quantile(CLIP_UPPER)
        train[col] = train[col].clip(low, high)
        test[col] = test[col].clip(low, high)
        span = high - low
        if span:
            train[col] = (train[col] - low) / span
            test[col] = (test[col] - low) / span
    categorical = list(train.select_dtypes(include=["object", "category"]).columns)
    if categorical:
        train = pd.get_dummies(train, columns=categorical, dtype=float)
        test = pd.get_dummies(test, columns=categorical, dtype=float).reindex(columns=train.columns, fill_value=0)
    return train.astype(float), test.astype(float)


def preprocess_all_datasets(final_ml_clean):
    """
    Runs preprocess_ml_dataset() over every entry in final_ml_clean.
    Returns (processed_ml_datasets, dataset_encoders), both dicts keyed
    by dataset name (churn, ltv, demand, forecast, recommendation).
    """

    processed_ml_datasets = {}
    dataset_encoders = {}

    for dataset_name, dataset in final_ml_clean.items():
        processed, encoders = preprocess_ml_dataset(dataset_name, dataset)
        processed_ml_datasets[dataset_name] = processed
        dataset_encoders[dataset_name] = encoders

    print("\n" + "=" * 80)
    print("PROCESSED (MODEL-READY) DATASETS")
    print("=" * 80)
    for name, df in processed_ml_datasets.items():
        print(f"{name:20s} {df.shape}")

    return processed_ml_datasets, dataset_encoders


# ============================================================
# CREATE ALL TRAIN / TEST SPLITS
# ============================================================

def split_ml_dataset(
    df,
    target_column,
    max_rows=50000,
    split_type="random",
    test_size=0.2,
    group_column=None,
    time_column=None,
    random_state=42
):
    """
    Splits ML datasets safely.

    split_type: "random" | "time" | "group"
    time_column / group_column are used for ordering/grouping only
    and are dropped from the returned feature matrices - they are
    never fed to the model as raw features.
    """

    df = df.copy()

    if target_column not in df.columns:
        raise ValueError(f"{target_column} not found in dataframe")

    if split_type not in ["random", "time", "group"]:
        raise ValueError("split_type must be random, time, or group")

    if len(df) > max_rows:

        if split_type == "time":
            df = df.sort_values(time_column)

        sample_df = df.iloc[:max_rows].copy()
        unused_df = df.iloc[max_rows:].copy()

    else:
        sample_df = df.copy()
        unused_df = pd.DataFrame()

    X = sample_df.drop(columns=[target_column])
    y = sample_df[target_column]

    if split_type == "random":

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    elif split_type == "time":

        if time_column is None:
            raise ValueError("time_column required for time split")

        ordered_df = sample_df.sort_values(time_column)
        split_index = int(len(ordered_df) * (1 - test_size))

        train_df = ordered_df.iloc[:split_index]
        test_df = ordered_df.iloc[split_index:]

        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]

        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

    elif split_type == "group":

        if group_column is None:
            raise ValueError("group_column required for group split")

        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_index, test_index = next(splitter.split(X, y, groups=sample_df[group_column]))

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

    drop_extra = [c for c in [time_column, group_column] if c]
    if drop_extra:
        X_train = X_train.drop(columns=[c for c in drop_extra if c in X_train.columns], errors="ignore")
        X_test = X_test.drop(columns=[c for c in drop_extra if c in X_test.columns], errors="ignore")

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    unused_df = unused_df.reset_index(drop=True)

    print("=" * 50)
    print("ML SPLIT COMPLETE")
    print("=" * 50)
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Unused rows:", len(unused_df))
    print("Features:", X_train.shape[1])

    return X_train, X_test, y_train, y_test, unused_df


SPLIT_CONFIG = {
    "churn": dict(split_type="random"),
    "ltv": dict(split_type="random"),
    "demand": dict(split_type="random"),
    "forecast": dict(split_type="time", time_column=TIME_COLUMNS["forecast"]),
}


def split_all_datasets(processed_ml_datasets, sample_size=LEARNING_SAMPLE_SIZE,
                        test_size=0.20, random_state=RANDOM_STATE):
    """
    Runs split_ml_dataset() for every supervised target in PRIMARY_TARGET,
    using each dataset's configured split strategy (SPLIT_CONFIG). Returns
    a dict: name -> (X_train, X_test, y_train, y_test).
    """

    split_datasets = {}

    for name, target_column in PRIMARY_TARGET.items():

        print(f"\nSplitting: {name} (target = {target_column})")

        X_train, X_test, y_train, y_test, _unused = split_ml_dataset(
            processed_ml_datasets[name],
            target_column=target_column,
            max_rows=sample_size,
            test_size=test_size,
            random_state=random_state,
            **SPLIT_CONFIG[name]
        )

        split_datasets[name] = (X_train, X_test, y_train, y_test)

    print("\n" + "=" * 80)
    print("TRAIN / TEST SUMMARY")
    print("=" * 80)

    for name, (Xtr, Xte, ytr, yte) in split_datasets.items():
        print(f"\n{name.upper()}")
        print("-" * 50)
        print("X Train:", Xtr.shape)
        print("X Test :", Xte.shape)
        print("y Train:", ytr.shape)
        print("y Test :", yte.shape)

    return split_datasets


# ============================================================
# ROW USAGE MANIFEST (replaces the old learning/remaining CSV dump)
# ============================================================

def split_learning_sales(df_sales_denormalized, sample_size=LEARNING_SAMPLE_SIZE):
    """
    Date-orders the fact_sales rows and splits them into the "learning
    sample" (the earliest `sample_size` rows - the ones this run's
    feature engineering treats as the historical training pool) and
    everything else. This is also the boundary the row-usage manifest
    below uses to decide "used" vs "not yet used".
    """

    ordered = df_sales_denormalized.sort_values("sales_date").reset_index(drop=True)

    learning_sales = ordered.head(sample_size).reset_index(drop=True)
    remaining_sales = ordered.iloc[sample_size:].reset_index(drop=True)

    return learning_sales, remaining_sales


def build_row_usage_manifest(learning_sales, remaining_sales, star_schema_tables):
    """
    Builds a per-table manifest of exactly which primary-key values were
    (and were not) pulled into this run's learning sample.

    fact_sales: used = sales_sk values in learning_sales,
                unused = sales_sk values in remaining_sales.
    Every other star-schema table: used = the set of that table's
        primary-key values referenced by the foreign keys of the used
        fact_sales rows; unused = every other primary-key value present
        in that table.

    Returns: dict table_name -> {
        "pk_column": str,
        "used_ids": sorted list,
        "unused_ids": sorted list,
        "used_count": int,
        "unused_count": int,
    }
    """

    manifest = {}

    fact_pk = PRIMARY_KEYS["fact_sales"][0]  # sales_sk

    used_fact_ids = set(learning_sales[fact_pk].dropna())
    unused_fact_ids = set(remaining_sales[fact_pk].dropna()) if len(remaining_sales) else set()

    manifest["fact_sales"] = {
        "pk_column": fact_pk,
        "used_ids": sorted(used_fact_ids, key=str),
        "unused_ids": sorted(unused_fact_ids, key=str),
        "used_count": len(used_fact_ids),
        "unused_count": len(unused_fact_ids),
    }

    for fk_column, (table_name, parent_column) in FOREIGN_KEYS["fact_sales"].items():

        if fk_column not in learning_sales.columns or table_name not in star_schema_tables:
            continue

        used_values = set(learning_sales[fk_column].dropna())
        all_values = set(star_schema_tables[table_name][parent_column].dropna())
        unused_values = all_values - used_values

        manifest[table_name] = {
            "pk_column": parent_column,
            "used_ids": sorted(used_values, key=str),
            "unused_ids": sorted(unused_values, key=str),
            "used_count": len(used_values),
            "unused_count": len(unused_values),
        }

    return manifest


def export_row_usage_manifest(manifest, output_folder=ROW_USAGE_FOLDER):
    """
    Writes one CSV per table (<table_name>_row_usage.csv) with columns
    [pk_column, used_in_training], plus a row_usage_summary.json with
    just the counts. Filter any of the CSVs to used_in_training == True
    to get the exact rows this run already consumed - handy for safely
    deleting/archiving them from your own staging tables.
    """

    os.makedirs(output_folder, exist_ok=True)
    written_paths = []

    print("\n" + "=" * 80)
    print("ROW USAGE MANIFEST (used vs. not-yet-used rows, per table)")
    print("=" * 80)

    for table_name, info in manifest.items():

        pk_col = info["pk_column"]

        rows = [{pk_col: v, "used_in_training": True} for v in info["used_ids"]]
        rows += [{pk_col: v, "used_in_training": False} for v in info["unused_ids"]]

        manifest_df = pd.DataFrame(rows)
        path = os.path.join(output_folder, f"{table_name}_row_usage.csv")
        manifest_df.to_csv(path, index=False)
        written_paths.append(path)

        print(f"{table_name:20s} used={info['used_count']:>8}  "
              f"unused={info['unused_count']:>8}  -> {path}")

    summary = {
        table_name: {
            "pk_column": info["pk_column"],
            "used_count": info["used_count"],
            "unused_count": info["unused_count"],
        }
        for table_name, info in manifest.items()
    }

    summary_path = os.path.join(output_folder, "row_usage_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    written_paths.append(summary_path)

    print(f"\nSummary written to: {summary_path}")
    print("Tip: filter each *_row_usage.csv to used_in_training == True to get")
    print("the exact rows already consumed by this run.")

    return written_paths


# ============================================================
# OPTIONAL: RAW (UNSCALED / UNCLIPPED) FEATURE VIEW - FOR INSPECTION ONLY
# ============================================================

def clean_feature_dataset_raw(df):

    df = df.copy()

    df.drop_duplicates(inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].fillna("Unknown")

    return df.reset_index(drop=True)


def inspect_dataset(df, name):

    print("\n" + "=" * 90)
    print(name.upper())
    print("=" * 90)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(list(df.columns))

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    missing = df.isna().sum()
    print(missing[missing > 0])

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    numeric = df.select_dtypes(include=np.number)

    if len(numeric.columns) > 0:
        print("\nNumeric Summary:")
        summary = numeric.describe().T
        summary["skew"] = numeric.skew()
        print(summary)

    categorical = df.select_dtypes(include=["object", "category"])

    print("\nCategorical Columns:")
    for col in categorical.columns:
        print(col, "unique:", df[col].nunique())

    print("\nHEAD:")
    print(df.head())


def run_raw_feature_diagnostics(ml_features):
    """
    Optional, verbose inspection of the unscaled/unclipped feature
    tables. Not run automatically - call explicitly if you want it.
    """

    ml_features_raw_clean = {}

    for name, df in ml_features.items():
        print("\nCleaning (raw view, no scaling/clipping):", name)
        ml_features_raw_clean[name] = clean_feature_dataset_raw(df)

    for name, df in ml_features_raw_clean.items():
        inspect_dataset(df, name)

    return ml_features_raw_clean


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


# ============================================================
# PICKLE SAVE / LOAD HELPERS
# ============================================================

def save_pickle(obj, path):
    """Save a Python object to a pickle file."""

    folder = os.path.dirname(path)

    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(path, "wb") as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved artifact: {path}")


def load_pickle(path):
    """Load a Python object from a pickle file."""

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "rb") as file:
        obj = pickle.load(file)

    print(f"Loaded artifact: {path}")

    return obj


# ============================================================
# MODEL EXPORT / IMPORT WRAPPERS
# ============================================================

def export_model_package(package, folder, filename):

    os.makedirs(folder, exist_ok=True)

    if not filename.endswith(".pkl"):
        filename += ".pkl"

    path = os.path.join(folder, filename)

    save_pickle(package, path)

    print(f"Exported model package: {path}")

    return path


def import_model_package(folder, filename):

    path = os.path.join(folder, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model package not found: {path}")

    package = load_pickle(path)

    print(f"Loaded model package: {path}")

    return package


# ============================================================
# STAR STORE MODEL REGISTRY
# ============================================================

def build_model_registry():
    """
    Factory that returns a FRESH model registry every time it's called.
    Kept as a factory (rather than a shared module-level dict) so that
    every pipeline run / retrain gets its own model instances instead of
    accidentally sharing (and mutating, e.g. via set_params) the same
    objects across runs or across multiple app contexts.
    """

    if XGBRegressor is None or XGBClassifier is None or LGBMRegressor is None:
        raise RuntimeError("ML dependencies are unavailable. Install backend/requirements.txt before training models.")
    return {

        "forecast": {
            "task": "regression",
            "time_series": True,
            "model": XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=DEFAULT_N_JOBS
            ),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0]
            }
        },

        "churn": {
            "task": "classification",
            "time_series": False,
            "model": XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                scale_pos_weight=1,
                n_jobs=DEFAULT_N_JOBS
            ),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0]
            }
        },

        "ltv": {
            "task": "regression",
            "time_series": False,
            "model": LGBMRegressor(
                random_state=RANDOM_STATE,
                n_jobs=DEFAULT_N_JOBS
            ),
            "params": {
                "n_estimators": [100, 200, 300],
                "num_leaves": [15, 31, 63],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [-1, 5, 10],
                "min_child_samples": [10, 20, 50]
            }
        },

        "demand": {
            "task": "regression",
            "time_series": False,
            "model": XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=DEFAULT_N_JOBS
            ),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0]
            }
        },

        "recommendation": {
            "task": "unsupervised",
            "time_series": False,
            "model": NearestNeighbors(
                n_neighbors=5,
                metric="cosine",
                n_jobs=DEFAULT_N_JOBS
            ),
            "params": {
                "n_neighbors": [3, 5, 10, 20],
                "metric": ["cosine", "euclidean"]
            }
        }
    }


def validate_model_registry(models):

    print("\n" + "=" * 60)
    print("MODEL REGISTRY VALIDATION")
    print("=" * 60)

    required_keys = ["task", "model", "params"]

    for name, info in models.items():

        missing = [key for key in required_keys if key not in info]

        if missing:
            print(f"{name}: Missing {missing}")
        else:
            print(f"{name}: READY")

    print("=" * 60)


def compute_scale_pos_weight(y):
    """
    Class-imbalance helper: ratio of negative to positive labels.
    Returns 1.0 if there are no positives (nothing to weight against).
    """

    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())

    if positives == 0:
        return 1.0

    return negatives / positives


# ============================================================
# MODEL TUNING FUNCTION
# ============================================================

def tune_model(
    model_name,
    X_train,
    y_train,
    models,
    search_type="random",
    iterations=SEARCH_ITERATIONS,
    cv_splits=CV_SPLITS
):

    print("\n" + "=" * 80)
    print(f"TUNING MODEL: {model_name}")
    print("=" * 80)

    if model_name not in models:
        raise ValueError(f"{model_name} not found in registry")

    config = models[model_name]

    task = config["task"]

    if task == "unsupervised":
        raise ValueError("Unsupervised models do not use the tuning wrapper")

    model = config["model"]
    params = config["params"]
    time_series = config.get("time_series", False)

    # Resource tracking
    process = psutil.Process()
    memory_start = process.memory_info().rss / 1024 ** 2
    cpu_start = time.process_time()
    start_time = time.time()

    print(f"Training rows: {X_train.shape[0]}")
    print(f"Features: {X_train.shape[1]}")

    if time_series:
        cv = TimeSeriesSplit(n_splits=cv_splits)
    elif task == "classification":
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    else:
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    scoring = "roc_auc" if task == "classification" else "neg_root_mean_squared_error"

    # PERFORMANCE FIX: the search itself stays single-threaded on
    # purpose. Each XGBoost/LightGBM model already parallelizes
    # internally via DEFAULT_N_JOBS; if the search ALSO ran with
    # n_jobs=DEFAULT_N_JOBS, every one of its parallel workers would
    # spawn its own full set of per-core boosting threads - CPU
    # oversubscription that's often slower than running serially, not
    # faster. Only one of {search, model} should ever parallelize.
    if search_type == "grid":
        search = GridSearchCV(
            estimator=model,
            param_grid=params,
            scoring=scoring,
            cv=cv,
            n_jobs=1,
            verbose=1
        )
    else:
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=iterations,
            scoring=scoring,
            cv=cv,
            n_jobs=1,
            random_state=RANDOM_STATE,
            verbose=1
        )

    search.fit(X_train, y_train)

    elapsed = time.time() - start_time
    cpu_used = time.process_time() - cpu_start
    memory_end = process.memory_info().rss / 1024 ** 2

    print("\nCOMPUTATION REPORT")
    print("-" * 50)
    print(f"Runtime: {elapsed:.2f} seconds")
    print(f"CPU time: {cpu_used:.2f} seconds")
    print(f"Memory used: {memory_end - memory_start:.2f} MB")

    print("\nBEST PARAMETERS")
    print(search.best_params_)

    print("\nBEST CV SCORE")
    print(search.best_score_)

    return {
        "model": search.best_estimator_,
        "parameters": search.best_params_,
        "score": search.best_score_,
        "runtime_seconds": elapsed,
        "memory_used_mb": memory_end - memory_start
    }


# ============================================================
# FINAL MODEL TRAINING + TEST EVALUATION
# ============================================================

def train_model(tuned_result, X_train, X_test, y_train, y_test, task):

    print("\n" + "=" * 80)
    print("FINAL MODEL TRAINING")
    print("=" * 80)

    start_time = time.time()

    model = tuned_result["model"]
    parameters = tuned_result["parameters"]

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    results = {}

    if task == "classification":

        accuracy = accuracy_score(y_test, predictions)
        results["accuracy"] = accuracy

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)[:, 1]
            results["roc_auc"] = roc_auc_score(y_test, probabilities)

    else:

        rmse = mean_squared_error(y_test, predictions) ** 0.5
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        results["rmse"] = rmse
        results["mae"] = mae
        results["r2"] = r2

    runtime = time.time() - start_time

    print("\nTEST RESULTS")
    print("-" * 50)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    print(f"\nRuntime: {runtime:.2f} seconds")

    return {
        "model": model,
        "parameters": parameters,
        "score": results,
        "runtime_seconds": runtime
    }


def train_recommendation_model(X_train, models, model_name="recommendation"):

    print("\n" + "=" * 80)
    print("TRAINING RECOMMENDATION MODEL")
    print("=" * 80)

    if model_name not in models:
        raise ValueError(f"{model_name} not found in registry")

    config = models[model_name]

    if config["task"] != "unsupervised":
        raise ValueError("This function is only for unsupervised models")

    process = psutil.Process()
    memory_start = process.memory_info().rss / 1024 ** 2
    start_time = time.time()

    model = NearestNeighbors(n_neighbors=5, metric="cosine", n_jobs=DEFAULT_N_JOBS)

    print("Training rows:", X_train.shape[0])
    print("Features:", X_train.shape[1])

    model.fit(X_train)

    runtime = time.time() - start_time
    memory_end = process.memory_info().rss / 1024 ** 2

    print("\nCOMPUTATION REPORT")
    print("-" * 50)
    print(f"Runtime: {runtime:.2f} seconds")
    print(f"Memory Used: {memory_end - memory_start:.2f} MB")

    return {
        "model": model,
        "parameters": {"n_neighbors": 5, "metric": "cosine"},
        "score": None,
        "runtime_seconds": runtime
    }


def create_model_package(model, parameters, score):
    """Bundles a trained model with its parameters and evaluation score."""

    return {
        "model": model,
        "parameters": parameters,
        "score": score
    }


# ============================================================
# MODEL MONITORING + SELF-ADJUSTING RETRAINING
# ============================================================

def monitor_and_retrain_model(
    model_name,
    current_package,
    X_train,
    X_test,
    y_train,
    y_test,
    models,
    performance_threshold=0.0,
    save_path=None
):
    """
    Evaluates the currently deployed model against a freshly-tuned
    candidate trained on the latest data, and only swaps the model
    if the candidate is actually better (>= performance_threshold
    improvement). Otherwise the current model is kept as-is.
    """

    print("\n" + "=" * 80)
    print("MODEL MONITORING + RETRAINING")
    print("=" * 80)

    task = models[model_name]["task"]
    current_model = current_package["model"]

    # --------------------------------------------------------
    # Evaluate current model
    # --------------------------------------------------------

    current_predictions = current_model.predict(X_test)

    if task == "classification":
        current_probabilities = current_model.predict_proba(X_test)[:, 1]
        current_score = roc_auc_score(y_test, current_probabilities)
    else:
        current_score = r2_score(y_test, current_predictions)

    print("Current model score:", current_score)

    # --------------------------------------------------------
    # Adjust for class imbalance using the latest training labels
    # --------------------------------------------------------

    if task == "classification":

        scale_pos_weight = compute_scale_pos_weight(y_train)
        print("Dynamic class weight (scale_pos_weight):", scale_pos_weight)

        model_object = models[model_name]["model"]

        if "scale_pos_weight" in model_object.get_params():
            model_object.set_params(scale_pos_weight=scale_pos_weight)

    # --------------------------------------------------------
    # Tune + train candidate model on the latest data
    # --------------------------------------------------------

    print("\nTraining candidate model...")

    tuned_result = tune_model(model_name, X_train, y_train, models)
    candidate_result = train_model(tuned_result, X_train, X_test, y_train, y_test, task)

    candidate_score = (
        candidate_result["score"]["roc_auc"] if task == "classification"
        else candidate_result["score"]["r2"]
    )

    print("Candidate score:", candidate_score)

    improvement = candidate_score - current_score
    print("Improvement:", improvement)

    if improvement >= performance_threshold:

        print("\nNew model accepted")

        package = create_model_package(
            candidate_result["model"],
            candidate_result["parameters"],
            candidate_result["score"]
        )

        if save_path:
            save_pickle(package, save_path)

        return package

    print("\nCurrent model kept")
    return current_package


def train_all_initial_models(split_datasets, processed_ml_datasets, models_registry,
                              save_dir=MODEL_SAVE_DIR):
    """
    Trains every supervised model once on split_datasets, plus the
    unsupervised recommendation model on the full interaction matrix.
    Saves each trained package to disk and returns a dict of them,
    keyed by name (churn, ltv, demand, forecast, recommendation).
    """

    trained_packages = {}

    for name, (X_train, X_test, y_train, y_test) in split_datasets.items():

        task = models_registry[name]["task"]

        if task == "classification":
            scale_pos_weight = compute_scale_pos_weight(y_train)
            print(f"[{name}] scale_pos_weight set to {scale_pos_weight:.3f} "
                  f"(positives={int((y_train == 1).sum())}, negatives={int((y_train == 0).sum())})")

            model_object = models_registry[name]["model"]
            if "scale_pos_weight" in model_object.get_params():
                model_object.set_params(scale_pos_weight=scale_pos_weight)

        tuned = tune_model(name, X_train, y_train, models_registry)
        result = train_model(tuned, X_train, X_test, y_train, y_test, task)

        package = create_model_package(result["model"], result["parameters"], result["score"])
        trained_packages[name] = package

        export_model_package(package, folder=save_dir, filename=f"{name}_model")

    recommendation_result = train_recommendation_model(
        processed_ml_datasets["recommendation"], models_registry
    )
    trained_packages["recommendation"] = create_model_package(
        recommendation_result["model"],
        recommendation_result["parameters"],
        recommendation_result["score"]
    )
    export_model_package(
        trained_packages["recommendation"], folder=save_dir, filename="recommendation_model"
    )

    return trained_packages


def ingest_new_data_and_retrain(
    new_raw_sales_df,
    context,
    max_rows=None,
    save_dir=None,
    export_row_usage=EXPORT_ROW_USAGE_MANIFEST,
    row_usage_folder=None,
):
    """
    Self-adjusting training entry point, rewritten to take an explicit
    `context` dict (as produced by run_initial_pipeline(), or by a
    previous call to this function) instead of module-level globals.

    Call this whenever new raw sales transactions arrive (same schema
    as fact_sales_denormalized). It:

      1. appends the new rows to the historical learning sample,
      2. rebuilds every engineered feature table and model-ready
         dataset from scratch,
      3. re-splits train/test data per target,
      4. asks monitor_and_retrain_model to decide - per model -
         whether a freshly tuned candidate actually beats the
         currently deployed model before replacing it,
      5. refreshes the (unsupervised) recommendation model directly,
         since it has no target/score to compare against,
      6. rebuilds and (optionally) re-exports the row-usage manifest.

    Models that don't improve are left untouched, so a batch of noisy
    or low-volume new data can never silently degrade production
    models - only genuine improvements get deployed.

    Returns a NEW context dict - store this back wherever you keep your
    application's pipeline state (do not keep using the old one).
    """

    max_rows = max_rows or context.get("sample_size", LEARNING_SAMPLE_SIZE)
    save_dir = save_dir or context.get("model_save_dir", MODEL_SAVE_DIR)
    row_usage_folder = row_usage_folder or ROW_USAGE_FOLDER

    print("\n" + "=" * 80)
    print("INGESTING NEW DATA AND RE-EVALUATING MODELS")
    print("=" * 80)

    new_clean = clean_dataframe(new_raw_sales_df)

    df_sales_denormalized = (
        pd.concat([context["df_sales_denormalized"], new_clean], ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print("Updated historical sales rows:", len(df_sales_denormalized))

    df_customers = context["df_customers"]
    df_products = context["df_products"]
    df_dates = context["df_dates"]
    models_registry = context["models"]
    trained_packages = dict(context["trained_packages"])

    ml_features = create_all_ml_feature_tables(df_sales_denormalized, df_customers, df_products)
    final_ml = create_final_ml_datasets(ml_features, df_customers, df_products, df_dates, df_sales_denormalized)
    final_ml_clean = clean_all_ml_training_datasets(final_ml)

    processed_ml_datasets, dataset_encoders = preprocess_all_datasets(final_ml_clean)
    split_datasets = split_all_datasets(processed_ml_datasets, sample_size=max_rows)

    for name, (X_train, X_test, y_train, y_test) in split_datasets.items():

        if name not in trained_packages:
            continue

        trained_packages[name] = monitor_and_retrain_model(
            model_name=name,
            current_package=trained_packages[name],
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            models=models_registry,
            save_path=os.path.join(save_dir, f"{name}_model.pkl")
        )

    # Recommendation model: unsupervised, so just refresh it on the
    # newest interaction matrix instead of comparing scores.
    recommendation_result = train_recommendation_model(
        processed_ml_datasets["recommendation"], models_registry
    )
    trained_packages["recommendation"] = create_model_package(
        recommendation_result["model"],
        recommendation_result["parameters"],
        recommendation_result["score"]
    )
    export_model_package(
        trained_packages["recommendation"], folder=save_dir, filename="recommendation_model"
    )

    star_schema_tables = dict(context["star_schema_tables"])
    star_schema_tables["fact_sales"] = df_sales_denormalized

    learning_sales, remaining_sales = split_learning_sales(df_sales_denormalized, sample_size=max_rows)
    row_usage_manifest = build_row_usage_manifest(learning_sales, remaining_sales, star_schema_tables)

    if export_row_usage:
        export_row_usage_manifest(row_usage_manifest, output_folder=row_usage_folder)

    new_context = dict(context)
    new_context.update({
        "df_sales_denormalized": df_sales_denormalized,
        "star_schema_tables": star_schema_tables,
        "ml_features": ml_features,
        "final_ml": final_ml,
        "final_ml_clean": final_ml_clean,
        "processed_ml_datasets": processed_ml_datasets,
        "dataset_encoders": dataset_encoders,
        "split_datasets": split_datasets,
        "trained_packages": trained_packages,
        "row_usage_manifest": row_usage_manifest,
        "learning_sales": learning_sales,
        "remaining_sales": remaining_sales,
        "sample_size": max_rows,
        "model_save_dir": save_dir,
    })

    return new_context


# ============================================================
# TOP-LEVEL ORCHESTRATOR - RUN THE WHOLE PIPELINE ONCE
# ============================================================

def run_initial_pipeline(
    dataset_slug=DATASET_SLUG,
    sample_size=LEARNING_SAMPLE_SIZE,
    model_save_dir=MODEL_SAVE_DIR,
    row_usage_folder=ROW_USAGE_FOLDER,
    cleanup_first=False,
    export_row_usage=EXPORT_ROW_USAGE_MANIFEST,
    data_dir=None,
    max_fact_rows=None,
    train_models=True,
):
    """
    Runs the whole pipeline end-to-end (download -> clean -> validate ->
    engineer features -> build final datasets -> preprocess -> split ->
    train every model -> build the row-usage manifest) and returns a
    single `context` dict holding every intermediate artifact.

    Keep this context around (in memory, in your app's session/state
    store, pickled to disk - whatever fits) and pass it straight into
    ingest_new_data_and_retrain() whenever new sales rows arrive.
    """

    if cleanup_first:
        cleanup_previous_run(dataset_slug)

    raw_datasets = download_and_load_datasets(dataset_slug, data_dir=data_dir, max_fact_rows=max_fact_rows)
    raw_frames = extract_star_schema_frames(raw_datasets)
    cleaned_frames = clean_all_star_schema_tables(raw_frames)
    star_schema_tables = build_star_schema_tables(cleaned_frames)

    validation_results = validate_star_schema(star_schema_tables)

    df_customers = cleaned_frames["dim_customers"]
    df_products = cleaned_frames["dim_products"]
    df_dates = cleaned_frames["dim_dates"]
    df_sales_denormalized = cleaned_frames["fact_sales_denormalized"]

    ml_features = create_all_ml_feature_tables(df_sales_denormalized, df_customers, df_products)

    print("\nFEATURE TABLES CREATED")
    for name, df in ml_features.items():
        print(name, df.shape)

    final_ml = create_final_ml_datasets(ml_features, df_customers, df_products, df_dates, df_sales_denormalized)

    print("\nFINAL ML DATASETS CREATED")
    for name, df in final_ml.items():
        print(name, df.shape)

    final_ml_clean = clean_all_ml_training_datasets(final_ml)

    processed_ml_datasets, dataset_encoders = preprocess_all_datasets(final_ml_clean)
    split_datasets = split_all_datasets(processed_ml_datasets, sample_size=sample_size)

    models_registry = {}
    trained_packages = {}
    if train_models:
        models_registry = build_model_registry()
        validate_model_registry(models_registry)
        trained_packages = train_all_initial_models(
            split_datasets, processed_ml_datasets, models_registry, save_dir=model_save_dir
        )
        X_train, X_test = fit_transform_train_test(X_train, X_test)

    learning_sales, remaining_sales = split_learning_sales(df_sales_denormalized, sample_size=sample_size)
    row_usage_manifest = build_row_usage_manifest(learning_sales, remaining_sales, star_schema_tables)

    if export_row_usage:
        export_row_usage_manifest(row_usage_manifest, output_folder=row_usage_folder)

    context = {
        "dataset_slug": dataset_slug,
        "cleaned_frames": cleaned_frames,
        "star_schema_tables": star_schema_tables,
        "star_schema_validation": validation_results,
        "df_customers": df_customers,
        "df_products": df_products,
        "df_dates": df_dates,
        "df_sales_denormalized": df_sales_denormalized,
        "ml_features": ml_features,
        "final_ml": final_ml,
        "final_ml_clean": final_ml_clean,
        "processed_ml_datasets": processed_ml_datasets,
        "dataset_encoders": dataset_encoders,
        "split_datasets": split_datasets,
        "models": models_registry,
        "trained_packages": trained_packages,
        "row_usage_manifest": row_usage_manifest,
        "learning_sales": learning_sales,
        "remaining_sales": remaining_sales,
        "sample_size": sample_size,
        "model_save_dir": model_save_dir,
    }

    print("\nPipeline complete.")

    return context


# ============================================================
# STANDALONE SCRIPT ENTRYPOINT
# ============================================================

if __name__ == "__main__":

    pipeline_context = run_initial_pipeline()

    if RUN_RAW_FEATURE_DIAGNOSTICS:
        run_raw_feature_diagnostics(pipeline_context["ml_features"])

    if EXPORT_EVENT_BANK:
        run_event_bank_pipeline(pipeline_context["df_sales_denormalized"])

    print("\n" + "=" * 80)
    print("INITIAL TRAINING COMPLETE")
    print("=" * 80)
    for name, package in pipeline_context["trained_packages"].items():
        print(f"{name:15s} -> score: {package['score']}")

    print(
        "\nTo self-adjust as new sales data comes in later, call:\n"
        "    pipeline_context = ingest_new_data_and_retrain(new_sales_df, pipeline_context)\n"
        "where new_sales_df has the same columns as fact_sales_denormalized. Each\n"
        "model is only replaced if the retrained candidate actually beats the\n"
        "currently deployed one on held-out data. Every call also refreshes the\n"
        f"row-usage manifest under '{ROW_USAGE_FOLDER}/' so you always know exactly\n"
        "which source rows (per table) have been consumed so far."
    )
