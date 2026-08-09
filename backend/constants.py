from pathlib import Path

# Root
ROOT_DIR = Path(__file__).resolve().parent.parent

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
    "churn": [],  # historical recency is valid when churn is labelled after the cutoff
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
