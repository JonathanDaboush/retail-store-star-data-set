import numpy as np
import pandas as pd
from constants import *
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

            missing_count = (~child_df[fk_column].isin(parent_df[parent_column])).sum()

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