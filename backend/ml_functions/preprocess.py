from typing import Set
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from constants import *
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

    df, encoders = encode_categories(df)

    df = clip_numeric_columns(df, exclude=protected)
    df = scale_numeric_columns(df, exclude=protected)

    print("Final shape:", df.shape)
    print("Remaining object columns:", list(df.select_dtypes(include=["object", "category"]).columns))
    print("\nHead:")
    print(df.head())

    return df, encoders


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