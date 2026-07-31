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

