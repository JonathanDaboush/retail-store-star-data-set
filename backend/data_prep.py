import pandas as pd

from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder,
    StandardScaler,
    MinMaxScaler
)
# ============================================================
# 1. RETRIEVE / FILTER SALES DATA
# ============================================================

def retrieve_store_sales_data(
    dim_fact_sales,
    store_id,
    sales_sk=None,
    sales_id=None,
    customer_sk=None,
    product_sk=None,
    salesperson_sk=None,
    campaign_sk=None,
    date_min=None,
    date_max=None
):

    # Convert list of dictionaries into dataframe

    sales_df = pd.DataFrame(dim_fact_sales)


    # Convert dates

    sales_df["sales_date"] = pd.to_datetime(
        sales_df["sales_date"]
    )


    filtered_sales = sales_df.copy()



    # Apply filters

    if store_id is not None:

        filtered_sales = filtered_sales[
            filtered_sales["store_sk"] == store_id
        ]


    if sales_sk is not None:

        filtered_sales = filtered_sales[
            filtered_sales["sales_sk"] == sales_sk
        ]


    if sales_id is not None:

        filtered_sales = filtered_sales[
            filtered_sales["sales_id"] == sales_id
        ]


    if customer_sk is not None:

        filtered_sales = filtered_sales[
            filtered_sales["customer_sk"] == customer_sk
        ]


    if product_sk is not None:

        filtered_sales = filtered_sales[
            filtered_sales["product_sk"] == product_sk
        ]


    if salesperson_sk is not None:

        filtered_sales = filtered_sales[
            filtered_sales["salesperson_sk"] == salesperson_sk
        ]


    if campaign_sk is not None:

        filtered_sales = filtered_sales[
            filtered_sales["campaign_sk"] == campaign_sk
        ]


    if date_min is not None:

        filtered_sales = filtered_sales[
            filtered_sales["sales_date"] >= date_min
        ]


    if date_max is not None:

        filtered_sales = filtered_sales[
            filtered_sales["sales_date"] <= date_max
        ]



    # Sort results

    filtered_sales = filtered_sales.sort_values(
        "sales_date"
    )


    # Reset index

    return filtered_sales.reset_index(
        drop=True
    )




# ============================================================
# 2. MERGE TABLES USING KEYS
# ============================================================

def merge_tables(
    df1,
    df2,
    df1_key,
    df2_key,
    cols
):

    # Keep only needed columns from second table

    df2_subset = df2[
        [df2_key] + cols
    ]


    # Left join

    merged_df = df1.merge(

        df2_subset,

        left_on=df1_key,

        right_on=df2_key,

        how="left"

    )


    # Remove duplicate key if names differ

    if df1_key != df2_key:

        merged_df = merged_df.drop(
            columns=[df2_key]
        )


    return merged_df




# ============================================================
# 3. PREPROCESS DATA
# ============================================================

def preprocess_data(
    df,
    date_columns=None,
    categorical_columns=None,
    numerical_columns=None,
    scaling="standard"
):

    df = df.copy()



    # -------------------------------
    # Date processing
    # -------------------------------

    if date_columns:

        for column in date_columns:

            df[column] = pd.to_datetime(
                df[column]
            )


            df[column + "_year"] = (
                df[column].dt.year
            )


            df[column + "_month"] = (
                df[column].dt.month
            )


            df[column + "_day"] = (
                df[column].dt.day
            )


            df[column + "_weekday"] = (
                df[column].dt.dayofweek
            )


            # Remove raw date

            df = df.drop(
                columns=[column]
            )



    # -------------------------------
    # Encode categorical columns
    # -------------------------------

    if categorical_columns:

        encoder = OneHotEncoder(

            handle_unknown="ignore",

            sparse_output=False

        )


        encoded = encoder.fit_transform(

            df[categorical_columns]

        )


        encoded_df = pd.DataFrame(

            encoded,

            columns=encoder.get_feature_names_out(
                categorical_columns
            ),

            index=df.index

        )


        df = df.drop(
            columns=categorical_columns
        )


        df = pd.concat(
            [
                df,
                encoded_df
            ],
            axis=1
        )



    # -------------------------------
    # Scale numerical columns
    # -------------------------------

    if numerical_columns:

        scaler = MinMaxScaler()
        if scaling == "standard":

            scaler = StandardScaler()


        elif scaling == "minmax":

            scaler = MinMaxScaler()


        df[numerical_columns] = scaler.fit_transform(

            df[numerical_columns]

        )


    return df