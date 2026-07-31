import plotly.express as px
import pandas as pd


def plot_raw_dataset_sizes(datasets):

    summary = pd.DataFrame({
        "table": datasets.keys(),
        "rows": [
            len(df)
            for df in datasets.values()
        ]
    })

    fig = px.bar(
        summary,
        x="rows",
        y="table",
        orientation="h",
        title="Raw Dataset Sizes"
    )

    return fig



def plot_target_distribution(df, target):

    counts = (
        df[target]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "class",
        "count"
    ]

    fig = px.bar(
        counts,
        x="class",
        y="count",
        title=f"Target Distribution: {target}"
    )

    return fig



def plot_model_metrics(results):

    metrics = pd.DataFrame({
        "metric": results.keys(),
        "value": results.values()
    })

    fig = px.bar(
        metrics,
        x="metric",
        y="value",
        title="Model Performance"
    )

    return fig



def plot_feature_importance(model, feature_names):

    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            "Model does not support feature importance"
        )

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
        .head(20)
    )

    fig = px.bar(
        importance,
        x="importance",
        y="feature",
        orientation="h",
        title="Top Feature Importance"
    )

    return fig