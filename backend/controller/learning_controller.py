from constants import *
# Keep the controller as a thin orchestration layer.  The previous version
# referenced pipeline functions without importing them, so every real ingest
# failed at the first feature-engineering call.
from pipelin import (
    create_all_ml_feature_tables, create_final_ml_datasets,
    clean_all_ml_training_datasets, preprocess_all_datasets,
    split_all_datasets, monitor_and_retrain_model, tune_model, train_model,
    create_model_package, train_recommendation_model, compute_scale_pos_weight,
    build_row_usage_manifest, export_row_usage_manifest, split_learning_sales,
    save_pickle,
)
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
import pandas as pd
import numpy as np
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
def ingest_update_and_dashboard(new_raw_sales_df, context, performance_threshold=0.0,
                                max_rows=None, save_dir=None, export_row_usage=False):
    """
    Ingest new raw sales data, auto-adjust every model, and export a fresh
    .pkl file only when a model is actually updated or initially trained.

    Returns
    -------
    dict
        {
            "context": <updated pipeline context>,
            "dashboard": <metrics / health / row-usage payload>,
            "models": <current trained_packages dict>
        }
    """
    # ------------------------------------------------------------------
    # Defaults from context or module constants
    # ------------------------------------------------------------------
    max_rows = max_rows or context.get("sample_size", LEARNING_SAMPLE_SIZE)
    save_dir = save_dir or context.get("model_save_dir", MODEL_SAVE_DIR)

    # ------------------------------------------------------------------
    # 1. Ingest & deduplicate new sales rows
    # ------------------------------------------------------------------
    new_clean = clean_dataframe(new_raw_sales_df)

    df_sales_denormalized = (
        pd.concat([context["df_sales_denormalized"], new_clean], ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print("\n" + "=" * 80)
    print("INGEST & AUTO-UPDATE")
    print("=" * 80)
    print(f"Previous sales rows : {len(context['df_sales_denormalized']):,}")
    print(f"New rows added    : {len(new_clean):,}")
    print(f"Total sales rows  : {len(df_sales_denormalized):,}")

    # Pull refs from the incoming context
    df_customers    = context["df_customers"]
    df_products     = context["df_products"]
    df_dates        = context["df_dates"]
    models_registry = context["models"]
    trained_packages = dict(context.get("trained_packages", {}))

    # ------------------------------------------------------------------
    # 2. Rebuild features, datasets, and train/test splits
    # ------------------------------------------------------------------
    ml_features = create_all_ml_feature_tables(df_sales_denormalized, df_customers, df_products)
    final_ml    = create_final_ml_datasets(ml_features, df_customers, df_products, df_dates, df_sales_denormalized)
    final_ml_clean = clean_all_ml_training_datasets(final_ml)
    processed_ml_datasets, dataset_encoders = preprocess_all_datasets(final_ml_clean)
    split_datasets = split_all_datasets(processed_ml_datasets, sample_size=max_rows)

    # ------------------------------------------------------------------
    # 3. Dashboard skeleton
    # ------------------------------------------------------------------
    dashboard = {
        "ingestion": {
            "previous_sales_rows": len(context["df_sales_denormalized"]),
            "new_rows_added": len(new_clean),
            "total_sales_rows": len(df_sales_denormalized),
            "sample_size": max_rows,
        },
        "datasets": {},
        "models": {},
        "row_usage": {},
    }

    for name, df in processed_ml_datasets.items():
        dashboard["datasets"][name] = {"shape": df.shape}

    # ------------------------------------------------------------------
    # 4. Supervised models: tune / challenge / export .pkl on improvement
    # ------------------------------------------------------------------
    for name, (X_train, X_test, y_train, y_test) in split_datasets.items():
        task      = models_registry[name]["task"]
        model_obj = models_registry[name]["model"]

        # Class-imbalance guard (same logic as your original pipeline)
        if task == "classification":
            spw = compute_scale_pos_weight(y_train)
            if "scale_pos_weight" in model_obj.get_params():
                model_obj.set_params(scale_pos_weight=spw)

        save_path = os.path.join(save_dir, f"{name}_model.pkl") if save_dir else None

        # --- existing model: challenge with fresh candidate ----------------
        if name in trained_packages:
            current_pkg = trained_packages[name]

            new_pkg = monitor_and_retrain_model(
                model_name=name,
                current_package=current_pkg,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                models=models_registry,
                performance_threshold=performance_threshold,
                save_path=save_path,          # <-- .pkl written ONLY if accepted
            )

            status = "updated" if new_pkg is not current_pkg else "kept"
            trained_packages[name] = new_pkg

        # --- first time seeing this target --------------------------------
        else:
            tuned  = tune_model(name, X_train, y_train, models_registry)
            result = train_model(tuned, X_train, X_test, y_train, y_test, task)

            trained_packages[name] = create_model_package(
                result["model"], result["parameters"], result["score"]
            )

            if save_path:
                save_pickle(trained_packages[name], save_path)

            status = "initial_training"

        # Evaluate the final package (whether kept or new)
        final_mdl = trained_packages[name]["model"]
        preds     = final_mdl.predict(X_test)

        perf = {"status": status, "parameters": trained_packages[name]["parameters"]}

        if task == "classification":
            perf["accuracy"] = float(accuracy_score(y_test, preds))
            if hasattr(final_mdl, "predict_proba"):
                probs = final_mdl.predict_proba(X_test)[:, 1]
                perf["roc_auc"] = float(roc_auc_score(y_test, probs))
        else:
            perf["rmse"] = float(mean_squared_error(y_test, preds) ** 0.5)
            perf["mae"]  = float(mean_absolute_error(y_test, preds))
            perf["r2"]   = float(r2_score(y_test, preds))

        # Feature importances when available
        if hasattr(final_mdl, "feature_importances_"):
            cols = list(X_train.columns) if hasattr(X_train, "columns") else \
                   [f"f{i}" for i in range(X_train.shape[1])]
            perf["feature_importances"] = dict(
                zip(cols, final_mdl.feature_importances_.tolist())
            )

        dashboard["models"][name] = perf

    # ------------------------------------------------------------------
    # 5. Recommendation: always refresh & always export .pkl
    # ------------------------------------------------------------------
    rec_result = train_recommendation_model(
        processed_ml_datasets["recommendation"], models_registry
    )
    trained_packages["recommendation"] = create_model_package(
        rec_result["model"], rec_result["parameters"], rec_result["score"]
    )

    rec_save_path = os.path.join(save_dir, "recommendation_model.pkl") if save_dir else None
    if rec_save_path:
        save_pickle(trained_packages["recommendation"], rec_save_path)

    dashboard["models"]["recommendation"] = {
        "status": "refreshed",
        "parameters": rec_result["parameters"],
    }

    # ------------------------------------------------------------------
    # 6. Row-usage manifest (in-memory; CSV export only if requested)
    # ------------------------------------------------------------------
    star_schema_tables = dict(context.get("star_schema_tables", {}))
    star_schema_tables["fact_sales"] = df_sales_denormalized

    learning_sales, remaining_sales = split_learning_sales(df_sales_denormalized, sample_size=max_rows)
    row_usage_manifest = build_row_usage_manifest(learning_sales, remaining_sales, star_schema_tables)

    dashboard["row_usage"] = {
        table: {
            "pk_column": info["pk_column"],
            "used_count": info["used_count"],
            "unused_count": info["unused_count"],
            "total": info["used_count"] + info["unused_count"],
        }
        for table, info in row_usage_manifest.items()
    }

    if export_row_usage:
        export_row_usage_manifest(row_usage_manifest)

    # ------------------------------------------------------------------
    # 7. Assemble updated context
    # ------------------------------------------------------------------
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

    return {
        "context": new_context,
        "dashboard": dashboard,
        "models": trained_packages,
    }
