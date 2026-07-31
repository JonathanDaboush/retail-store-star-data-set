# ============================================================
# 17. MODEL MONITORING + SELF-ADJUSTING RETRAINING
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
    candidate trained on the latest data, and only swaps the model if
    the candidate is actually better (>= performance_threshold
    improvement). Otherwise the current model is kept as-is.
    """

    print("\n" + "=" * 80)
    print(f"MODEL MONITORING + RETRAINING: {model_name}")
    print("=" * 80)

    task = models[model_name]["task"]
    current_model = current_package["model"]

    current_predictions = current_model.predict(X_test)

    if task == "classification":
        current_probabilities = current_model.predict_proba(X_test)[:, 1]
        current_score = roc_auc_score(y_test, current_probabilities)
    else:
        current_score = r2_score(y_test, current_predictions)

    print("Current model score:", current_score)

    if task == "classification":
        scale_pos_weight = compute_scale_pos_weight(y_train)
        print("Dynamic class weight (scale_pos_weight):", scale_pos_weight)

        model_object = models[model_name]["model"]
        if "scale_pos_weight" in model_object.get_params():
            model_object.set_params(scale_pos_weight=scale_pos_weight)

    print("\nTraining candidate model...")
    tuned_result = tune_model(model_name, X_train, y_train, models)

    if tuned_result is None:
        print("Candidate failed to tune - current model kept")
        return current_package

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
            candidate_result["model"], candidate_result["parameters"], candidate_result["score"],
            model_name=model_name, target_name=PRIMARY_TARGET.get(model_name)
        )
        if save_path:
            save_pickle(package, save_path)
        return package

    print("\nCurrent model kept")
    return current_package


# ============================================================
# 18. TRAIN / INGEST ORCHESTRATION
# ============================================================

def train_all_initial_models(split_datasets, models_registry, preprocess_artifacts,
                              recommendation_matrix, save_dir=MODEL_SAVE_DIR):
    """
    Trains every supervised model once on split_datasets, plus the
    unsupervised recommendation model (if there's any interaction data).
    Saves each trained package to disk and returns a dict of them.
    """

    os.makedirs(save_dir, exist_ok=True)
    trained_packages = {}

    for name, (X_train, X_test, y_train, y_test) in split_datasets.items():

        task = models_registry[name]["task"]

        if task == "classification":

            if y_train.nunique() < 2:
                print(f"{name} skipped: target has only one class "
                      f"(try a larger LEARNING_SAMPLE_SIZE or a longer future window).")
                continue

            scale_pos_weight = compute_scale_pos_weight(y_train)
            print(f"[{name}] scale_pos_weight set to {scale_pos_weight:.3f} "
                  f"(positives={int((y_train == 1).sum())}, negatives={int((y_train == 0).sum())})")

            model_object = models_registry[name]["model"]
            if "scale_pos_weight" in model_object.get_params():
                model_object.set_params(scale_pos_weight=scale_pos_weight)

        tuned = tune_model(name, X_train, y_train, models_registry)

        if tuned is None:
            print(f"{name} failed during tuning - no model saved.")
            continue

        result = train_model(tuned, X_train, X_test, y_train, y_test, task)

        package = create_model_package(
            result["model"], result["parameters"], result["score"],
            preprocessing_artifact=preprocess_artifacts.get(name),
            target_name=PRIMARY_TARGET[name], model_name=name
        )
        trained_packages[name] = package

        export_model_package(package, folder=save_dir, filename=name)

    if recommendation_matrix is not None:

        recommendation_result = train_recommendation_model(recommendation_matrix, models_registry)
        trained_packages["product_recommendation"] = create_model_package(
            recommendation_result["model"], recommendation_result["parameters"], recommendation_result["score"],
            model_name="product_recommendation"
        )
        export_model_package(
            trained_packages["product_recommendation"], folder=save_dir, filename="product_recommendation"
        )
    else:
        print("Recommendation skipped: no interaction data in the learning sample")

    return trained_packages


def ingest_new_data_and_retrain(
    new_raw_tables,
    context,
    future_orders=None,
    max_rows=None,
    save_dir=None,
    export_row_usage=EXPORT_ROW_USAGE_MANIFEST,
    row_usage_folder=None,
):
    """
    Self-adjusting training entry point, rewritten to take an explicit
    `context` dict (as produced by run_initial_pipeline(), or by a
    previous call to this function) instead of module-level globals.

    `new_raw_tables` is a dict that may contain any subset of the keys
    {"orders", "customers", "order_items", "products", "sellers",
    "payments", "reviews"} (same columns as the corresponding raw
    tables). It:

      1. appends the new rows to the matching historical raw table,
      2. samples a fresh learning sample from the combined pool
         (rebuild_related_tables keeps everything relationally
         consistent),
      3. rebuilds every engineered feature table, re-cleans,
         re-preprocesses, and re-splits it,
      4. asks monitor_and_retrain_model to decide - per model - whether
         a freshly tuned candidate actually beats the currently deployed
         model before replacing it,
      5. refreshes the (unsupervised) recommendation model directly,
         since it has no target/score to compare against,
      6. rebuilds and (optionally) re-exports the row-usage manifest.

    Models that don't improve are left untouched, so a batch of noisy or
    low-volume new data can never silently degrade production models -
    only genuine improvements get deployed.

    Returns a NEW context dict - store this back wherever you keep your
    application's pipeline state (do not keep using the old one).
    """

    max_rows = max_rows or context.get("sample_size", LEARNING_SAMPLE_SIZE)
    save_dir = save_dir or context.get("model_save_dir", MODEL_SAVE_DIR)
    row_usage_folder = row_usage_folder or ROW_USAGE_FOLDER

    print("\n" + "=" * 80)
    print("INGESTING NEW DATA AND RE-EVALUATING MODELS")
    print("=" * 80)

    tables = dict(context["tables"])
    trained_packages = dict(context["trained_packages"])
    models_registry = context["models"]

    for key, new_df in new_raw_tables.items():

        if key not in tables:
            print(f"Skipping unknown table key: {key}")
            continue

        new_clean = clean_dataframe(new_df)
        current = tables[key]
        updated = pd.concat([current, new_clean], ignore_index=True).drop_duplicates().reset_index(drop=True)
        tables[key] = updated
        print(f"Updated {key}: {len(current):,} -> {len(updated):,} rows")

    if future_orders is None:
        future_orders = context["future_source"]["orders"]

    all_orders = tables["orders"].sort_values("order_purchase_timestamp").reset_index(drop=True)
    n_learn = min(max_rows, len(all_orders))
    refreshed_learning_orders = all_orders.sample(n=n_learn, random_state=RANDOM_STATE)

    refreshed_learning_source = rebuild_related_tables(refreshed_learning_orders, tables)
    validate_star_schema(refreshed_learning_source, label="REFRESHED LEARNING SAMPLE")

    refreshed_ml_data = create_all_ml_feature_tables(refreshed_learning_source, future_orders)
    refreshed_ml_clean = clean_all_ml_training_datasets(refreshed_ml_data)
    refreshed_processed, refreshed_artifacts = preprocess_all_ml_datasets(refreshed_ml_clean)
    refreshed_splits = split_all_ml_datasets(refreshed_processed, max_rows=max_rows)

    for name, (X_train, X_test, y_train, y_test) in refreshed_splits.items():

        if name not in trained_packages:
            continue

        trained_packages[name] = monitor_and_retrain_model(
            model_name=name,
            current_package=trained_packages[name],
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            models=models_registry,
            save_path=os.path.join(save_dir, f"{name}.pkl")
        )

    refreshed_matrix = (
        build_recommendation_matrix(refreshed_ml_clean["product_recommendation"])
        if len(refreshed_ml_clean["product_recommendation"]) > 0 else None
    )

    if refreshed_matrix is not None:
        recommendation_result = train_recommendation_model(refreshed_matrix, models_registry)
        trained_packages["product_recommendation"] = create_model_package(
            recommendation_result["model"], recommendation_result["parameters"], recommendation_result["score"],
            model_name="product_recommendation"
        )
        export_model_package(
            trained_packages["product_recommendation"], folder=save_dir, filename="product_recommendation"
        )

    row_usage_manifest = build_row_usage_manifest(refreshed_learning_source, tables)
    if export_row_usage:
        export_row_usage_manifest(row_usage_manifest, output_folder=row_usage_folder)

    new_context = dict(context)
    new_context.update({
        "tables": tables,
        "learning_orders": refreshed_learning_orders,
        "learning_source": refreshed_learning_source,
        "row_usage_manifest": row_usage_manifest,
        "ml_data": refreshed_ml_data,
        "ml_data_clean": refreshed_ml_clean,
        "processed_ml_datasets": refreshed_processed,
        "preprocess_artifacts": refreshed_artifacts,
        "split_datasets": refreshed_splits,
        "recommendation_matrix": refreshed_matrix,
        "trained_packages": trained_packages,
        "sample_size": max_rows,
        "model_save_dir": save_dir,
    })

    return new_context
