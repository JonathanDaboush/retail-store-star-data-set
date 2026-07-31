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
    final_ml = create_final_ml_datasets(ml_features, df_customers, df_products, df_dates)
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
