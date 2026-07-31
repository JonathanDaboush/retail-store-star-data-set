# ============================================================
# 14. MODEL TUNING FUNCTION
# ============================================================

def tune_model(model_name, X_train, y_train, models, search_type="random", iterations=10, cv_splits=3):

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
    scoring = config["scoring"]

    print(f"Training rows: {X_train.shape[0]} | Features: {X_train.shape[1]}")
    if task == "classification":
        print("Target distribution:")
        print(y_train.value_counts())

        if y_train.nunique() < 2:
            print(f"{model_name}: target has only one class - skipping tuning.")
            return None

    if time_series:
        cv = TimeSeriesSplit(n_splits=cv_splits)
    elif task == "classification" and y_train.nunique() == 2:
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    else:
        cv = cv_splits

    start_time = time.time()

    # search itself stays single-threaded on purpose - the underlying
    # models already parallelize via DEFAULT_N_JOBS, and nesting two
    # levels of parallelism is what actually kills laptop performance.
    if search_type == "grid":
        search = GridSearchCV(
            estimator=model, param_grid=params, scoring=scoring,
            cv=cv, n_jobs=1, error_score="raise", return_train_score=True
        )
    else:
        search = RandomizedSearchCV(
            estimator=model, param_distributions=params, n_iter=min(iterations, 10),
            scoring=scoring, cv=cv, random_state=RANDOM_STATE, n_jobs=1,
            error_score="raise", return_train_score=True
        )

    try:
        search.fit(X_train, y_train)
    except Exception as e:
        print("\nMODEL FAILED")
        print("Error type:", type(e).__name__)
        print("Reason:", e)
        return None

    elapsed = time.time() - start_time

    print(f"\nRuntime: {elapsed:.2f} seconds")
    print("Best score:", search.best_score_)
    print("Best parameters:", search.best_params_)

    return {
        "model": search.best_estimator_,
        "parameters": search.best_params_,
        "score": search.best_score_,
        "runtime_seconds": elapsed
    }


# ============================================================
# 15. FINAL MODEL TRAINING + TEST EVALUATION
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

        results["accuracy"] = accuracy_score(y_test, predictions)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)[:, 1]
            results["roc_auc"] = roc_auc_score(y_test, probabilities)

    else:

        results["rmse"] = mean_squared_error(y_test, predictions) ** 0.5
        results["mae"] = mean_absolute_error(y_test, predictions)
        results["r2"] = r2_score(y_test, predictions)

    runtime = time.time() - start_time

    print("\nTEST RESULTS")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    print(f"Runtime: {runtime:.2f} seconds")

    return {"model": model, "parameters": parameters, "score": results, "runtime_seconds": runtime}


def train_recommendation_model(matrix, models, model_name="product_recommendation"):

    print("\n" + "=" * 80)
    print("TRAINING RECOMMENDATION MODEL")
    print("=" * 80)

    if model_name not in models:
        raise ValueError(f"{model_name} not found in registry")

    if models[model_name]["task"] != "unsupervised":
        raise ValueError("This function is only for unsupervised models")

    start_time = time.time()

    model = NearestNeighbors(n_neighbors=5, metric="cosine", n_jobs=DEFAULT_N_JOBS)
    model.fit(matrix)

    runtime = time.time() - start_time
    print(f"Runtime: {runtime:.2f} seconds")

    return {
        "model": model,
        "parameters": {"n_neighbors": 5, "metric": "cosine"},
        "score": None,
        "runtime_seconds": runtime
    }


def create_model_package(model, parameters, score, preprocessing_artifact=None,
                          target_name=None, model_name=None):
    """Bundles a trained model with its parameters, score, and (optionally)
    the preprocessing artifact needed to transform new raw rows the same
    way at inference time."""

    package = {
        "model": model,
        "parameters": parameters,
        "score": score,
        "model_name": model_name,
        "target": target_name,
        "created": str(datetime.now()),
        "encoder": None,
        "scaler": None,
        "fill_values": None,
        "clipping_values": None,
        "dropped_columns": None,
        "feature_names": None,
    }

    if preprocessing_artifact is not None:
        package["encoder"] = preprocessing_artifact.get("encoder")
        package["scaler"] = preprocessing_artifact.get("scaler")
        package["fill_values"] = preprocessing_artifact.get("fill_values")
        package["clipping_values"] = preprocessing_artifact.get("clipping_values")
        package["dropped_columns"] = preprocessing_artifact.get("dropped_columns")
        package["feature_names"] = preprocessing_artifact["data"].columns.tolist()

    return package


# ============================================================
# 16. PICKLE SAVE / LOAD + EXPORT / IMPORT WRAPPERS
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


def export_model_package(package, folder, filename):

    os.makedirs(folder, exist_ok=True)
    if not filename.endswith(".pkl"):
        filename += ".pkl"

    filepath = os.path.join(folder, filename)
    save_pickle(package, filepath)
    return filepath


def import_model_package(folder, filename):

    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model package not found: {filepath}")

    return load_pickle(filepath)
