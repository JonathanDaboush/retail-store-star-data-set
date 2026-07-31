import time

import psutil
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, TimeSeriesSplit
from sklearn.neighbors import NearestNeighbors
from constants import *

def tune_model(
    model_name,
    X_train,
    y_train,
    models,
    search_type="random",
    iterations=SEARCH_ITERATIONS,
    cv_splits=CV_SPLITS
):

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

    # Resource tracking
    process = psutil.Process()
    memory_start = process.memory_info().rss / 1024 ** 2
    cpu_start = time.process_time()
    start_time = time.time()

    print(f"Training rows: {X_train.shape[0]}")
    print(f"Features: {X_train.shape[1]}")

    if time_series:
        cv = TimeSeriesSplit(n_splits=cv_splits)
    elif task == "classification":
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    else:
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    scoring = "roc_auc" if task == "classification" else "neg_root_mean_squared_error"

    # PERFORMANCE FIX: the search itself stays single-threaded on
    # purpose. Each XGBoost/LightGBM model already parallelizes
    # internally via DEFAULT_N_JOBS; if the search ALSO ran with
    # n_jobs=DEFAULT_N_JOBS, every one of its parallel workers would
    # spawn its own full set of per-core boosting threads - CPU
    # oversubscription that's often slower than running serially, not
    # faster. Only one of {search, model} should ever parallelize.
    if search_type == "grid":
        search = GridSearchCV(
            estimator=model,
            param_grid=params,
            scoring=scoring,
            cv=cv,
            n_jobs=1,
            verbose=1
        )
    else:
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=iterations,
            scoring=scoring,
            cv=cv,
            n_jobs=1,
            random_state=RANDOM_STATE,
            verbose=1
        )

    search.fit(X_train, y_train)

    elapsed = time.time() - start_time
    cpu_used = time.process_time() - cpu_start
    memory_end = process.memory_info().rss / 1024 ** 2

    print("\nCOMPUTATION REPORT")
    print("-" * 50)
    print(f"Runtime: {elapsed:.2f} seconds")
    print(f"CPU time: {cpu_used:.2f} seconds")
    print(f"Memory used: {memory_end - memory_start:.2f} MB")

    print("\nBEST PARAMETERS")
    print(search.best_params_)

    print("\nBEST CV SCORE")
    print(search.best_score_)

    return {
        "model": search.best_estimator_,
        "parameters": search.best_params_,
        "score": search.best_score_,
        "runtime_seconds": elapsed,
        "memory_used_mb": memory_end - memory_start
    }


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
