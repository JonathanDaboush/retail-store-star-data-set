# ============================================================
# STAR STORE MODEL REGISTRY
# ============================================================

def build_model_registry():
    """
    Factory that returns a FRESH model registry every time it's called.
    Kept as a factory (rather than a shared module-level dict) so that
    every pipeline run / retrain gets its own model instances instead of
    accidentally sharing (and mutating, e.g. via set_params) the same
    objects across runs or across multiple app contexts.
    """

    return {

        "forecast": {
            "task": "regression",
            "time_series": True,
            "model": XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=DEFAULT_N_JOBS
            ),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0]
            }
        },

        "churn": {
            "task": "classification",
            "time_series": False,
            "model": XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                scale_pos_weight=1,
                n_jobs=DEFAULT_N_JOBS
            ),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0]
            }
        },

        "ltv": {
            "task": "regression",
            "time_series": False,
            "model": LGBMRegressor(
                random_state=RANDOM_STATE,
                n_jobs=DEFAULT_N_JOBS
            ),
            "params": {
                "n_estimators": [100, 200, 300],
                "num_leaves": [15, 31, 63],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [-1, 5, 10],
                "min_child_samples": [10, 20, 50]
            }
        },

        "demand": {
            "task": "regression",
            "time_series": False,
            "model": XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=DEFAULT_N_JOBS
            ),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0]
            }
        },

        "recommendation": {
            "task": "unsupervised",
            "time_series": False,
            "model": NearestNeighbors(
                n_neighbors=5,
                metric="cosine",
                n_jobs=DEFAULT_N_JOBS
            ),
            "params": {
                "n_neighbors": [3, 5, 10, 20],
                "metric": ["cosine", "euclidean"]
            }
        }
    }


def validate_model_registry(models):

    print("\n" + "=" * 60)
    print("MODEL REGISTRY VALIDATION")
    print("=" * 60)

    required_keys = ["task", "model", "params"]

    for name, info in models.items():

        missing = [key for key in required_keys if key not in info]

        if missing:
            print(f"{name}: Missing {missing}")
        else:
            print(f"{name}: READY")

    print("=" * 60)


def compute_scale_pos_weight(y):
    """
    Class-imbalance helper: ratio of negative to positive labels.
    Returns 1.0 if there are no positives (nothing to weight against).
    """

    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())

    if positives == 0:
        return 1.0

    return negatives / positives


# ============================================================
# MODEL TUNING FUNCTION
# ============================================================

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
