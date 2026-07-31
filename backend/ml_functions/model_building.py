# ============================================================
# 12. RECOMMENDATION MATRIX (unsupervised - handled separately since
# it has no target and isn't part of PRIMARY_TARGET/split_datasets)
# ============================================================

def build_recommendation_matrix(recommendation_df):
    customer_codes = recommendation_df["customer_id"].astype("category").cat.codes
    product_codes = recommendation_df["product_id"].astype("category").cat.codes
    return csr_matrix((recommendation_df["purchase_count"], (customer_codes, product_codes)))


# ============================================================
# 13. MODEL REGISTRY
# ============================================================

def build_model_registry():
    """
    Factory that returns a FRESH model registry every time it's called.
    Kept as a factory (rather than a shared module-level dict) so every
    pipeline run / retrain gets its own model instances instead of
    accidentally sharing (and mutating, e.g. via set_params) the same
    objects across runs or across multiple app contexts.
    """

    return {

        "delivery_delay": {
            "task": "classification",
            "time_series": False,
            "scoring": "roc_auc",
            "model": LGBMClassifier(
                objective="binary", random_state=RANDOM_STATE, n_jobs=DEFAULT_N_JOBS, verbosity=-1
            ),
            "params": {
                "n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 6],
                "num_leaves": [20, 40], "min_child_samples": [20, 50], "scale_pos_weight": [1, 3]
            }
        },

        "order_cancellation": {
            "task": "classification",
            "time_series": False,
            "scoring": "average_precision",
            "model": RandomForestClassifier(
                random_state=RANDOM_STATE, n_jobs=DEFAULT_N_JOBS,
                max_samples=0.75, class_weight="balanced_subsample"
            ),
            "params": {
                "n_estimators": [100, 200], "max_depth": [5, 10], "min_samples_leaf": [2, 5],
                "max_features": ["sqrt"], "class_weight": ["balanced", "balanced_subsample"]
            }
        },

        "review_prediction": {
            "task": "regression",
            "time_series": False,
            "scoring": "neg_mean_absolute_error",
            "model": ElasticNet(
                alpha=0.1, selection="random", random_state=RANDOM_STATE, max_iter=5000, tol=0.001
            ),
            "params": {"alpha": [0.01, 0.1, 1], "l1_ratio": [0.2, 0.5, 0.8]}
        },

        "demand_forecasting": {
            "task": "regression",
            "time_series": True,
            "scoring": "neg_root_mean_squared_error",
            "model": XGBRegressor(
                objective="reg:squarederror", random_state=RANDOM_STATE,
                n_jobs=DEFAULT_N_JOBS, tree_method="hist"
            ),
            "params": {
                "n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 6],
                "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0]
            }
        },

        "customer_purchase_prediction": {
            "task": "classification",
            "time_series": False,
            "scoring": "roc_auc",
            "model": XGBClassifier(
                objective="binary:logistic", random_state=RANDOM_STATE,
                n_jobs=DEFAULT_N_JOBS, tree_method="hist", eval_metric="logloss"
            ),
            "params": {
                "n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 6],
                "scale_pos_weight": [1, 3]
            }
        },

        "product_recommendation": {
            "task": "unsupervised",
            "time_series": False,
            "scoring": None,
            "model": NearestNeighbors(n_neighbors=5, metric="cosine", n_jobs=DEFAULT_N_JOBS),
            "params": {"n_neighbors": [5, 10, 20], "metric": ["cosine"], "algorithm": ["auto", "brute"]}
        },
    }


def validate_model_registry(models):

    print("\n" + "=" * 60)
    print("MODEL REGISTRY VALIDATION")
    print("=" * 60)

    required_keys = ["task", "model", "params"]

    for name, info in models.items():
        missing = [key for key in required_keys if key not in info]
        print(f"{name}: {'Missing ' + str(missing) if missing else 'READY'}")


def compute_scale_pos_weight(y):
    """
    Class-imbalance helper: ratio of negative to positive labels.
    Returns 1.0 if there are no positives (nothing to weight against).
    """

    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())

    return negatives / positives if positives > 0 else 1.0