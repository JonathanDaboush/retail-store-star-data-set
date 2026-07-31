from constants import *
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor
from sklearn.neighbors import NearestNeighbors
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
