from sklearn.metrics import r2_score, roc_auc_score

from backend.ml_functions.i_o_functionality import export_model_package, save_pickle
from backend.ml_functions.registry import compute_scale_pos_weight
from backend.ml_functions.star_schema_validation import clean_dataframe
from backend.ml_functions.training import create_model_package, train_model, train_recommendation_model, tune_model
import os
import pandas as pd
from constants import *

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


