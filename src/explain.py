
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold

from config import *
from src.train import build_pipeline, get_candidate_models


def explain_model():

    print("\nStarting model explainability...")

    # 1. Load the selected model information and cleaned dataset.
    bundle = joblib.load(MODEL_FILE)
    df = pd.read_csv(CLEAN_FILE)

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    model_name = bundle["model_name"]

    print("Selected model:", model_name)

    # 2. Create a group-aware training and validation split.
    groups = pd.util.hash_pandas_object(
        X, index=False
    ).astype(str)

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    train_indices, validation_indices = next(
        cv.split(X, y, groups)
    )

    X_train = X.iloc[train_indices]
    X_validation = X.iloc[validation_indices]

    y_train = y.iloc[train_indices]
    y_validation = y.iloc[validation_indices]

    print("Training explanation model...")

    # 3. Fit the selected model on the training partition.
    estimator = get_candidate_models()[model_name]

    explanation_model = build_pipeline(estimator)

    explanation_model.fit(X_train, y_train)

    # 4. Calculate permutation importance on held-out records.
    print("Calculating permutation importance...")

    permutation_result = permutation_importance(
        explanation_model,
        X_validation,
        y_validation,
        n_repeats=5,
        random_state=SEED,
        scoring="average_precision",
        n_jobs=1
    )

    permutation_table = pd.DataFrame({
        "feature": FEATURES,
        "importance": permutation_result.importances_mean,
        "std": permutation_result.importances_std
    }).sort_values(
        "importance",
        ascending=False
    )

    permutation_table.to_csv(
        RESULTS / "permutation_importance.csv",
        index=False
    )

    # 5. Create permutation importance chart.
    top_features = (
        permutation_table.head(15)
        .sort_values("importance")
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(
        top_features["feature"],
        top_features["importance"],
        xerr=top_features["std"]
    )

    ax.set_title(
        "Permutation Feature Importance"
    )

    ax.set_xlabel(
        "Decrease in Average Precision"
    )

    ax.set_ylabel("Original booking feature")

    fig.tight_layout()

    fig.savefig(
        FIGURES / "explainability" / "permutation_importance.png",
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)

    # 6. Prepare a sample of held-out records for SHAP.
    print("Preparing SHAP explanations...")

    shap_sample = X_validation.sample(
        n=min(500, len(X_validation)),
        random_state=SEED
    )

    # Transform original data using the fitted preprocessing pipeline.
    preprocessor = explanation_model.named_steps["preprocessor"]

    transformed_data = preprocessor.transform(shap_sample)

    # Convert sparse matrices to dense arrays for SHAP.
    if hasattr(transformed_data, "toarray"):
        transformed_data = transformed_data.toarray()

    feature_names = preprocessor.get_feature_names_out()

    transformed_df = pd.DataFrame(
        transformed_data,
        columns=feature_names,
        index=shap_sample.index
    )

    # 7. Extract the fitted tree-based estimator.
    fitted_estimator = explanation_model.named_steps["model"]

    print("Calculating SHAP values...")

    explainer = shap.TreeExplainer(fitted_estimator)

    shap_values = explainer.shap_values(
        transformed_df,
        check_additivity=False
    )

    # Handle different SHAP output formats.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    # 8. Calculate mean absolute SHAP importance.
    shap_importance = np.abs(shap_values).mean(axis=0)

    shap_table = pd.DataFrame({
        "feature": feature_names,
        "mean_absolute_shap": shap_importance
    }).sort_values(
        "mean_absolute_shap",
        ascending=False
    )

    shap_table.to_csv(
        RESULTS / "shap_feature_importance.csv",
        index=False
    )

    # 9. Generate SHAP summary plot.
    print("Creating SHAP summary plot...")

    plt.figure(figsize=(11, 8))

    shap.summary_plot(
        shap_values,
        transformed_df,
        max_display=20,
        show=False
    )

    plt.title(
        f"SHAP Summary - {model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES / "explainability" / "shap_summary.png",
        dpi=160,
        bbox_inches="tight"
    )

    plt.close()

    # 10. Generate SHAP global importance chart.
    plt.figure(figsize=(10, 7))

    shap.summary_plot(
        shap_values,
        transformed_df,
        plot_type="bar",
        max_display=20,
        show=False
    )

    plt.title(
        f"SHAP Global Feature Importance - {model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES / "explainability" / "shap_importance.png",
        dpi=160,
        bbox_inches="tight"
    )

    plt.close()

    print("\nExplainability completed successfully.")

    print("\nTop 10 permutation features:")
    print(permutation_table.head(10).to_string(index=False))

    print("\nTop 10 SHAP features:")
    print(shap_table.head(10).to_string(index=False))

    return permutation_table, shap_table
