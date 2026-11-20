
import joblib
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import *


def build_pipeline(estimator):

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, CATEGORICAL)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])


def get_candidate_models():

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            solver="saga",
            tol=0.001,
            class_weight="balanced",
            random_state=SEED
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=180,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED
        ),

        "XGBoost": XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=SEED
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            verbosity=-1,
            n_jobs=-1,
            random_state=SEED
        )
    }


def calculate_metrics(y_true, probabilities):

    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities)
    }


def train_models(df):

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Group identical predictor profiles so they remain
    # together in the same validation fold.
    groups = pd.util.hash_pandas_object(
        X, index=False
    ).astype(str)

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    candidates = get_candidate_models()

    # Store one out-of-fold probability for every booking.
    oof_predictions = {
        name: pd.Series(index=X.index, dtype=float)
        for name in candidates
    }

    fold_results = []

    for fold_number, (train_indices, validation_indices) in enumerate(
        cv.split(X, y, groups), start=1
    ):

        print(f"\n========== FOLD {fold_number}/5 ==========")

        X_train = X.iloc[train_indices]
        X_validation = X.iloc[validation_indices]

        y_train = y.iloc[train_indices]
        y_validation = y.iloc[validation_indices]

        for name, estimator in candidates.items():

            print(f"Training {name}...")

            pipeline = build_pipeline(estimator)

            pipeline.fit(X_train, y_train)

            probabilities = pipeline.predict_proba(
                X_validation
            )[:, 1]

            oof_predictions[name].iloc[
                validation_indices
            ] = probabilities

            fold_metrics = calculate_metrics(
                y_validation,
                probabilities
            )

            fold_metrics["model"] = name
            fold_metrics["fold"] = fold_number

            fold_results.append(fold_metrics)

            print(
                f"{name} | "
                f"PR-AUC: {fold_metrics['pr_auc']:.4f}"
            )

    # Calculate overall out-of-fold performance.
    comparison_rows = []

    for name, probabilities in oof_predictions.items():

        if probabilities.isna().any():
            raise ValueError(
                f"Some rows have no out-of-fold predictions for {name}."
            )

        metrics = calculate_metrics(y, probabilities)

        comparison_rows.append({
            "model": name,
            **metrics
        })

    comparison = (
        pd.DataFrame(comparison_rows)
        .sort_values("pr_auc", ascending=False)
        .reset_index(drop=True)
    )

    # Save individual fold results for further analysis.
    fold_results_df = pd.DataFrame(fold_results)

    fold_results_df.to_csv(
        RESULTS / "cross_validation_fold_results.csv",
        index=False
    )

    comparison.to_csv(
        METRICS_FILE,
        index=False
    )

    # Save the out-of-fold predictions for every model.
    oof_output = pd.DataFrame({
        "actual": y
    })

    for name, probabilities in oof_predictions.items():

        safe_name = (
            name.lower()
            .replace(" ", "_")
        )

        oof_output[safe_name + "_probability"] = probabilities

    oof_output.to_csv(
        RESULTS / "oof_predictions.csv",
        index=False
    )

    # Select the model with the highest overall PR-AUC.
    best_model_name = comparison.iloc[0]["model"]

    print("\nSelected model:", best_model_name)
    print("Training final model on all available data...")

    final_model = build_pipeline(
        get_candidate_models()[best_model_name]
    )

    final_model.fit(X, y)

    # Save the final fitted model and its metadata.
    model_bundle = {
        "model": final_model,
        "model_name": best_model_name,
        "features": FEATURES,
        "threshold": 0.5,
        "metrics": comparison.iloc[0].to_dict(),
        "validation_method": "5-fold StratifiedGroupKFold",
        "selection_metric": "PR-AUC"
    }

    joblib.dump(model_bundle, MODEL_FILE)

    # Create confusion matrix from out-of-fold predictions.
    best_probabilities = oof_predictions[best_model_name]

    best_predictions = (
        best_probabilities >= 0.5
    ).astype(int)

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y,
        best_predictions,
        ax=ax,
        display_labels=["Not Cancelled", "Cancelled"],
        cmap="Blues"
    )

    ax.set_title(
        f"{best_model_name} - 5-Fold Out-of-Fold Results"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES / "models" / "confusion_matrix.png",
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nModel comparison:")
    print(comparison.to_string(index=False))

    print("\nFive-fold model validation completed.")

    return comparison
