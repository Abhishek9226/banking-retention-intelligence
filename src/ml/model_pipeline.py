"""
=============================================================================
PHASE 7 — MACHINE LEARNING PIPELINE
src/ml/model_pipeline.py
=============================================================================
Production-grade churn prediction system with:
  - 5 models (LR, DT, RF, XGBoost, LightGBM)
  - SMOTE oversampling
  - Cross-validation
  - Hyperparameter tuning (RandomizedSearchCV)
  - Full evaluation metrics
  - Serialised model artefacts
=============================================================================
"""

import logging
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline      import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("XGBoost not installed; skipping XGBClassifier")

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logger.warning("LightGBM not installed; skipping LGBMClassifier")


# =============================================================================
# FEATURE SELECTION FOR ML
# =============================================================================

ML_FEATURES = [
    # Raw
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
    # Engineered
    "EngagementScore", "RelationshipStrengthIndex", "ProductUtilizationScore",
    "LoyaltyScore", "PremiumRiskScore", "FinancialCommitmentIndex",
    "ActivityIntensityScore", "ProductDepthIndex", "StabilityScore",
    "BalanceSalaryRatio", "AgeSegment", "ChurnRiskCategory",
    # Geography dummies (created in prep)
    "Geo_Germany", "Geo_Spain", "Gender_Male",
]

TARGET = "Exited"


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_ml_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    One-hot encode Geography / Gender and return feature matrix X and target y.

    Returns
    -------
    (X, y) ready for train/test split
    """
    df = df.copy()

    # One-hot encode Geography
    df["Geo_Germany"] = (df["Geography"].astype(str) == "Germany").astype(int)
    df["Geo_Spain"]   = (df["Geography"].astype(str) == "Spain").astype(int)
    df["Gender_Male"] = (df["Gender"].astype(str) == "Male").astype(int)

    available_features = [f for f in ML_FEATURES if f in df.columns]
    missing            = [f for f in ML_FEATURES if f not in df.columns]

    if missing:
        logger.warning(f"Missing features (will be skipped): {missing}")

    X = df[available_features].fillna(0)
    y = df[TARGET]

    logger.info(f"ML data prepared: {X.shape[0]:,} rows × {X.shape[1]} features")
    return X, y


# =============================================================================
# MODEL REGISTRY
# =============================================================================

def build_model_registry() -> Dict:
    """Return dict of model_name → sklearn estimator."""
    registry = {
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", class_weight="balanced",
            random_state=42,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=8, min_samples_split=20, min_samples_leaf=10,
            class_weight="balanced", random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=42,
        ),
    }

    if HAS_XGB:
        registry["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.80, colsample_bytree=0.80, scale_pos_weight=4,
            eval_metric="auc", random_state=42, verbosity=0,
        )

    if HAS_LGB:
        registry["LightGBM"] = LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=63, class_weight="balanced",
            n_jobs=-1, random_state=42, verbose=-1,
        )

    return registry


# =============================================================================
# TRAINING PIPELINE (with SMOTE)
# =============================================================================

def train_model_with_smote(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
) -> object:
    """
    Wrap a model in an imbalanced-learn Pipeline with SMOTE and StandardScaler,
    then fit on training data.

    Returns
    -------
    Fitted imblearn Pipeline
    """
    pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(sampling_strategy=0.75, random_state=42)),
        ("model",  model),
    ])

    pipeline.fit(X_train, y_train)
    logger.info(f"[{model_name}] Fitted on {len(X_train):,} samples (+ SMOTE)")
    return pipeline


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> Dict:
    """
    Compute full classification metrics for a fitted pipeline.

    Returns
    -------
    dict with accuracy, precision, recall, f1, roc_auc, confusion_matrix
    """
    y_pred     = pipeline.predict(X_test)
    y_prob     = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name":       model_name,
        "accuracy":         round(accuracy_score(y_test, y_pred), 4),
        "precision":        round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":           round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":         round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":          round(roc_auc_score(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    logger.info(
        f"[{model_name}] Acc={metrics['accuracy']:.3f} | "
        f"F1={metrics['f1_score']:.3f} | AUC={metrics['roc_auc']:.3f}"
    )
    return metrics


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    cv: int = 5,
) -> Dict:
    """
    Stratified k-fold cross-validation (no SMOTE, uses class_weight).

    Returns
    -------
    dict with mean and std of AUC and F1
    """
    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    auc_scores = cross_val_score(model, X_sc, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    f1_scores  = cross_val_score(model, X_sc, y, cv=skf, scoring="f1",       n_jobs=-1)

    cv_result = {
        "model_name":  model_name,
        "cv_auc_mean": round(auc_scores.mean(), 4),
        "cv_auc_std":  round(auc_scores.std(),  4),
        "cv_f1_mean":  round(f1_scores.mean(),  4),
        "cv_f1_std":   round(f1_scores.std(),   4),
    }

    logger.info(
        f"[{model_name}] CV AUC={cv_result['cv_auc_mean']:.3f}±{cv_result['cv_auc_std']:.3f}"
    )
    return cv_result


# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================

PARAM_GRIDS = {
    "RandomForest": {
        "model__n_estimators":   [200, 300, 400],
        "model__max_depth":      [8, 12, None],
        "model__min_samples_leaf": [5, 10],
    },
    "XGBoost": {
        "model__n_estimators":  [200, 300],
        "model__max_depth":     [4, 6],
        "model__learning_rate": [0.03, 0.05, 0.10],
    },
}


def tune_model(
    model_name: str,
    base_model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 10,
) -> object:
    """
    RandomizedSearchCV on the ImbPipeline for the specified model.

    Returns
    -------
    Best fitted pipeline
    """
    if model_name not in PARAM_GRIDS:
        logger.info(f"[{model_name}] No tuning grid defined; skipping tuning")
        return None

    pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(sampling_strategy=0.75, random_state=42)),
        ("model",  base_model),
    ])

    search = RandomizedSearchCV(
        pipeline, PARAM_GRIDS[model_name],
        n_iter=n_iter, cv=3, scoring="roc_auc",
        n_jobs=-1, random_state=42, verbose=0,
    )
    search.fit(X_train, y_train)

    logger.info(f"[{model_name}] Best CV AUC (tuned): {search.best_score_:.4f}")
    logger.info(f"[{model_name}] Best params: {search.best_params_}")
    return search.best_estimator_


# =============================================================================
# MASTER TRAINING FUNCTION
# =============================================================================

def run_full_ml_pipeline(
    df: pd.DataFrame,
    artifacts_dir: Optional[Path] = None,
    tune: bool = False,
) -> Tuple[Dict, pd.DataFrame, pd.Series, pd.Series]:
    """
    Execute the full ML pipeline:
    1. Data preparation
    2. Train/test split (80/20, stratified)
    3. Train all models with SMOTE
    4. Evaluate on holdout set
    5. Cross-validate
    6. (optional) hyperparameter tuning
    7. Save best model

    Returns
    -------
    (results_dict, X_test, y_test, best_model_probs)
    """
    X, y               = prepare_ml_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    model_registry     = build_model_registry()
    results            = {}
    fitted_models      = {}

    # ── Train & evaluate ──────────────────────────────────────────────────
    for name, model in model_registry.items():
        pipeline           = train_model_with_smote(model, X_train, y_train, name)
        metrics            = evaluate_model(pipeline, X_test, y_test, name)
        cv_metrics         = cross_validate_model(model, X, y, name)

        metrics.update(cv_metrics)
        results[name]      = metrics
        fitted_models[name] = pipeline

    # ── Optional tuning on top-2 models ──────────────────────────────────
    if tune:
        for model_name in ["RandomForest", "XGBoost"]:
            if model_name in model_registry:
                best = tune_model(model_name, model_registry[model_name], X_train, y_train)
                if best is not None:
                    metrics = evaluate_model(best, X_test, y_test, f"{model_name}_Tuned")
                    results[f"{model_name}_Tuned"] = metrics
                    fitted_models[f"{model_name}_Tuned"] = best

    # ── Identify best model by ROC-AUC ───────────────────────────────────
    best_name   = max(results, key=lambda k: results[k]["roc_auc"])
    best_model  = fitted_models[best_name]
    best_probs  = best_model.predict_proba(X_test)[:, 1]

    logger.info(f"\n{'='*55}")
    logger.info(f"  BEST MODEL : {best_name}")
    logger.info(f"  ROC-AUC    : {results[best_name]['roc_auc']:.4f}")
    logger.info(f"  F1-Score   : {results[best_name]['f1_score']:.4f}")
    logger.info(f"{'='*55}")

    # ── Save artefacts ───────────────────────────────────────────────────
    if artifacts_dir is not None:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        for name, pipeline in fitted_models.items():
            path = artifacts_dir / f"{name.lower().replace(' ', '_')}.pkl"
            with open(path, "wb") as f:
                pickle.dump(pipeline, f)

        feature_path = artifacts_dir / "feature_columns.pkl"
        with open(feature_path, "wb") as f:
            pickle.dump(list(X.columns), f)

        logger.info(f"Models saved to {artifacts_dir}")

    return results, X_test, y_test, best_probs, fitted_models, best_name


# =============================================================================
# RESULTS FORMATTER
# =============================================================================

def format_results_table(results: Dict) -> pd.DataFrame:
    """Convert results dict to a comparison DataFrame."""
    rows = []
    for model_name, metrics in results.items():
        rows.append({
            "Model":     model_name,
            "Accuracy":  metrics.get("accuracy", np.nan),
            "Precision": metrics.get("precision", np.nan),
            "Recall":    metrics.get("recall", np.nan),
            "F1-Score":  metrics.get("f1_score", np.nan),
            "ROC-AUC":   metrics.get("roc_auc", np.nan),
            "CV AUC μ":  metrics.get("cv_auc_mean", np.nan),
            "CV AUC σ":  metrics.get("cv_auc_std", np.nan),
        })
    return pd.DataFrame(rows).set_index("Model").sort_values("ROC-AUC", ascending=False)


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from config.config import FEAT_DATA_FILE, ARTIFACTS_DIR

    df   = pd.read_parquet(FEAT_DATA_FILE)
    results, X_test, y_test, best_probs, fitted_models, best_name = run_full_ml_pipeline(
        df, artifacts_dir=ARTIFACTS_DIR, tune=False
    )

    print("\n=== MODEL COMPARISON TABLE ===")
    print(format_results_table(results).round(4).to_string())
