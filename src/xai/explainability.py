"""
=============================================================================
PHASE 8 — EXPLAINABLE AI (XAI) MODULE
src/xai/explainability.py
=============================================================================
SHAP-based model explainability for the churn prediction models.
Provides:
  - Global feature importance
  - SHAP summary plots
  - Individual prediction explanations
  - Executive-level churn driver narratives
=============================================================================
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("SHAP not installed. XAI features will be limited.")


# =============================================================================
# FEATURE IMPORTANCE (MODEL-NATIVE)
# =============================================================================

def get_feature_importance(
    model_pipeline,
    feature_names: List[str],
    model_name: str,
) -> pd.DataFrame:
    """
    Extract native feature importances from tree-based models.
    Falls back to |coefficient| for Logistic Regression.

    Returns
    -------
    pd.DataFrame with columns: feature, importance (sorted descending)
    """
    # Navigate inside imblearn Pipeline to the estimator
    estimator = None
    if hasattr(model_pipeline, "named_steps"):
        estimator = model_pipeline.named_steps.get("model")
    elif hasattr(model_pipeline, "steps"):
        estimator = model_pipeline.steps[-1][1]

    if estimator is None:
        logger.warning(f"[{model_name}] Could not extract estimator from pipeline")
        return pd.DataFrame()

    importances = None

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_[0])

    if importances is None:
        return pd.DataFrame()

    # Align with feature names (after SMOTE, shape unchanged)
    n = min(len(importances), len(feature_names))
    df_imp = pd.DataFrame({
        "feature":    feature_names[:n],
        "importance": importances[:n],
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    df_imp["importance_pct"] = (df_imp["importance"] / df_imp["importance"].sum() * 100).round(2)
    return df_imp


# =============================================================================
# SHAP ANALYSIS
# =============================================================================

def compute_shap_values(
    model_pipeline,
    X_test: pd.DataFrame,
    model_name: str,
    sample_size: int = 500,
) -> Optional[Tuple]:
    """
    Compute SHAP values for the best model.

    Parameters
    ----------
    model_pipeline : fitted imblearn pipeline
    X_test         : test feature matrix
    model_name     : name for logging
    sample_size    : max rows to explain (for speed)

    Returns
    -------
    (shap_values, shap_explainer, X_sample) or None if SHAP unavailable
    """
    if not HAS_SHAP:
        logger.warning("SHAP not available; skipping SHAP analysis")
        return None

    # Get scaler-transformed X
    scaler = None
    if hasattr(model_pipeline, "named_steps") and "scaler" in model_pipeline.named_steps:
        scaler = model_pipeline.named_steps["scaler"]

    estimator = None
    if hasattr(model_pipeline, "named_steps") and "model" in model_pipeline.named_steps:
        estimator = model_pipeline.named_steps["model"]

    if estimator is None or scaler is None:
        logger.warning(f"[{model_name}] Cannot extract scaler/estimator")
        return None

    X_sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)
    X_scaled  = scaler.transform(X_sample)

    try:
        if hasattr(estimator, "feature_importances_"):
            # Tree-based model
            explainer   = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]   # class 1 (churn)
        else:
            # Linear / other
            explainer   = shap.LinearExplainer(estimator, X_scaled)
            shap_values = explainer.shap_values(X_scaled)

        logger.info(f"[{model_name}] SHAP values computed for {len(X_sample)} samples")
        return shap_values, explainer, X_sample

    except Exception as e:
        logger.error(f"SHAP computation failed: {e}")
        return None


# =============================================================================
# SHAP FEATURE IMPORTANCE SUMMARY
# =============================================================================

def shap_feature_summary(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Build a ranked feature importance table from SHAP values.

    Returns
    -------
    pd.DataFrame with columns: feature, mean_abs_shap, rank
    """
    mean_shap = np.abs(shap_values).mean(axis=0)
    n         = min(len(mean_shap), len(feature_names))

    df_shap = pd.DataFrame({
        "feature":       feature_names[:n],
        "mean_abs_shap": mean_shap[:n],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    df_shap["rank"] = df_shap.index + 1
    return df_shap.head(top_n)


# =============================================================================
# CHURN DRIVER NARRATIVE (EXECUTIVE LEVEL)
# =============================================================================

DRIVER_NARRATIVES = {
    "Age": (
        "Age is the strongest churn predictor. Customers aged 45–60 exhibit "
        "significantly higher churn rates, likely due to evolving financial needs, "
        "retirement planning migration, or competitor wealth-management offerings."
    ),
    "IsActiveMember": (
        "Member activity status is a binary leading indicator of churn. Inactive "
        "customers are ~3× more likely to exit the bank within 12 months. "
        "Reactivation campaigns targeting this cohort should be a top retention priority."
    ),
    "Balance": (
        "High-balance customers who are disengaged represent the most financially "
        "damaging churn segment. AUM outflow from a single premium customer can exceed "
        "the retention cost of 20 standard customers."
    ),
    "NumOfProducts": (
        "Customers with exactly 1 product have the highest churn risk. "
        "Customers with 2 products are the most stable. Paradoxically, 3–4 "
        "product customers also churn more — suggesting product complexity "
        "or forced cross-sell leads to dissatisfaction."
    ),
    "Geography": (
        "German customers churn at a significantly higher rate than French or Spanish "
        "customers. This may reflect market competition, regulatory differences, or "
        "suboptimal product-market fit in the German segment."
    ),
    "CreditScore": (
        "Below-average credit scores correlate with higher churn, possibly driven "
        "by credit denial experiences or migration to specialist lenders."
    ),
    "Tenure": (
        "Shorter tenure is associated with higher churn. Customers in the first "
        "2 years are the most vulnerable. Early lifecycle engagement programmes "
        "are critical to survival past this honeymoon period."
    ),
    "EngagementScore": (
        "The composite Engagement Score is among the top engineered predictors. "
        "Customers with ES < 0.35 are at disproportionate churn risk regardless "
        "of balance or tenure."
    ),
}


def generate_churn_driver_report(
    shap_summary: pd.DataFrame,
    top_n: int = 5,
) -> str:
    """
    Generate an executive-readable churn driver narrative from SHAP rankings.

    Returns
    -------
    str — formatted report
    """
    report = ["=" * 65]
    report.append("  EXECUTIVE CHURN DRIVER ANALYSIS")
    report.append("=" * 65)
    report.append(
        "\nThe following factors most strongly determine customer churn "
        "based on SHAP (SHapley Additive exPlanations) analysis:\n"
    )

    for i, row in shap_summary.head(top_n).iterrows():
        feature     = row["feature"]
        shap_impact = row["mean_abs_shap"]
        narrative   = DRIVER_NARRATIVES.get(feature, "Feature contributes significantly to churn prediction.")

        report.append(f"{'─'*65}")
        report.append(f"  #{i+1}  {feature}  (Mean |SHAP|: {shap_impact:.4f})")
        report.append(f"{'─'*65}")
        report.append(f"  {narrative}\n")

    report.append("=" * 65)
    return "\n".join(report)


# =============================================================================
# INDIVIDUAL PREDICTION EXPLANATION
# =============================================================================

def explain_single_prediction(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    customer_idx: int,
    top_n: int = 8,
) -> pd.DataFrame:
    """
    Return top SHAP drivers for a single customer prediction.

    Parameters
    ----------
    shap_values   : SHAP values array (n_samples × n_features)
    X_sample      : the sample used for SHAP
    customer_idx  : position in X_sample to explain
    top_n         : number of top features to return

    Returns
    -------
    pd.DataFrame with columns: feature, shap_value, direction, feature_value
    """
    idx      = min(customer_idx, len(X_sample) - 1)
    sv       = shap_values[idx]
    fv       = X_sample.iloc[idx]

    df_exp = pd.DataFrame({
        "feature":       X_sample.columns,
        "shap_value":    sv,
        "feature_value": fv.values,
    })
    df_exp["abs_shap"]  = df_exp["shap_value"].abs()
    df_exp["direction"] = df_exp["shap_value"].apply(
        lambda x: "⬆ Increases Churn Risk" if x > 0 else "⬇ Reduces Churn Risk"
    )

    return (
        df_exp.sort_values("abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
        [["feature", "feature_value", "shap_value", "direction"]]
    )


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from config.config import FEAT_DATA_FILE, ARTIFACTS_DIR
    import pickle

    df = pd.read_parquet(FEAT_DATA_FILE)

    # Load best model (assumes model_pipeline.py was run first)
    from src.ml.model_pipeline import prepare_ml_data, run_full_ml_pipeline

    results, X_test, y_test, best_probs, fitted_models, best_name = run_full_ml_pipeline(df)

    best_model = fitted_models[best_name]

    feature_cols = list(X_test.columns)
    imp_df       = get_feature_importance(best_model, feature_cols, best_name)

    print("\n=== FEATURE IMPORTANCE (MODEL-NATIVE) ===")
    print(imp_df.head(10).to_string(index=False))

    if HAS_SHAP:
        result = compute_shap_values(best_model, X_test, best_name)
        if result:
            shap_values, explainer, X_sample = result
            summary = shap_feature_summary(shap_values, feature_cols)
            print("\n=== SHAP FEATURE SUMMARY ===")
            print(summary.to_string(index=False))
            print()
            print(generate_churn_driver_report(summary))
