"""
=============================================================================
PROJECT CONFIGURATION — Banking Customer Retention Intelligence Platform
=============================================================================
Centralised configuration for paths, model hyperparameters, feature weights,
KPI thresholds, and business rules. Edit this file to tune the system without
touching any source module.
=============================================================================
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# ROOT PATHS
# ---------------------------------------------------------------------------
ROOT_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = ROOT_DIR / "data"
RAW_DATA_DIR    = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
MODELS_DIR      = ROOT_DIR / "models" / "trained"
ARTIFACTS_DIR   = ROOT_DIR / "models" / "artifacts"
REPORTS_DIR     = ROOT_DIR / "reports"
FIGURES_DIR     = REPORTS_DIR / "figures"
EXECUTIVE_DIR   = REPORTS_DIR / "executive"

# ---------------------------------------------------------------------------
# DATA FILES
# ---------------------------------------------------------------------------
RAW_DATA_FILE   = RAW_DATA_DIR / "churn_data.csv"
CLEAN_DATA_FILE = PROCESSED_DIR / "churn_clean.parquet"
FEAT_DATA_FILE  = PROCESSED_DIR / "churn_features.parquet"
SEG_DATA_FILE   = PROCESSED_DIR / "churn_segmented.parquet"

# ---------------------------------------------------------------------------
# TARGET & ID COLUMNS
# ---------------------------------------------------------------------------
TARGET_COL      = "Exited"
ID_COL          = "CustomerId"
DROP_COLS       = ["CustomerId", "Surname"]

# ---------------------------------------------------------------------------
# FEATURE GROUPS
# ---------------------------------------------------------------------------
NUMERIC_COLS    = ["CreditScore", "Age", "Tenure", "Balance",
                   "NumOfProducts", "EstimatedSalary"]
CATEGORICAL_COLS = ["Geography", "Gender"]
BINARY_COLS     = ["HasCrCard", "IsActiveMember"]

# ---------------------------------------------------------------------------
# FEATURE ENGINEERING WEIGHTS  (must sum to 1.0 within each index)
# ---------------------------------------------------------------------------
ENGAGEMENT_WEIGHTS = {
    "IsActiveMember":  0.40,
    "NumOfProducts":   0.30,
    "HasCrCard":       0.15,
    "Tenure":          0.15,
}

RELATIONSHIP_WEIGHTS = {
    "Tenure":          0.35,
    "NumOfProducts":   0.30,
    "Balance":         0.20,
    "CreditScore":     0.15,
}

LOYALTY_WEIGHTS = {
    "Tenure":          0.40,
    "IsActiveMember":  0.35,
    "NumOfProducts":   0.25,
}

FINANCIAL_COMMITMENT_WEIGHTS = {
    "Balance":         0.50,
    "EstimatedSalary": 0.30,
    "NumOfProducts":   0.20,
}

# ---------------------------------------------------------------------------
# KPI THRESHOLDS
# ---------------------------------------------------------------------------
HIGH_VALUE_BALANCE_PERCENTILE  = 75   # top-quartile balance = high-value
PREMIUM_CREDIT_SCORE_THRESHOLD = 700
HIGH_RISK_CHURN_PROBABILITY    = 0.65
ENGAGEMENT_ALERT_THRESHOLD     = 0.35  # below = disengaged
LOYALTY_ALERT_THRESHOLD        = 0.40

# ---------------------------------------------------------------------------
# CUSTOMER SEGMENT LABELS
# ---------------------------------------------------------------------------
SEGMENT_LABELS = {
    0: "Loyal Active Customers",
    1: "Silent Churn Risk",
    2: "Premium Disengaged",
    3: "Low-Value High-Risk",
    4: "Multi-Product Loyalists",
    5: "Young Digital Customers",
}

# ---------------------------------------------------------------------------
# MACHINE LEARNING
# ---------------------------------------------------------------------------
RANDOM_STATE        = 42
TEST_SIZE           = 0.20
CV_FOLDS            = 5
SMOTE_SAMPLING      = 0.75   # minority:majority after SMOTE

MODEL_PARAMS = {
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
        "class_weight": "balanced",
    },
    "decision_tree": {
        "max_depth": 8,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "class_weight": "balanced",
    },
    "random_forest": {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_split": 15,
        "min_samples_leaf": 5,
        "class_weight": "balanced",
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "scale_pos_weight": 4,   # handles imbalance
        "eval_metric": "auc",
        "use_label_encoder": False,
    },
    "lightgbm": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "class_weight": "balanced",
        "n_jobs": -1,
        "verbose": -1,
    },
}

HYPERPARAMETER_GRID = {
    "random_forest": {
        "model__n_estimators": [200, 300],
        "model__max_depth": [8, 12, None],
        "model__min_samples_leaf": [5, 10],
    },
    "xgboost": {
        "model__n_estimators": [200, 300],
        "model__max_depth": [4, 6],
        "model__learning_rate": [0.05, 0.10],
    },
}

# ---------------------------------------------------------------------------
# STREAMLIT DASHBOARD
# ---------------------------------------------------------------------------
DASHBOARD_TITLE     = "BankRetain IQ — Customer Retention Intelligence Platform"
DASHBOARD_ICON      = "🏦"
PRIMARY_COLOR       = "#0A2342"
ACCENT_COLOR        = "#D4AF37"
DANGER_COLOR        = "#E63946"
SUCCESS_COLOR       = "#2EC4B6"
WARNING_COLOR       = "#FF9F1C"
NEUTRAL_COLOR       = "#6C757D"
BACKGROUND_COLOR    = "#F8F9FA"

# ---------------------------------------------------------------------------
# RETENTION BUSINESS RULES
# ---------------------------------------------------------------------------
RETENTION_RULES = {
    "premium_disengaged": {
        "condition": "balance_high AND NOT active",
        "action": "Assign dedicated relationship manager + premium loyalty program",
        "urgency": "CRITICAL",
    },
    "multi_product_risk": {
        "condition": "num_products >= 3 AND churn_prob > 0.5",
        "action": "Personalised product retention bundle + fee waiver",
        "urgency": "HIGH",
    },
    "young_at_risk": {
        "condition": "age < 35 AND tenure < 2 AND NOT active",
        "action": "Digital engagement campaign + mobile app incentives",
        "urgency": "MEDIUM",
    },
    "credit_inactive": {
        "condition": "has_card AND NOT active",
        "action": "Credit card rewards activation + cashback offer",
        "urgency": "MEDIUM",
    },
    "low_product_depth": {
        "condition": "num_products == 1 AND tenure > 3",
        "action": "Cross-sell savings / loan product + relationship banker outreach",
        "urgency": "LOW",
    },
}
