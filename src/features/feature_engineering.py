"""
=============================================================================
PHASE 3 — ADVANCED FEATURE ENGINEERING
src/features/feature_engineering.py
=============================================================================
Constructs 10+ domain-specific behavioural indices that encode the
business intelligence required for high-accuracy churn prediction and
customer segmentation.

Each derived feature is:
  - Normalised to [0, 1] unless noted
  - Accompanied by a business rationale docstring
  - Computed via a transparent weighted formula
=============================================================================
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# Pre-instantiate a single scaler; features are independently normalised.
_scaler = MinMaxScaler()


# =============================================================================
# UTILITY HELPERS
# =============================================================================

def _norm(series: pd.Series) -> pd.Series:
    """Min-max normalise a pandas Series to [0, 1]."""
    s_min, s_max = series.min(), series.max()
    if s_max == s_min:
        return series * 0.0
    return (series - s_min) / (s_max - s_min)


# =============================================================================
# FEATURE 1 — ENGAGEMENT SCORE
# =============================================================================

def compute_engagement_score(df: pd.DataFrame) -> pd.Series:
    """
    Engagement Score (ES)
    ─────────────────────
    Measures how actively a customer interacts with the bank's ecosystem.

    Formula
    -------
    ES = 0.40 × IsActiveMember
       + 0.30 × norm(NumOfProducts / 4)
       + 0.15 × HasCrCard
       + 0.15 × norm(Tenure / 10)

    Business Meaning
    ----------------
    A high ES indicates a customer who is genuinely embedded in the bank's
    product ecosystem.  Low-ES customers are prime silent-churn candidates
    even if they hold a balance.
    """
    es = (
        0.40 * df["IsActiveMember"]
        + 0.30 * _norm(df["NumOfProducts"])
        + 0.15 * df["HasCrCard"]
        + 0.15 * _norm(df["Tenure"])
    )
    return es.rename("EngagementScore")


# =============================================================================
# FEATURE 2 — RELATIONSHIP STRENGTH INDEX
# =============================================================================

def compute_relationship_strength_index(df: pd.DataFrame) -> pd.Series:
    """
    Relationship Strength Index (RSI)
    ──────────────────────────────────
    Quantifies the depth and durability of the customer–bank relationship.

    Formula
    -------
    RSI = 0.35 × norm(Tenure)
        + 0.30 × norm(NumOfProducts)
        + 0.20 × norm(Balance)
        + 0.15 × norm(CreditScore)

    Business Meaning
    ----------------
    High RSI = long-tenured, multi-product customer with substantial assets.
    These customers are expensive to replace; early RSI decline signals risk.
    """
    rsi = (
        0.35 * _norm(df["Tenure"])
        + 0.30 * _norm(df["NumOfProducts"])
        + 0.20 * _norm(df["Balance"])
        + 0.15 * _norm(df["CreditScore"])
    )
    return rsi.rename("RelationshipStrengthIndex")


# =============================================================================
# FEATURE 3 — PRODUCT UTILIZATION SCORE
# =============================================================================

def compute_product_utilization_score(df: pd.DataFrame) -> pd.Series:
    """
    Product Utilization Score (PUS)
    ────────────────────────────────
    Captures breadth and depth of product adoption.

    Formula
    -------
    PUS = norm(NumOfProducts / 4) × (1 + 0.3 × HasCrCard) × IsActiveMember_adj

    Where IsActiveMember_adj = 0.7 + 0.3 × IsActiveMember  (non-zero floor)

    Business Meaning
    ----------------
    Customers with many products but low activity scores represent "dormant
    multi-bankers" — they hold products but don't use them, signalling
    imminent attrition.
    """
    active_adj = 0.70 + 0.30 * df["IsActiveMember"]
    pus = _norm(df["NumOfProducts"]) * (1 + 0.30 * df["HasCrCard"]) * active_adj
    return _norm(pus).rename("ProductUtilizationScore")


# =============================================================================
# FEATURE 4 — CUSTOMER LOYALTY SCORE
# =============================================================================

def compute_loyalty_score(df: pd.DataFrame) -> pd.Series:
    """
    Customer Loyalty Score (CLS)
    ─────────────────────────────
    Blend of tenure, activity, and product depth as a loyalty proxy.

    Formula
    -------
    CLS = 0.40 × norm(Tenure)
        + 0.35 × IsActiveMember
        + 0.25 × norm(NumOfProducts)

    Business Meaning
    ----------------
    High CLS customers are the backbone of the bank's revenue base.
    Loyalty interventions should be triggered when CLS drops below 0.40.
    """
    cls = (
        0.40 * _norm(df["Tenure"])
        + 0.35 * df["IsActiveMember"]
        + 0.25 * _norm(df["NumOfProducts"])
    )
    return cls.rename("LoyaltyScore")


# =============================================================================
# FEATURE 5 — PREMIUM RISK SCORE
# =============================================================================

def compute_premium_risk_score(df: pd.DataFrame) -> pd.Series:
    """
    Premium Risk Score (PRS)
    ────────────────────────
    Identifies high-value customers with elevated churn risk — the
    most costly segment for the bank to lose.

    Formula
    -------
    PRS = norm(Balance) × (1 - IsActiveMember) × norm(1 / (CreditScore + 1))

    Business Meaning
    ----------------
    A premium risk customer holds significant assets but is disengaged and
    may have credit concerns.  Losing one such customer can represent
    hundreds of thousands in AUM outflow.
    """
    inv_credit = _norm(1 / (df["CreditScore"] + 1))
    prs = _norm(df["Balance"]) * (1 - df["IsActiveMember"]) * inv_credit
    return _norm(prs).rename("PremiumRiskScore")


# =============================================================================
# FEATURE 6 — FINANCIAL COMMITMENT INDEX
# =============================================================================

def compute_financial_commitment_index(df: pd.DataFrame) -> pd.Series:
    """
    Financial Commitment Index (FCI)
    ─────────────────────────────────
    Measures the financial depth of a customer's engagement with the bank.

    Formula
    -------
    FCI = 0.50 × norm(Balance)
        + 0.30 × norm(EstimatedSalary)
        + 0.20 × norm(NumOfProducts)

    Business Meaning
    ----------------
    High FCI customers contribute disproportionately to NII (Net Interest
    Income) and fee revenues.  FCI should gate premium-tier segmentation.
    """
    fci = (
        0.50 * _norm(df["Balance"])
        + 0.30 * _norm(df["EstimatedSalary"])
        + 0.20 * _norm(df["NumOfProducts"])
    )
    return fci.rename("FinancialCommitmentIndex")


# =============================================================================
# FEATURE 7 — ACTIVITY INTENSITY SCORE
# =============================================================================

def compute_activity_intensity_score(df: pd.DataFrame) -> pd.Series:
    """
    Activity Intensity Score (AIS)
    ──────────────────────────────
    Composite measure of how intensely the customer uses the bank's services.

    Formula
    -------
    AIS = IsActiveMember × (0.5 × norm(NumOfProducts) + 0.5 × HasCrCard)
        × (1 + 0.2 × norm(Tenure))

    Business Meaning
    ----------------
    Distinguishes passive account holders from genuinely active customers.
    AIS near zero on an active account signals data quality or engagement drop.
    """
    intensity = (
        df["IsActiveMember"]
        * (0.50 * _norm(df["NumOfProducts"]) + 0.50 * df["HasCrCard"])
        * (1 + 0.20 * _norm(df["Tenure"]))
    )
    return _norm(intensity).rename("ActivityIntensityScore")


# =============================================================================
# FEATURE 8 — PRODUCT DEPTH INDEX
# =============================================================================

def compute_product_depth_index(df: pd.DataFrame) -> pd.Series:
    """
    Product Depth Index (PDI)
    ─────────────────────────
    Captures cross-sell penetration relative to tenure.

    Formula
    -------
    PDI = norm(NumOfProducts × (Tenure + 1)) × HasCrCard_adj

    Where HasCrCard_adj = 1 if HasCrCard else 0.6

    Business Meaning
    ----------------
    A high PDI customer has organically deepened their product portfolio
    over time — they are more expensive to acquire and more profitable to retain.
    """
    card_adj = np.where(df["HasCrCard"] == 1, 1.0, 0.6)
    pdi      = _norm(df["NumOfProducts"] * (df["Tenure"] + 1)) * card_adj
    return _norm(pdi).rename("ProductDepthIndex")


# =============================================================================
# FEATURE 9 — CUSTOMER STABILITY SCORE
# =============================================================================

def compute_stability_score(df: pd.DataFrame) -> pd.Series:
    """
    Customer Stability Score (CSS)
    ──────────────────────────────
    Measures the resilience and stability of a customer's profile.

    Formula
    -------
    CSS = 0.35 × norm(CreditScore)
        + 0.30 × norm(Tenure)
        + 0.20 × norm(Balance)
        + 0.15 × norm(EstimatedSalary)

    Business Meaning
    ----------------
    High CSS customers have strong financial health and a long banking
    history — they weather economic shocks and rarely churn voluntarily.
    """
    css = (
        0.35 * _norm(df["CreditScore"])
        + 0.30 * _norm(df["Tenure"])
        + 0.20 * _norm(df["Balance"])
        + 0.15 * _norm(df["EstimatedSalary"])
    )
    return css.rename("StabilityScore")


# =============================================================================
# FEATURE 10 — CHURN RISK CATEGORY (ordinal)
# =============================================================================

def compute_churn_risk_category(df: pd.DataFrame) -> pd.Series:
    """
    Churn Risk Category (CRC)
    ──────────────────────────
    Rule-based ordinal risk bucket derived from engagement and demographics.

    Levels: 0=Very Low, 1=Low, 2=Moderate, 3=High, 4=Critical

    Business Meaning
    ----------------
    Provides a human-readable risk label for CRM and campaign targeting
    without requiring a trained model — useful for real-time decisioning.
    """
    score = (
        (df["IsActiveMember"] == 0).astype(int) * 2
        + (df["Age"] > 50).astype(int)
        + (df["NumOfProducts"] >= 3).astype(int)
        + (df["Balance"] == 0).astype(int)
        + (df["Tenure"] < 2).astype(int)
    )
    bins   = [-1, 0, 1, 2, 3, np.inf]
    labels = [0, 1, 2, 3, 4]
    return pd.cut(score, bins=bins, labels=labels).astype(int).rename("ChurnRiskCategory")


# =============================================================================
# FEATURE 11 — BALANCE-TO-SALARY RATIO
# =============================================================================

def compute_balance_salary_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Balance-to-Salary Ratio (BSR)
    ──────────────────────────────
    Captures the proportion of estimated salary held as bank deposits.

    Formula:  BSR = Balance / (EstimatedSalary + 1)

    Business Meaning
    ----------------
    High BSR customers are financially committed to this bank as their
    primary savings institution.  Low BSR on a high-salary customer may
    indicate deposits are held elsewhere.
    """
    bsr = df["Balance"] / (df["EstimatedSalary"] + 1)
    return _norm(bsr).rename("BalanceSalaryRatio")


# =============================================================================
# FEATURE 12 — AGE SEGMENT (one-hot encoded ordinal)
# =============================================================================

def compute_age_segment(df: pd.DataFrame) -> pd.Series:
    """
    Age Segment (ordinal)
    ─────────────────────
    0=Young(18-30), 1=MidAge(31-45), 2=Senior(46-60), 3=Retiree(60+)
    """
    bins   = [17, 30, 45, 60, 120]
    labels = [0, 1, 2, 3]
    return pd.cut(df["Age"], bins=bins, labels=labels).astype(int).rename("AgeSegment")


# =============================================================================
# MASTER FEATURE ENGINEERING FUNCTION
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations to the cleaned dataframe.

    Returns
    -------
    pd.DataFrame
        Original columns + 12 engineered behavioural features.
    """
    df = df.copy()

    df["EngagementScore"]          = compute_engagement_score(df)
    df["RelationshipStrengthIndex"] = compute_relationship_strength_index(df)
    df["ProductUtilizationScore"]  = compute_product_utilization_score(df)
    df["LoyaltyScore"]             = compute_loyalty_score(df)
    df["PremiumRiskScore"]         = compute_premium_risk_score(df)
    df["FinancialCommitmentIndex"] = compute_financial_commitment_index(df)
    df["ActivityIntensityScore"]   = compute_activity_intensity_score(df)
    df["ProductDepthIndex"]        = compute_product_depth_index(df)
    df["StabilityScore"]           = compute_stability_score(df)
    df["ChurnRiskCategory"]        = compute_churn_risk_category(df)
    df["BalanceSalaryRatio"]       = compute_balance_salary_ratio(df)
    df["AgeSegment"]               = compute_age_segment(df)

    engineered_cols = [
        "EngagementScore", "RelationshipStrengthIndex", "ProductUtilizationScore",
        "LoyaltyScore", "PremiumRiskScore", "FinancialCommitmentIndex",
        "ActivityIntensityScore", "ProductDepthIndex", "StabilityScore",
        "ChurnRiskCategory", "BalanceSalaryRatio", "AgeSegment",
    ]

    logger.info(f"Feature engineering complete — added {len(engineered_cols)} features")
    return df


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from config.config import CLEAN_DATA_FILE, FEAT_DATA_FILE

    df_clean = pd.read_parquet(CLEAN_DATA_FILE)
    df_feat  = engineer_features(df_clean)

    FEAT_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_parquet(FEAT_DATA_FILE, index=False)

    print("\n=== ENGINEERED FEATURES SUMMARY ===")
    eng_cols = [c for c in df_feat.columns if c not in df_clean.columns]
    print(df_feat[eng_cols].describe().round(3).to_string())
    print(f"\n✅  Feature dataset saved → {FEAT_DATA_FILE}")
