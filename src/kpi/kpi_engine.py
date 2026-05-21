"""
=============================================================================
PHASE 6 — KPI ENGINEERING MODULE
src/kpi/kpi_engine.py
=============================================================================
Computes eight production-grade banking KPIs from the feature-engineered
customer dataset.  Each KPI has:
  - A clear formula
  - A business interpretation
  - A threshold-based health signal (Green / Amber / Red)
=============================================================================
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# KPI THRESHOLDS  (RAG — Red / Amber / Green)
# ─────────────────────────────────────────────────────────────────────────────
KPI_THRESHOLDS = {
    "EngagementRetentionRatio":         {"green": 0.70, "amber": 0.50},
    "ProductDepthIndex_portfolio":      {"green": 0.55, "amber": 0.40},
    "RelationshipStrengthIndex_avg":    {"green": 0.60, "amber": 0.45},
    "PremiumChurnRiskScore":            {"green": 0.20, "amber": 0.35},   # lower is better
    "CreditCardStickinessScore":        {"green": 0.65, "amber": 0.50},
    "ActiveLoyaltyIndex":               {"green": 0.60, "amber": 0.45},
    "CustomerLifetimeStabilityIndicator": {"green": 0.65, "amber": 0.50},
    "BehavioralRiskScore":              {"green": 0.25, "amber": 0.40},   # lower is better
}


def _rag(value: float, thresholds: dict, lower_is_better: bool = False) -> str:
    """Return Red/Amber/Green based on value and threshold dict."""
    g, a = thresholds["green"], thresholds["amber"]
    if lower_is_better:
        return "🟢 Green" if value <= a else ("🟡 Amber" if value <= g else "🔴 Red")
    return "🟢 Green" if value >= g else ("🟡 Amber" if value >= a else "🔴 Red")


# =============================================================================
# KPI 1 — ENGAGEMENT RETENTION RATIO
# =============================================================================

def kpi_engagement_retention_ratio(df: pd.DataFrame) -> Dict:
    """
    Engagement Retention Ratio (ERR)
    ─────────────────────────────────
    ERR = (Active & Retained Customers) / Total Customers

    Business Meaning
    ----------------
    Measures the fraction of the customer base that is both engaged (active
    member) AND retained (not churned).  Target: > 70 %.
    """
    active_retained = ((df["IsActiveMember"] == 1) & (df["Exited"] == 0)).sum()
    value = active_retained / len(df)
    return {
        "name":             "Engagement Retention Ratio",
        "value":            round(value, 4),
        "display":          f"{value:.1%}",
        "formula":          "Active & Retained / Total Customers",
        "interpretation":   "% of customers who are engaged AND retained",
        "health":           _rag(value, KPI_THRESHOLDS["EngagementRetentionRatio"]),
    }


# =============================================================================
# KPI 2 — PRODUCT DEPTH INDEX (PORTFOLIO)
# =============================================================================

def kpi_product_depth_index(df: pd.DataFrame) -> Dict:
    """
    Portfolio Product Depth Index
    ─────────────────────────────
    PDI = mean(NumOfProducts) / 4  (normalised to max products)

    Business Meaning
    ----------------
    Average cross-sell penetration across the portfolio.
    Higher PDI = deeper product relationships = lower attrition risk.
    """
    value = df["NumOfProducts"].mean() / 4
    return {
        "name":           "Product Depth Index",
        "value":          round(value, 4),
        "display":        f"{value:.1%}",
        "formula":        "mean(NumOfProducts) / 4",
        "interpretation": "Average product penetration per customer",
        "health":         _rag(value, KPI_THRESHOLDS["ProductDepthIndex_portfolio"]),
    }


# =============================================================================
# KPI 3 — RELATIONSHIP STRENGTH INDEX (AVERAGE)
# =============================================================================

def kpi_relationship_strength(df: pd.DataFrame) -> Dict:
    """
    Portfolio-Level Relationship Strength
    ──────────────────────────────────────
    RSI_avg = mean(RelationshipStrengthIndex)

    Business Meaning
    ----------------
    Summarises the depth of customer–bank relationships across the portfolio.
    Decline in RSI over time predicts upcoming churn waves.
    """
    if "RelationshipStrengthIndex" not in df.columns:
        value = 0.0
    else:
        value = df["RelationshipStrengthIndex"].mean()

    return {
        "name":           "Avg Relationship Strength Index",
        "value":          round(value, 4),
        "display":        f"{value:.3f}",
        "formula":        "mean(RelationshipStrengthIndex)",
        "interpretation": "Average depth of customer-bank relationship",
        "health":         _rag(value, KPI_THRESHOLDS["RelationshipStrengthIndex_avg"]),
    }


# =============================================================================
# KPI 4 — PREMIUM CHURN RISK SCORE
# =============================================================================

def kpi_premium_churn_risk(df: pd.DataFrame) -> Dict:
    """
    Premium Churn Risk Score (PCRS)
    ────────────────────────────────
    PCRS = Churned High-Value Customers / Total High-Value Customers

    High-value = Balance > 75th percentile

    Business Meaning
    ----------------
    Measures AUM erosion risk at the premium segment.
    Every 1 % rise in PCRS represents significant revenue exposure.
    """
    p75         = df["Balance"].quantile(0.75)
    premium_seg = df[df["Balance"] >= p75]
    value       = premium_seg["Exited"].mean() if len(premium_seg) > 0 else 0.0

    return {
        "name":           "Premium Churn Risk Score",
        "value":          round(value, 4),
        "display":        f"{value:.1%}",
        "formula":        "Churned(Balance>P75) / Total(Balance>P75)",
        "interpretation": "Churn rate among top-quartile balance customers",
        "health":         _rag(value, KPI_THRESHOLDS["PremiumChurnRiskScore"], lower_is_better=True),
    }


# =============================================================================
# KPI 5 — CREDIT CARD STICKINESS SCORE
# =============================================================================

def kpi_credit_card_stickiness(df: pd.DataFrame) -> Dict:
    """
    Credit Card Stickiness Score (CCSS)
    ────────────────────────────────────
    CCSS = Retained Card Holders / Total Card Holders

    Business Meaning
    ----------------
    Credit cards are high-margin fee products.  Low stickiness indicates
    missed cross-sell opportunities or competitor card migration.
    """
    card_holders = df[df["HasCrCard"] == 1]
    value        = (card_holders["Exited"] == 0).mean() if len(card_holders) > 0 else 0.0

    return {
        "name":           "Credit Card Stickiness Score",
        "value":          round(value, 4),
        "display":        f"{value:.1%}",
        "formula":        "Retained(HasCrCard=1) / Total(HasCrCard=1)",
        "interpretation": "Retention rate among credit card holders",
        "health":         _rag(value, KPI_THRESHOLDS["CreditCardStickinessScore"]),
    }


# =============================================================================
# KPI 6 — ACTIVE LOYALTY INDEX
# =============================================================================

def kpi_active_loyalty_index(df: pd.DataFrame) -> Dict:
    """
    Active Loyalty Index (ALI)
    ───────────────────────────
    ALI = mean(LoyaltyScore) × IsActiveMember_rate

    Business Meaning
    ----------------
    Penalises loyalty scores by the share of actually-active customers.
    A portfolio with high loyalty scores but low activity rates is vulnerable.
    """
    if "LoyaltyScore" not in df.columns:
        value = 0.0
    else:
        active_rate = df["IsActiveMember"].mean()
        value       = df["LoyaltyScore"].mean() * active_rate

    return {
        "name":           "Active Loyalty Index",
        "value":          round(value, 4),
        "display":        f"{value:.3f}",
        "formula":        "mean(LoyaltyScore) × Active_Rate",
        "interpretation": "Loyalty score weighted by actual activity level",
        "health":         _rag(value, KPI_THRESHOLDS["ActiveLoyaltyIndex"]),
    }


# =============================================================================
# KPI 7 — CUSTOMER LIFETIME STABILITY INDICATOR
# =============================================================================

def kpi_lifetime_stability(df: pd.DataFrame) -> Dict:
    """
    Customer Lifetime Stability Indicator (CLSI)
    ─────────────────────────────────────────────
    CLSI = mean(StabilityScore) × (1 - ChurnRate)

    Business Meaning
    ----------------
    Combines financial health (stability) with actual retention outcome.
    High CLSI = stable customers who are staying → optimal portfolio state.
    """
    if "StabilityScore" not in df.columns:
        value = 0.0
    else:
        churn_rate = df["Exited"].mean()
        value      = df["StabilityScore"].mean() * (1 - churn_rate)

    return {
        "name":           "Customer Lifetime Stability Indicator",
        "value":          round(value, 4),
        "display":        f"{value:.3f}",
        "formula":        "mean(StabilityScore) × (1 - ChurnRate)",
        "interpretation": "Portfolio stability adjusted for actual churn",
        "health":         _rag(value, KPI_THRESHOLDS["CustomerLifetimeStabilityIndicator"]),
    }


# =============================================================================
# KPI 8 — BEHAVIORAL RISK SCORE
# =============================================================================

def kpi_behavioral_risk_score(df: pd.DataFrame) -> Dict:
    """
    Behavioral Risk Score (BRS)
    ────────────────────────────
    BRS = Inactive Customers with Balance > P50 / Total Customers

    Business Meaning
    ----------------
    Identifies the "ticking time-bomb" segment: customers with assets but no
    engagement.  High BRS signals an imminent AUM outflow risk.
    """
    p50          = df["Balance"].quantile(0.50)
    risky        = ((df["IsActiveMember"] == 0) & (df["Balance"] > p50)).sum()
    value        = risky / len(df)

    return {
        "name":           "Behavioral Risk Score",
        "value":          round(value, 4),
        "display":        f"{value:.1%}",
        "formula":        "Inactive(Balance>P50) / Total Customers",
        "interpretation": "Share of disengaged mid-to-high-balance customers",
        "health":         _rag(value, KPI_THRESHOLDS["BehavioralRiskScore"], lower_is_better=True),
    }


# =============================================================================
# MASTER KPI DASHBOARD
# =============================================================================

def compute_all_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 8 KPIs and return as a tidy DataFrame.

    Returns
    -------
    pd.DataFrame with columns: name, value, display, formula, interpretation, health
    """
    kpi_functions = [
        kpi_engagement_retention_ratio,
        kpi_product_depth_index,
        kpi_relationship_strength,
        kpi_premium_churn_risk,
        kpi_credit_card_stickiness,
        kpi_active_loyalty_index,
        kpi_lifetime_stability,
        kpi_behavioral_risk_score,
    ]

    records = [fn(df) for fn in kpi_functions]
    df_kpi  = pd.DataFrame(records)
    logger.info(f"Computed {len(df_kpi)} KPIs")
    return df_kpi


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from config.config import FEAT_DATA_FILE

    df = pd.read_parquet(FEAT_DATA_FILE)
    kpis = compute_all_kpis(df)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║         BANKING RETENTION KPI DASHBOARD                 ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    for _, row in kpis.iterrows():
        print(f"  {row['health']}  {row['name']:45s}  {row['display']:>8s}")
    print()
