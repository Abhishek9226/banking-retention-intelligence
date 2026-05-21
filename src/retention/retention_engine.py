"""
=============================================================================
PHASE 9 — RETENTION INTELLIGENCE ENGINE
src/retention/retention_engine.py
=============================================================================
Rule-based + ML-driven retention recommendation system that maps each
customer to personalised intervention strategies with urgency tiers.
=============================================================================
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# RETENTION ACTION LIBRARY
# =============================================================================

RETENTION_ACTIONS = {
    "premium_relationship_manager": {
        "action":       "Assign dedicated Relationship Manager",
        "channel":      "Direct outreach (phone + email)",
        "timeline":     "Within 48 hours",
        "cost_tier":    "High",
        "description":  "High-balance disengaged customers warrant white-glove attention. "
                        "A personal RM touchpoint is proven to reduce churn by 35–45% in this segment.",
    },
    "loyalty_rewards_upgrade": {
        "action":       "Upgrade to Premium Loyalty Programme",
        "channel":      "Email + in-app notification",
        "timeline":     "Within 1 week",
        "cost_tier":    "Medium",
        "description":  "Offer fee-waiver, cashback acceleration, and priority service to long-tenured members.",
    },
    "credit_card_activation_campaign": {
        "action":       "Credit Card Activation + Rewards Push",
        "channel":      "SMS + mobile push + email",
        "timeline":     "Within 3 days",
        "cost_tier":    "Low",
        "description":  "Cardholders who have not used their card in 90+ days respond well to "
                        "time-limited cashback offers and spend-linked rewards.",
    },
    "cross_sell_second_product": {
        "action":       "Cross-sell Second Banking Product",
        "channel":      "Branch + digital assistant",
        "timeline":     "Within 2 weeks",
        "cost_tier":    "Low",
        "description":  "Single-product customers are the highest churn risk. "
                        "Introducing a savings account, personal loan, or investment product "
                        "doubles the switching cost.",
    },
    "digital_engagement_onboarding": {
        "action":       "Digital Engagement Campaign + Mobile Onboarding",
        "channel":      "App notification + email series",
        "timeline":     "Immediate",
        "cost_tier":    "Very Low",
        "description":  "Young customers respond to digital-first engagement. "
                        "Offer gamified savings goals, spending insights, and app-exclusive benefits.",
    },
    "reactivation_outreach": {
        "action":       "Inactive Member Reactivation Outreach",
        "channel":      "Phone + email + direct mail",
        "timeline":     "Within 1 week",
        "cost_tier":    "Medium",
        "description":  "Proactively contact inactive members with a 'We miss you' message "
                        "and a time-sensitive re-engagement incentive (e.g., fee waiver, bonus interest).",
    },
    "product_simplification": {
        "action":       "Product Simplification & Rationalisation Review",
        "channel":      "Branch appointment",
        "timeline":     "Within 2 weeks",
        "cost_tier":    "Medium",
        "description":  "Customers with 3–4 products often churn due to complexity. "
                        "A product review meeting to consolidate or optimise their portfolio "
                        "can significantly improve satisfaction.",
    },
    "geographic_targeted_offer": {
        "action":       "Region-Specific Retention Offer (Germany)",
        "channel":      "Email + direct mail",
        "timeline":     "Within 2 weeks",
        "cost_tier":    "Medium",
        "description":  "German customers churn at higher rates. Localised offers "
                        "addressing competitive differentiation (better rates, local partnerships) "
                        "can address market-specific attrition drivers.",
    },
    "senior_wealth_planning": {
        "action":       "Senior Customer Wealth Planning Session",
        "channel":      "Branch + phone consultation",
        "timeline":     "Within 1 month",
        "cost_tier":    "High",
        "description":  "Customers aged 50+ with high balances are likely evaluating "
                        "retirement / inheritance planning. Proactive wealth advisory "
                        "conversations can lock in these assets long-term.",
    },
    "watch_and_monitor": {
        "action":       "Monitor & Automated Early Warning",
        "channel":      "CRM flag + quarterly review",
        "timeline":     "Ongoing",
        "cost_tier":    "Negligible",
        "description":  "Low-risk customers who fall below engagement thresholds should "
                        "be flagged in CRM for automated monitoring and quarterly check-in.",
    },
}

URGENCY_COLOURS = {
    "CRITICAL": "🔴",
    "HIGH":     "🟠",
    "MEDIUM":   "🟡",
    "LOW":      "🟢",
    "MONITOR":  "⚪",
}


# =============================================================================
# RULE ENGINE
# =============================================================================

def determine_retention_actions(row: pd.Series) -> Dict:
    """
    Apply business rules to a single customer row and return a retention plan.

    Parameters
    ----------
    row : pd.Series
        A row from the segmented + feature-engineered dataframe.

    Returns
    -------
    dict with keys: urgency, primary_action, secondary_action, rationale
    """
    urgency          = "MONITOR"
    primary_action   = "watch_and_monitor"
    secondary_action = None
    rationale        = []

    # ── RULE 1: Premium disengaged ─────────────────────────────────────────
    high_balance = row.get("Balance", 0) > 50_000
    inactive     = row.get("IsActiveMember", 1) == 0
    if high_balance and inactive:
        urgency          = "CRITICAL"
        primary_action   = "premium_relationship_manager"
        secondary_action = "loyalty_rewards_upgrade"
        rationale.append("High-balance + inactive — AUM flight risk")

    # ── RULE 2: Multi-product high churn prob ─────────────────────────────
    elif row.get("NumOfProducts", 1) >= 3 and row.get("ChurnRiskCategory", 0) >= 3:
        urgency          = "HIGH"
        primary_action   = "product_simplification"
        secondary_action = "loyalty_rewards_upgrade"
        rationale.append("3+ products with high churn risk category — complexity churn")

    # ── RULE 3: German inactive customer ──────────────────────────────────
    elif str(row.get("Geography", "")).lower() == "germany" and inactive:
        urgency          = "HIGH"
        primary_action   = "geographic_targeted_offer"
        secondary_action = "reactivation_outreach"
        rationale.append("German market + inactive — regional churn risk")

    # ── RULE 4: Single product, early lifecycle ───────────────────────────
    elif row.get("NumOfProducts", 2) == 1 and row.get("Tenure", 5) < 3:
        urgency          = "HIGH"
        primary_action   = "cross_sell_second_product"
        secondary_action = "digital_engagement_onboarding"
        rationale.append("Single product + early tenure — highest switching risk segment")

    # ── RULE 5: Senior high-balance customer ──────────────────────────────
    elif row.get("Age", 30) >= 50 and high_balance:
        urgency          = "MEDIUM"
        primary_action   = "senior_wealth_planning"
        secondary_action = "loyalty_rewards_upgrade"
        rationale.append("50+ age with high balance — retirement planning migration risk")

    # ── RULE 6: Card holder + inactive ────────────────────────────────────
    elif row.get("HasCrCard", 0) == 1 and inactive:
        urgency          = "MEDIUM"
        primary_action   = "credit_card_activation_campaign"
        secondary_action = "reactivation_outreach"
        rationale.append("Card holder + inactive — card dormancy precedes full attrition")

    # ── RULE 7: Low engagement score ──────────────────────────────────────
    elif row.get("EngagementScore", 0.5) < 0.35:
        urgency          = "MEDIUM"
        primary_action   = "reactivation_outreach"
        secondary_action = "cross_sell_second_product"
        rationale.append("Low engagement score — disengagement pattern detected")

    # ── RULE 8: Young digital customer ────────────────────────────────────
    elif row.get("Age", 40) < 30:
        urgency          = "LOW"
        primary_action   = "digital_engagement_onboarding"
        secondary_action = "cross_sell_second_product"
        rationale.append("Young customer — digital engagement programme applicable")

    return {
        "urgency":          urgency,
        "urgency_icon":     URGENCY_COLOURS[urgency],
        "primary_action":   primary_action,
        "secondary_action": secondary_action,
        "rationale":        "; ".join(rationale) if rationale else "Standard monitoring",
        "primary_detail":   RETENTION_ACTIONS.get(primary_action, {}),
        "secondary_detail": RETENTION_ACTIONS.get(secondary_action, {}) if secondary_action else {},
    }


# =============================================================================
# BATCH RETENTION SCORING
# =============================================================================

def score_retention_actions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the retention rule engine to every customer and append recommendations.

    Returns
    -------
    pd.DataFrame with added retention columns
    """
    df = df.copy()
    plans = df.apply(determine_retention_actions, axis=1)

    df["RetentionUrgency"]          = plans.apply(lambda x: x["urgency"])
    df["RetentionUrgencyIcon"]      = plans.apply(lambda x: x["urgency_icon"])
    df["PrimaryAction"]             = plans.apply(lambda x: x["primary_action"])
    df["SecondaryAction"]           = plans.apply(lambda x: lambda x: x["secondary_action"])
    df["RetentionRationale"]        = plans.apply(lambda x: x["rationale"])

    logger.info("Retention action scoring complete")
    logger.info(df["RetentionUrgency"].value_counts().to_string())
    return df


# =============================================================================
# RETENTION DASHBOARD SUMMARY
# =============================================================================

def retention_summary(df: pd.DataFrame) -> Dict:
    """Return high-level retention intervention counts by urgency tier."""
    if "RetentionUrgency" not in df.columns:
        df = score_retention_actions(df)

    summary = {
        "total_customers":    len(df),
        "critical_count":     (df["RetentionUrgency"] == "CRITICAL").sum(),
        "high_count":         (df["RetentionUrgency"] == "HIGH").sum(),
        "medium_count":       (df["RetentionUrgency"] == "MEDIUM").sum(),
        "low_count":          (df["RetentionUrgency"] == "LOW").sum(),
        "monitor_count":      (df["RetentionUrgency"] == "MONITOR").sum(),
        "action_distribution": df["PrimaryAction"].value_counts().to_dict(),
    }

    return summary


# =============================================================================
# INDIVIDUAL RETENTION CARD
# =============================================================================

def get_retention_card(customer_row: pd.Series) -> str:
    """Generate a formatted retention recommendation card for one customer."""
    plan = determine_retention_actions(customer_row)
    pa   = plan["primary_detail"]
    sa   = plan["secondary_detail"]

    lines = [
        f"╔{'═'*60}╗",
        f"║  {plan['urgency_icon']}  RETENTION PLAN — Urgency: {plan['urgency']:10s}  ║",
        f"╠{'═'*60}╣",
        f"║  Rationale : {plan['rationale'][:55]}",
        f"╠{'═'*60}╣",
        f"║  PRIMARY ACTION",
        f"║  → {pa.get('action', 'N/A')}",
        f"║  Channel  : {pa.get('channel', 'N/A')}",
        f"║  Timeline : {pa.get('timeline', 'N/A')}",
        f"║  Cost     : {pa.get('cost_tier', 'N/A')}",
    ]

    if sa:
        lines += [
            f"╠{'═'*60}╣",
            f"║  SECONDARY ACTION",
            f"║  → {sa.get('action', 'N/A')}",
        ]

    lines.append(f"╚{'═'*60}╝")
    return "\n".join(lines)


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from config.config import FEAT_DATA_FILE

    df    = pd.read_parquet(FEAT_DATA_FILE)
    df_r  = score_retention_actions(df)

    print("\n=== RETENTION SUMMARY ===")
    summary = retention_summary(df_r)
    for k, v in summary.items():
        if k != "action_distribution":
            print(f"  {k}: {v}")

    print("\n=== SAMPLE RETENTION CARD ===")
    sample = df_r[df_r["RetentionUrgency"] == "CRITICAL"].iloc[0]
    print(get_retention_card(sample))
