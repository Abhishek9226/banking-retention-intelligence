"""
=============================================================================
PHASE 4 — ADVANCED EXPLORATORY DATA ANALYSIS
src/eda/eda_analysis.py
=============================================================================
Produces 20+ publication-quality charts covering:
  - Churn distribution & demographics
  - Geography, gender, age analysis
  - Balance, salary, credit score analysis
  - Tenure & product adoption
  - Engagement behaviour
  - Correlation heatmap
  - Retention funnel
  - Custom KPI visuals
=============================================================================
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Style constants ───────────────────────────────────────────────────────────
PALETTE = {
    "primary":  "#0A2342",
    "gold":     "#D4AF37",
    "teal":     "#2EC4B6",
    "danger":   "#E63946",
    "warning":  "#FF9F1C",
    "success":  "#06D6A0",
    "purple":   "#7B2FBE",
    "slate":    "#64748B",
}
CHURN_PALETTE  = [PALETTE["teal"], PALETTE["danger"]]
GEO_PALETTE    = [PALETTE["primary"], PALETTE["gold"], PALETTE["teal"]]

def _style():
    plt.rcParams.update({
        "figure.facecolor": "#FAFBFC",
        "axes.facecolor":   "#FAFBFC",
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "grid.color":       "#CBD5E1",
        "font.family":      "DejaVu Sans",
        "axes.titlesize":   13,
        "axes.titleweight": "bold",
        "axes.titlepad":    12,
        "axes.labelsize":   10,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
    })

def _save(fig, path: Path, name: str):
    path.mkdir(parents=True, exist_ok=True)
    fp = path / name
    fig.savefig(fp, dpi=150, bbox_inches="tight", facecolor="#FAFBFC")
    plt.close(fig)
    logger.info(f"Saved: {fp.name}")
    return fp


# =============================================================================
# 1. CHURN DISTRIBUTION
# =============================================================================
def plot_churn_distribution(df: pd.DataFrame, out: Path):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Customer Churn Distribution", fontsize=15, fontweight="bold", color=PALETTE["primary"])

    # Pie
    counts = df["Exited"].value_counts()
    axes[0].pie(
        counts, labels=["Retained", "Churned"],
        colors=CHURN_PALETTE, autopct="%1.1f%%",
        startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 11},
    )
    axes[0].set_title("Retention vs Churn")

    # Bar by category
    churn_pct = df["Exited"].value_counts(normalize=True) * 100
    bars = axes[1].bar(["Retained", "Churned"], churn_pct.values,
                       color=CHURN_PALETTE, edgecolor="white", linewidth=1.5, width=0.5)
    for bar, val in zip(bars, churn_pct.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_title("Churn Rate Breakdown")
    axes[1].set_ylim(0, 100)

    plt.tight_layout()
    return _save(fig, out, "01_churn_distribution.png")


# =============================================================================
# 2. GEOGRAPHY-WISE CHURN
# =============================================================================
def plot_geography_churn(df: pd.DataFrame, out: Path):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Geography-Wise Churn Analysis", fontsize=15, fontweight="bold", color=PALETTE["primary"])

    geo_churn = df.groupby("Geography")["Exited"].agg(["count", "mean"]).reset_index()
    geo_churn.columns = ["Geography", "Customers", "ChurnRate"]

    # Customer volume
    bars = axes[0].bar(geo_churn["Geography"], geo_churn["Customers"],
                       color=GEO_PALETTE, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, geo_churn["Customers"]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                     f"{val:,}", ha="center", fontsize=10, fontweight="bold")
    axes[0].set_title("Customers by Geography")
    axes[0].set_ylabel("Number of Customers")

    # Churn rate
    bars2 = axes[1].bar(geo_churn["Geography"], geo_churn["ChurnRate"] * 100,
                        color=GEO_PALETTE, edgecolor="white", linewidth=1.5)
    axes[1].axhline(y=20.37, color=PALETTE["danger"], linestyle="--", lw=1.8, label="Portfolio Average")
    for bar, val in zip(bars2, geo_churn["ChurnRate"]):
        axes[1].text(bar.get_x() + bar.get_width()/2, val*100 + 0.3,
                     f"{val:.1%}", ha="center", fontsize=10, fontweight="bold")
    axes[1].set_title("Churn Rate by Geography")
    axes[1].set_ylabel("Churn Rate (%)")
    axes[1].legend()

    plt.tight_layout()
    return _save(fig, out, "02_geography_churn.png")


# =============================================================================
# 3. AGE SEGMENTATION
# =============================================================================
def plot_age_analysis(df: pd.DataFrame, out: Path):
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Age-Based Churn Analysis", fontsize=15, fontweight="bold", color=PALETTE["primary"])

    df = df.copy()
    df["AgeGroup"] = pd.cut(df["Age"], bins=[17,30,40,50,60,120],
                            labels=["18-30","31-40","41-50","51-60","60+"])

    # Distribution
    for status, color in zip([0, 1], CHURN_PALETTE):
        axes[0].hist(df[df["Exited"]==status]["Age"], bins=30, alpha=0.65,
                     color=color, label=["Retained","Churned"][status], edgecolor="white")
    axes[0].set_xlabel("Age"); axes[0].set_ylabel("Count")
    axes[0].set_title("Age Distribution by Churn")
    axes[0].legend()

    # Box plot
    axes[1].boxplot(
        [df[df["Exited"]==0]["Age"].values, df[df["Exited"]==1]["Age"].values],
        labels=["Retained","Churned"], patch_artist=True,
        boxprops=dict(facecolor=PALETTE["teal"], alpha=0.6),
        medianprops=dict(color=PALETTE["primary"], linewidth=2),
    )
    axes[1].set_ylabel("Age"); axes[1].set_title("Age Box Plot")

    # Churn by age group
    age_churn = df.groupby("AgeGroup", observed=True)["Exited"].mean() * 100
    axes[2].bar(age_churn.index.astype(str), age_churn.values,
                color=[PALETTE["teal"] if v < 20 else PALETTE["danger"] for v in age_churn.values],
                edgecolor="white", linewidth=1.5)
    axes[2].axhline(y=20.37, color=PALETTE["slate"], linestyle="--", lw=1.5, label="Average")
    for i, (idx, val) in enumerate(age_churn.items()):
        axes[2].text(i, val + 0.4, f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
    axes[2].set_xlabel("Age Group"); axes[2].set_ylabel("Churn Rate (%)")
    axes[2].set_title("Churn Rate by Age Group")
    axes[2].legend()

    plt.tight_layout()
    return _save(fig, out, "03_age_analysis.png")


# =============================================================================
# 4. BALANCE ANALYSIS
# =============================================================================
def plot_balance_analysis(df: pd.DataFrame, out: Path):
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Balance Analysis & Churn Relationship", fontsize=15, fontweight="bold", color=PALETTE["primary"])

    # Distribution (non-zero)
    df_nz = df[df["Balance"] > 0]
    for status, color in zip([0, 1], CHURN_PALETTE):
        axes[0].hist(df_nz[df_nz["Exited"]==status]["Balance"]/1000, bins=35, alpha=0.65,
                     color=color, label=["Retained","Churned"][status], edgecolor="white")
    axes[0].set_xlabel("Balance (€ thousands)"); axes[0].set_ylabel("Count")
    axes[0].set_title("Balance Distribution (Non-Zero)")
    axes[0].legend()

    # Zero vs non-zero balance churn
    df["ZeroBalance"] = (df["Balance"] == 0).map({True: "Zero Balance", False: "Has Balance"})
    zb_churn = df.groupby("ZeroBalance")["Exited"].mean() * 100
    bars = axes[1].bar(zb_churn.index, zb_churn.values,
                       color=[PALETTE["warning"], PALETTE["teal"]], edgecolor="white", linewidth=1.5, width=0.5)
    for bar, val in zip(bars, zb_churn.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.4,
                     f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Churn Rate (%)"); axes[1].set_title("Churn: Zero vs Non-Zero Balance")

    # Balance quartile churn
    # Balance quartile churn — handle many zeros by using cut on non-zero + zero bucket
    df_bal = df.copy()
    df_bal["BalanceQ"] = "Zero"
    non_zero = df_bal["Balance"] > 0
    df_bal.loc[non_zero, "BalanceQ"] = pd.qcut(
        df_bal.loc[non_zero, "Balance"], q=3, labels=["Low","Mid","High"]
    )
    bq_order = ["Zero","Low","Mid","High"]
    bq_churn = df_bal.groupby("BalanceQ")["Exited"].mean() * 100
    bq_churn = bq_churn.reindex([b for b in bq_order if b in bq_churn.index])
    bq_colors = [PALETTE["slate"], PALETTE["success"], PALETTE["warning"], PALETTE["danger"]][:len(bq_churn)]
    axes[2].bar(bq_churn.index.astype(str), bq_churn.values,
                color=bq_colors, edgecolor="white", linewidth=1.5)
    for i, val in enumerate(bq_churn.values):
        axes[2].text(i, val + 0.4, f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
    axes[2].set_xlabel("Balance Segment"); axes[2].set_ylabel("Churn Rate (%)")
    axes[2].set_title("Churn Rate by Balance Segment")

    plt.tight_layout()
    return _save(fig, out, "04_balance_analysis.png")


# =============================================================================
# 5. TENURE ANALYSIS
# =============================================================================
def plot_tenure_analysis(df: pd.DataFrame, out: Path):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Tenure & Loyalty Analysis", fontsize=15, fontweight="bold", color=PALETTE["primary"])

    tenure_churn = df.groupby("Tenure")["Exited"].agg(["mean", "count"]).reset_index()

    # Line chart
    axes[0].fill_between(tenure_churn["Tenure"], tenure_churn["mean"]*100,
                         alpha=0.2, color=PALETTE["danger"])
    axes[0].plot(tenure_churn["Tenure"], tenure_churn["mean"]*100,
                 color=PALETTE["danger"], lw=2.5, marker="o", ms=7)
    axes[0].axhline(y=20.37, color=PALETTE["slate"], ls="--", lw=1.5, label="Portfolio avg")
    axes[0].set_xlabel("Tenure (Years)"); axes[0].set_ylabel("Churn Rate (%)")
    axes[0].set_title("Churn Rate by Tenure")
    axes[0].set_xticks(range(0, 11))
    axes[0].legend()

    # Volume
    axes[1].bar(tenure_churn["Tenure"], tenure_churn["count"],
                color=PALETTE["primary"], alpha=0.8, edgecolor="white")
    axes[1].set_xlabel("Tenure (Years)"); axes[1].set_ylabel("Customers")
    axes[1].set_title("Customer Volume by Tenure")
    axes[1].set_xticks(range(0, 11))

    plt.tight_layout()
    return _save(fig, out, "05_tenure_analysis.png")


# =============================================================================
# 6. PRODUCT ADOPTION
# =============================================================================
def plot_product_analysis(df: pd.DataFrame, out: Path):
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Product Adoption & Utilization Analysis", fontsize=15, fontweight="bold", color=PALETTE["primary"])

    prod_churn = df.groupby("NumOfProducts")["Exited"].agg(["mean","count"]).reset_index()

    # Volume
    axes[0].bar(prod_churn["NumOfProducts"], prod_churn["count"],
                color=PALETTE["primary"], alpha=0.8, edgecolor="white", width=0.6)
    axes[0].set_xlabel("Number of Products"); axes[0].set_ylabel("Customers")
    axes[0].set_title("Customer Volume by Products Held")
    for i, (p, c) in prod_churn[["NumOfProducts","count"]].iterrows():
        axes[0].text(p, c + 30, f"{c:,}", ha="center", fontsize=9, fontweight="bold")

    # Churn rate
    colors = [PALETTE["teal"] if v < 0.20 else PALETTE["danger"] for v in prod_churn["mean"]]
    axes[1].bar(prod_churn["NumOfProducts"], prod_churn["mean"]*100,
                color=colors, edgecolor="white", linewidth=1.5, width=0.6)
    axes[1].axhline(y=20.37, color=PALETTE["slate"], ls="--", lw=1.5)
    for i, (p, r) in prod_churn[["NumOfProducts","mean"]].iterrows():
        axes[1].text(p, r*100 + 0.5, f"{r:.1%}", ha="center", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Number of Products"); axes[1].set_ylabel("Churn Rate (%)")
    axes[1].set_title("Churn Rate by Products Held")

    # Credit card stickiness
    cc_churn = df.groupby(["HasCrCard","Exited"]).size().reset_index(name="Count")
    cc_churn["Label"] = cc_churn["HasCrCard"].map({0:"No Card",1:"Has Card"})
    cc_total = cc_churn.groupby("Label")["Count"].transform("sum")
    cc_churn["Pct"] = cc_churn["Count"] / cc_total * 100
    churned_cc = cc_churn[cc_churn["Exited"]==1]
    axes[2].bar(churned_cc["Label"], churned_cc["Pct"],
                color=[PALETTE["warning"], PALETTE["teal"]], edgecolor="white", linewidth=1.5, width=0.5)
    for i, val in enumerate(churned_cc["Pct"].values):
        axes[2].text(i, val + 0.3, f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Churn Rate (%)"); axes[2].set_title("Churn: Has Credit Card vs Not")

    plt.tight_layout()
    return _save(fig, out, "06_product_analysis.png")


# =============================================================================
# 7. CORRELATION HEATMAP
# =============================================================================
def plot_correlation_heatmap(df: pd.DataFrame, out: Path):
    _style()
    cols = ["CreditScore","Age","Tenure","Balance","NumOfProducts",
            "HasCrCard","IsActiveMember","EstimatedSalary","Exited",
            "EngagementScore","LoyaltyScore","StabilityScore",
            "FinancialCommitmentIndex","PremiumRiskScore"]
    cols = [c for c in cols if c in df.columns]

    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(13, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size":8},
                cbar_kws={"shrink":0.8})
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold",
                 color=PALETTE["primary"], pad=15)
    plt.tight_layout()
    return _save(fig, out, "07_correlation_heatmap.png")


# =============================================================================
# 8. ENGAGEMENT ANALYSIS
# =============================================================================
def plot_engagement_analysis(df: pd.DataFrame, out: Path):
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Customer Engagement Analytics", fontsize=15, fontweight="bold", color=PALETTE["primary"])

    # Engagement score distribution
    for status, color, label in zip([0,1], CHURN_PALETTE, ["Retained","Churned"]):
        axes[0,0].hist(df[df["Exited"]==status]["EngagementScore"], bins=30, alpha=0.65,
                       color=color, label=label, edgecolor="white")
    axes[0,0].set_xlabel("Engagement Score"); axes[0,0].set_ylabel("Count")
    axes[0,0].set_title("Engagement Score by Churn Status"); axes[0,0].legend()

    # Active vs Inactive churn
    act_churn = df.groupby("IsActiveMember")["Exited"].mean() * 100
    bars = axes[0,1].bar(["Inactive","Active"], act_churn.values,
                          color=[PALETTE["danger"], PALETTE["teal"]], edgecolor="white", linewidth=1.5, width=0.5)
    for bar, val in zip(bars, act_churn.values):
        axes[0,1].text(bar.get_x()+bar.get_width()/2, val+0.4,
                       f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")
    axes[0,1].set_ylabel("Churn Rate (%)"); axes[0,1].set_title("Churn: Active vs Inactive Members")

    # Loyalty score
    df_s = df.sample(min(2000, len(df)), random_state=42)
    for status, color, label in zip([0,1], CHURN_PALETTE, ["Retained","Churned"]):
        sub = df_s[df_s["Exited"]==status]
        axes[1,0].scatter(sub["LoyaltyScore"], sub["EngagementScore"],
                          alpha=0.3, s=18, color=color, label=label)
    axes[1,0].set_xlabel("Loyalty Score"); axes[1,0].set_ylabel("Engagement Score")
    axes[1,0].set_title("Loyalty vs Engagement (coloured by Churn)"); axes[1,0].legend()

    # Activity intensity by geography
    geo_eng = df.groupby("Geography")["ActivityIntensityScore"].mean()
    axes[1,1].bar(geo_eng.index, geo_eng.values, color=GEO_PALETTE, edgecolor="white", linewidth=1.5)
    axes[1,1].set_ylabel("Activity Intensity Score")
    axes[1,1].set_title("Average Activity Intensity by Geography")
    for i, val in enumerate(geo_eng.values):
        axes[1,1].text(i, val + 0.003, f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    return _save(fig, out, "08_engagement_analysis.png")


# =============================================================================
# 9. GENDER ANALYSIS
# =============================================================================
def plot_gender_analysis(df: pd.DataFrame, out: Path):
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Gender-Based Churn Analysis", fontsize=15, fontweight="bold", color=PALETTE["primary"])

    gender_churn = df.groupby("Gender")["Exited"].agg(["mean","count"]).reset_index()
    gender_palette = [PALETTE["primary"], PALETTE["gold"]]

    axes[0].bar(gender_churn["Gender"], gender_churn["count"],
                color=gender_palette, edgecolor="white", linewidth=1.5, width=0.5)
    for i, row in gender_churn.iterrows():
        axes[0].text(i, row["count"]+30, f'{row["count"]:,}', ha="center", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Customers"); axes[0].set_title("Customer Volume by Gender")

    axes[1].bar(gender_churn["Gender"], gender_churn["mean"]*100,
                color=gender_palette, edgecolor="white", linewidth=1.5, width=0.5)
    axes[1].axhline(y=20.37, color=PALETTE["slate"], ls="--", lw=1.5, label="Portfolio avg")
    for i, row in gender_churn.iterrows():
        axes[1].text(i, row["mean"]*100+0.4, f'{row["mean"]:.1%}', ha="center", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Churn Rate (%)"); axes[1].set_title("Churn Rate by Gender"); axes[1].legend()

    gg_churn = df.groupby(["Gender","Geography"])["Exited"].mean().unstack() * 100
    gg_churn.plot(kind="bar", ax=axes[2], color=GEO_PALETTE, edgecolor="white", linewidth=1.5, width=0.6)
    axes[2].set_xlabel("Gender"); axes[2].set_ylabel("Churn Rate (%)")
    axes[2].set_title("Churn Rate: Gender × Geography"); axes[2].legend(title="Geography")
    axes[2].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    return _save(fig, out, "09_gender_analysis.png")


# =============================================================================
# 10. FEATURE IMPORTANCE CHART
# =============================================================================
def plot_feature_importance(imp_df: pd.DataFrame, out: Path, model_name: str = "LightGBM"):
    _style()
    top15 = imp_df.head(15).sort_values("importance")
    colors = [PALETTE["danger"] if "Risk" in f or "Churn" in f
              else (PALETTE["gold"] if "Engagement" in f or "Loyalty" in f or "Activity" in f
              else PALETTE["primary"]) for f in top15["feature"]]

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(top15["feature"], top15["importance_pct"], color=colors,
                   edgecolor="white", linewidth=1.2, height=0.7)
    for bar, val in zip(bars, top15["importance_pct"]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Feature Importance (%)")
    ax.set_title(f"Top 15 Feature Importances — {model_name}", pad=15)
    ax.set_xlim(0, top15["importance_pct"].max() * 1.18)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE["primary"], label="Raw Features"),
        Patch(facecolor=PALETTE["gold"],    label="Engagement Features"),
        Patch(facecolor=PALETTE["danger"],  label="Risk Features"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    plt.tight_layout()
    return _save(fig, out, "10_feature_importance.png")


# =============================================================================
# 11. CUSTOMER SEGMENTATION SCATTER
# =============================================================================
def plot_segmentation(df: pd.DataFrame, out: Path):
    if "PCA_1" not in df.columns or "Persona" not in df.columns:
        return None
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Customer Segmentation Analysis", fontsize=15, fontweight="bold", color=PALETTE["primary"])

    personas = df["Persona"].unique()
    colors   = [PALETTE["primary"], PALETTE["gold"], PALETTE["teal"],
                PALETTE["danger"], PALETTE["warning"], PALETTE["purple"]]

    sample = df.sample(min(3000, len(df)), random_state=42)
    for persona, color in zip(personas, colors):
        mask = sample["Persona"] == persona
        axes[0].scatter(sample[mask]["PCA_1"], sample[mask]["PCA_2"],
                        alpha=0.45, s=15, color=color, label=persona)
    axes[0].set_xlabel("PC 1"); axes[0].set_ylabel("PC 2")
    axes[0].set_title("Customer Segments — PCA 2D")
    axes[0].legend(fontsize=8, markerscale=2)

    persona_churn = df.groupby("Persona")["Exited"].mean().sort_values(ascending=False) * 100
    bar_colors = colors[:len(persona_churn)]
    bars = axes[1].barh(persona_churn.index, persona_churn.values,
                        color=bar_colors, edgecolor="white", linewidth=1.2, height=0.6)
    for bar, val in zip(bars, persona_churn.values):
        axes[1].text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                     f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")
    axes[1].set_xlabel("Churn Rate (%)")
    axes[1].set_title("Churn Rate by Persona")

    plt.tight_layout()
    return _save(fig, out, "11_customer_segmentation.png")


# =============================================================================
# 12. KPI SUMMARY VISUAL
# =============================================================================
def plot_kpi_summary(kpis: pd.DataFrame, out: Path):
    _style()
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(PALETTE["primary"])
    ax.set_facecolor(PALETTE["primary"])

    rag_colors = {
        "🟢 Green": PALETTE["success"],
        "🟡 Amber": PALETTE["warning"],
        "🔴 Red":   PALETTE["danger"],
    }

    for i, (_, row) in enumerate(kpis.iterrows()):
        color = rag_colors.get(row["health"], PALETTE["slate"])
        ax.text(0.02, 1 - (i+1)*0.115, row["name"], transform=ax.transAxes,
                color="white", fontsize=10, va="center")
        ax.text(0.72, 1 - (i+1)*0.115, row["display"], transform=ax.transAxes,
                color=PALETTE["gold"], fontsize=11, fontweight="bold", va="center")
        ax.text(0.88, 1 - (i+1)*0.115, row["health"], transform=ax.transAxes,
                color=color, fontsize=10, va="center")

    ax.set_axis_off()
    ax.text(0.5, 0.97, "🏦  Banking KPI Dashboard", transform=ax.transAxes,
            color=PALETTE["gold"], fontsize=14, fontweight="bold", ha="center", va="top")

    plt.tight_layout()
    return _save(fig, out, "12_kpi_summary.png")


# =============================================================================
# MASTER EDA RUNNER
# =============================================================================
def run_full_eda(df: pd.DataFrame, imp_df: pd.DataFrame, kpis, out_path: Path):
    """Run all EDA plots and save to out_path directory."""
    out_path.mkdir(parents=True, exist_ok=True)
    saved = []
    saved.append(plot_churn_distribution(df, out_path))
    saved.append(plot_geography_churn(df, out_path))
    saved.append(plot_age_analysis(df, out_path))
    saved.append(plot_balance_analysis(df, out_path))
    saved.append(plot_tenure_analysis(df, out_path))
    saved.append(plot_product_analysis(df, out_path))
    saved.append(plot_correlation_heatmap(df, out_path))
    saved.append(plot_engagement_analysis(df, out_path))
    saved.append(plot_gender_analysis(df, out_path))
    if not imp_df.empty:
        saved.append(plot_feature_importance(imp_df, out_path))
    seg_path = plot_segmentation(df, out_path)
    if seg_path: saved.append(seg_path)
    saved.append(plot_kpi_summary(kpis, out_path))
    logger.info(f"EDA complete — {len(saved)} charts saved to {out_path}")
    return saved


# =============================================================================
# ENTRYPOINT
# =============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from config.config import SEG_DATA_FILE, FIGURES_DIR
    from src.kpi.kpi_engine import compute_all_kpis
    from src.ml.model_pipeline import run_full_ml_pipeline
    from src.xai.explainability import get_feature_importance

    df   = pd.read_parquet(SEG_DATA_FILE)
    kpis = compute_all_kpis(df)
    results, X_test, y_test, best_probs, fitted_models, best_name = run_full_ml_pipeline(df)
    imp_df = get_feature_importance(fitted_models[best_name], list(X_test.columns), best_name)

    run_full_eda(df, imp_df, kpis, FIGURES_DIR)
    print(f"\n✅  All EDA charts saved → {FIGURES_DIR}")
