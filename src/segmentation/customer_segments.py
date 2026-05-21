"""
=============================================================================
PHASE 5 — CUSTOMER SEGMENTATION ENGINE
src/segmentation/customer_segments.py
=============================================================================
KMeans-based behavioural segmentation producing six business-interpretable
customer personas.  Includes cluster profiling, silhouette analysis, and
a rule-based persona labelling layer.
=============================================================================
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


# =============================================================================
# SEGMENT FEATURES
# =============================================================================

SEGMENT_FEATURES = [
    "EngagementScore",
    "RelationshipStrengthIndex",
    "ProductUtilizationScore",
    "LoyaltyScore",
    "PremiumRiskScore",
    "FinancialCommitmentIndex",
    "ActivityIntensityScore",
    "StabilityScore",
    "BalanceSalaryRatio",
]


# =============================================================================
# OPTIMAL K SELECTION
# =============================================================================

def find_optimal_k(
    df: pd.DataFrame,
    features: List[str],
    k_range: range = range(3, 10),
) -> Dict:
    """
    Run KMeans for each k in k_range and return inertia + silhouette scores.
    Used for elbow-method and silhouette analysis.

    Returns
    -------
    dict with keys 'k_values', 'inertia', 'silhouette'
    """
    X      = StandardScaler().fit_transform(df[features].fillna(0))
    results = {"k_values": [], "inertia": [], "silhouette": []}

    for k in k_range:
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X)
        results["k_values"].append(k)
        results["inertia"].append(km.inertia_)
        if k > 1:
            results["silhouette"].append(silhouette_score(X, lbl))
        else:
            results["silhouette"].append(None)

    logger.info(f"Optimal k analysis complete for k in {list(k_range)}")
    return results


# =============================================================================
# KMEANS CLUSTERING
# =============================================================================

def run_kmeans_segmentation(
    df: pd.DataFrame,
    n_clusters: int = 6,
    features: Optional[List[str]] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    Fit KMeans on behavioural features and append cluster labels.

    Parameters
    ----------
    df          : cleaned + feature-engineered dataframe
    n_clusters  : number of personas
    features    : feature columns to cluster on (default: SEGMENT_FEATURES)
    random_state: reproducibility seed

    Returns
    -------
    (df_segmented, fitted_kmeans, fitted_scaler)
    """
    if features is None:
        features = SEGMENT_FEATURES

    # Subset and scale
    X_raw   = df[features].fillna(0)
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Cluster
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=15)
    df = df.copy()
    df["Cluster"] = km.fit_predict(X_scaled)

    # PCA for 2-D visualisation
    pca          = PCA(n_components=2, random_state=random_state)
    X_pca        = pca.fit_transform(X_scaled)
    df["PCA_1"]  = X_pca[:, 0]
    df["PCA_2"]  = X_pca[:, 1]

    sil = silhouette_score(X_scaled, df["Cluster"])
    logger.info(f"KMeans k={n_clusters} | Silhouette={sil:.3f}")

    return df, km, scaler


# =============================================================================
# PERSONA LABELLING — RULE-BASED
# =============================================================================

PERSONA_RULES = {
    "Loyal Active Customers":       {"EngagementScore": (0.55, 1.0), "LoyaltyScore": (0.50, 1.0)},
    "Multi-Product Loyalists":      {"ProductUtilizationScore": (0.40, 1.0), "LoyaltyScore": (0.55, 1.0)},
    "Premium Disengaged":           {"FinancialCommitmentIndex": (0.38, 1.0), "ActivityIntensityScore": (0.0, 0.30)},
    "Silent Churn Risk":            {"EngagementScore": (0.0,  0.30), "FinancialCommitmentIndex": (0.0, 0.35)},
    "Low-Value High-Risk":          {"FinancialCommitmentIndex": (0.0, 0.28), "PremiumRiskScore": (0.04, 1.0)},
    "Young Digital Customers":      {"StabilityScore": (0.0, 0.52), "ProductDepthIndex": (0.0, 0.18)},
}


def label_personas(df: pd.DataFrame, cluster_profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a business persona label to each cluster based on its centroid
    characteristics compared to the persona rule thresholds.

    Parameters
    ----------
    df               : segmented dataframe with 'Cluster' column
    cluster_profiles : output of profile_clusters()

    Returns
    -------
    df with 'Persona' column added
    """
    cluster_persona_map: Dict[int, str] = {}

    for cluster_id, row in cluster_profiles.iterrows():
        best_persona  = "Uncategorised"
        best_score    = -1

        for persona, rules in PERSONA_RULES.items():
            match_score = 0
            for feature, (lo, hi) in rules.items():
                if feature in row.index:
                    val         = row[feature]
                    match_score += 1 if lo <= val <= hi else 0

            if match_score > best_score:
                best_score  = match_score
                best_persona = persona

        cluster_persona_map[cluster_id] = best_persona

    df = df.copy()
    df["Persona"] = df["Cluster"].map(cluster_persona_map)
    logger.info(f"Personas assigned: {cluster_persona_map}")
    return df


# =============================================================================
# CLUSTER PROFILING
# =============================================================================

def profile_clusters(df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compute mean of each feature per cluster and add churn rate.

    Returns
    -------
    pd.DataFrame indexed by cluster_id
    """
    if features is None:
        features = SEGMENT_FEATURES + ["Exited", "Age", "Balance", "Tenure"]

    present_features = [f for f in features if f in df.columns]
    profile = (
        df.groupby("Cluster")[present_features]
        .mean()
        .round(3)
    )
    profile["CustomerCount"] = df.groupby("Cluster").size()
    profile["ChurnRate"]     = df.groupby("Cluster")["Exited"].mean().round(3)
    return profile


# =============================================================================
# MASTER SEGMENTATION FUNCTION
# =============================================================================

def segment_customers(df: pd.DataFrame, n_clusters: int = 6) -> pd.DataFrame:
    """
    End-to-end customer segmentation pipeline.

    1. Run KMeans
    2. Profile clusters
    3. Label personas
    4. Return enriched dataframe

    Returns
    -------
    pd.DataFrame with 'Cluster', 'Persona', 'PCA_1', 'PCA_2' columns
    """
    # Check if required engineered features exist
    missing = [f for f in SEGMENT_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing engineered features: {missing}. Run feature_engineering first.")

    df_seg, km, scaler = run_kmeans_segmentation(df, n_clusters=n_clusters)
    profiles           = profile_clusters(df_seg)
    df_seg             = label_personas(df_seg, profiles)

    logger.info("Customer segmentation pipeline complete")
    return df_seg


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from config.config import FEAT_DATA_FILE, SEG_DATA_FILE

    df_feat = pd.read_parquet(FEAT_DATA_FILE)
    df_seg  = segment_customers(df_feat, n_clusters=6)

    SEG_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_seg.to_parquet(SEG_DATA_FILE, index=False)

    print("\n=== CLUSTER PROFILES ===")
    print(profile_clusters(df_seg).to_string())

    print("\n=== PERSONA DISTRIBUTION ===")
    print(df_seg["Persona"].value_counts().to_string())
    print(f"\n✅  Segmented data saved → {SEG_DATA_FILE}")
