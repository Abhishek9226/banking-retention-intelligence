"""
=============================================================================
MASTER PIPELINE RUNNER
run_pipeline.py
=============================================================================
Executes the complete end-to-end pipeline:
  Phase 2  → Data Engineering
  Phase 3  → Feature Engineering
  Phase 5  → Customer Segmentation
  Phase 6  → KPI Computation
  Phase 7  → Machine Learning
  Phase 8  → Explainable AI
  Phase 9  → Retention Scoring

Run from project root:  python run_pipeline.py
=============================================================================
"""

import logging
import sys
import time
from pathlib import Path

# Make sure all src modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def banner(title: str):
    logger.info("=" * 60)
    logger.info(f"  {title}")
    logger.info("=" * 60)


def main():
    t0 = time.time()

    # ── Imports ─────────────────────────────────────────────────────────
    from config.config import (
        RAW_DATA_FILE, CLEAN_DATA_FILE, FEAT_DATA_FILE,
        SEG_DATA_FILE, ARTIFACTS_DIR,
    )
    from data.generate_data                         import generate_banking_churn_dataset
    from src.ingestion.data_pipeline                import load_raw_data, clean_data, DataValidator
    from src.features.feature_engineering           import engineer_features
    from src.segmentation.customer_segments         import segment_customers, profile_clusters
    from src.kpi.kpi_engine                         import compute_all_kpis
    from src.ml.model_pipeline                      import run_full_ml_pipeline, format_results_table
    from src.xai.explainability                     import (
        get_feature_importance, compute_shap_values,
        shap_feature_summary, generate_churn_driver_report,
    )
    from src.retention.retention_engine             import score_retention_actions, retention_summary

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 2 — DATA ENGINEERING
    # ──────────────────────────────────────────────────────────────────────
    banner("PHASE 2 — DATA ENGINEERING")

    # Generate synthetic data if absent
    if not RAW_DATA_FILE.exists():
        logger.info("Generating synthetic dataset…")
        import pandas as pd
        df_gen = generate_banking_churn_dataset(10_000)
        RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        df_gen.to_csv(RAW_DATA_FILE, index=False)
        logger.info(f"Dataset saved → {RAW_DATA_FILE}")

    df_raw = load_raw_data(RAW_DATA_FILE)

    # Validate
    validator = DataValidator(df_raw)
    report    = validator.validate()
    logger.info(f"Validation passed: {report['passed']} | Issues: {report['issues']}")

    # Clean
    df_clean = clean_data(df_raw)
    CLEAN_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(CLEAN_DATA_FILE, index=False)
    logger.info(f"Clean data → {CLEAN_DATA_FILE} ({len(df_clean):,} rows)")

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 3 — FEATURE ENGINEERING
    # ──────────────────────────────────────────────────────────────────────
    banner("PHASE 3 — FEATURE ENGINEERING")

    df_feat = engineer_features(df_clean)
    FEAT_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_parquet(FEAT_DATA_FILE, index=False)
    logger.info(f"Feature dataset → {FEAT_DATA_FILE} ({df_feat.shape[1]} columns)")

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 5 — CUSTOMER SEGMENTATION
    # ──────────────────────────────────────────────────────────────────────
    banner("PHASE 5 — CUSTOMER SEGMENTATION")

    df_seg = segment_customers(df_feat, n_clusters=6)
    SEG_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_seg.to_parquet(SEG_DATA_FILE, index=False)

    logger.info("Cluster Profiles:")
    profiles = profile_clusters(df_seg)
    print(profiles[["CustomerCount", "ChurnRate", "EngagementScore"]].to_string())

    logger.info("\nPersona Distribution:")
    print(df_seg["Persona"].value_counts().to_string())

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 6 — KPI ENGINEERING
    # ──────────────────────────────────────────────────────────────────────
    banner("PHASE 6 — KPI ENGINEERING")

    kpis = compute_all_kpis(df_seg)
    print("\n" + "─" * 65)
    print(f"  {'KPI Name':<42}  {'Value':>8}  {'Status'}")
    print("─" * 65)
    for _, row in kpis.iterrows():
        print(f"  {row['name']:<42}  {row['display']:>8}  {row['health']}")
    print("─" * 65 + "\n")

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 7 — MACHINE LEARNING
    # ──────────────────────────────────────────────────────────────────────
    banner("PHASE 7 — MACHINE LEARNING")

    results, X_test, y_test, best_probs, fitted_models, best_name = run_full_ml_pipeline(
        df_seg, artifacts_dir=ARTIFACTS_DIR, tune=False
    )

    logger.info("\n=== MODEL COMPARISON ===")
    print(format_results_table(results).round(4).to_string())

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 8 — EXPLAINABLE AI
    # ──────────────────────────────────────────────────────────────────────
    banner("PHASE 8 — EXPLAINABLE AI")

    best_model   = fitted_models[best_name]
    feature_cols = list(X_test.columns)
    imp_df       = get_feature_importance(best_model, feature_cols, best_name)

    if not imp_df.empty:
        logger.info(f"Top 10 Features ({best_name}):")
        print(imp_df.head(10).to_string(index=False))

    try:
        shap_result = compute_shap_values(best_model, X_test, best_name, sample_size=300)
        if shap_result:
            shap_vals, _, X_sample = shap_result
            summary = shap_feature_summary(shap_vals, feature_cols)
            print(generate_churn_driver_report(summary))
    except Exception as e:
        logger.warning(f"SHAP analysis skipped: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 9 — RETENTION INTELLIGENCE
    # ──────────────────────────────────────────────────────────────────────
    banner("PHASE 9 — RETENTION INTELLIGENCE ENGINE")

    df_ret = score_retention_actions(df_seg)
    rsummary = retention_summary(df_ret)

    logger.info("Retention Action Summary:")
    for k, v in rsummary.items():
        if k != "action_distribution":
            logger.info(f"  {k}: {v}")

    # ──────────────────────────────────────────────────────────────────────
    # DONE
    # ──────────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"  ✅  PIPELINE COMPLETE in {elapsed:.1f}s")
    logger.info(f"  Best ML model : {best_name}")
    logger.info(f"  ROC-AUC       : {results[best_name]['roc_auc']:.4f}")
    logger.info(f"  F1-Score      : {results[best_name]['f1_score']:.4f}")
    logger.info("=" * 60)
    logger.info("  🚀  Launch dashboard: streamlit run dashboard/app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
