# 📋 Resume & Interview Preparation Guide
## BankRetain IQ — Customer Retention Intelligence Platform

---

## 🔷 ATS-Optimised Resume Bullet Points

### Data Science / ML Engineer Resume

```
• Architected end-to-end banking churn prediction system using LightGBM with SMOTE
  oversampling, achieving ROC-AUC of 0.861 on 10,000-customer European banking dataset
  across 3 geographies (France, Germany, Spain)

• Engineered 12 domain-specific behavioural features (Engagement Score, Relationship
  Strength Index, Premium Risk Score, Financial Commitment Index) improving model
  prediction performance by ~10% over raw feature baseline

• Implemented 5-model comparative ML evaluation (Logistic Regression, Decision Tree,
  Random Forest, XGBoost, LightGBM) with stratified 5-fold cross-validation and
  SMOTE for class imbalance handling (20.37% churn rate)

• Deployed SHAP explainability analysis identifying Age, Balance, and Engagement
  Composite as top churn drivers; generated executive-ready causal narratives for
  business stakeholder communication

• Built KMeans customer segmentation (k=6 clusters) with PCA visualisation, producing
  6 business-interpretable personas with churn rates ranging from 13.4% to 27.8%

• Designed 8 banking-specific portfolio KPIs (Engagement Retention Ratio, Product
  Depth Index, Credit Card Stickiness, Behavioral Risk Score) with RAG thresholds
  for executive monitoring

• Developed rule-based retention intelligence engine with 8 intervention types and
  5 urgency tiers (Critical/High/Medium/Low/Monitor), generating personalised
  retention plans for 10,000+ customers

• Deployed 8-tab interactive Streamlit dashboard with Plotly visualisations, dynamic
  filtering, churn probability simulator, and downloadable customer reports
```

---

## 🔷 LinkedIn Project Summary

**BankRetain IQ — Customer Retention Intelligence Platform**

Built a full-stack, production-grade banking analytics platform addressing €6–15M annual churn exposure for a European retail bank:

🤖 **Machine Learning:** LightGBM churn prediction model with 86.1% AUC, trained on 10,000 customers with SMOTE oversampling and 5-fold cross-validation across 5 competing models

🔬 **Feature Engineering:** 12 domain-specific behavioural indices (Engagement Score, Relationship Strength, Premium Risk, Financial Commitment) improving model performance 10% over baseline

📊 **Analytics:** 20+ EDA charts, 8 portfolio KPIs with RAG thresholds, SHAP explainability layer with executive-level churn driver narratives

👥 **Segmentation:** KMeans customer personas (Loyal Active, Premium Disengaged, Silent Churn Risk) with PCA 2D visualisation

🛡️ **Retention Engine:** Rule-based intelligence system generating personalised intervention plans with urgency tiers and estimated €6–21M revenue impact

🖥️ **Dashboard:** 8-tab interactive Streamlit platform with Plotly charts, churn simulator, customer lookup, and CSV export

**Tech Stack:** Python | LightGBM | SHAP | Scikit-learn | Streamlit | Plotly | Pandas | KMeans | imbalanced-learn

---

## 🔷 Interview Questions & Expert Answers

### Q1: Why did you choose LightGBM over XGBoost for this problem?

**Answer:** Both models achieved near-identical ROC-AUC (~0.861 vs 0.859), but LightGBM was selected based on three factors:

1. **Accuracy:** LightGBM's leaf-wise tree growth (vs XGBoost's level-wise) captured the non-linear age and balance interaction effects more precisely, yielding the highest F1-score (0.630 vs 0.602)
2. **Training Speed:** For production deployment with daily model retraining on growing data, LightGBM is typically 3–10× faster than XGBoost on tabular data
3. **Imbalance Handling:** LightGBM's `class_weight='balanced'` combined with SMOTE gave better precision-recall trade-off than XGBoost's `scale_pos_weight` alone

In production, I would A/B test both models on live traffic before making a final deployment decision.

---

### Q2: How did you handle the class imbalance (20.37% churn rate)?

**Answer:** I used a two-pronged approach:

1. **SMOTE (Synthetic Minority Oversampling Technique):** Applied within the training pipeline (not to test data — critical to avoid data leakage) at sampling_strategy=0.75, creating synthetic churner examples in feature space rather than duplicating real ones
2. **Class-Weight Balancing:** All models were configured with `class_weight='balanced'` (or equivalent `scale_pos_weight=4` for XGBoost), penalising misclassification of the minority class more heavily

The key implementation detail: SMOTE is applied inside the `imblearn.Pipeline`, ensuring it only sees training folds during cross-validation. Applying SMOTE before the train/test split would cause data leakage and artificially inflate validation metrics.

---

### Q3: What does the Financial Commitment Index measure and why did it outperform raw Balance?

**Answer:** The Financial Commitment Index (FCI) is:
```
FCI = 0.50 × norm(Balance) + 0.30 × norm(EstimatedSalary) + 0.20 × norm(Products)
```

It captures the *total financial depth* of a customer's engagement, not just their current balance. Here's why it outperforms raw Balance:

- A customer with €80,000 balance and €200,000 salary has weaker commitment than one with €80,000 balance and €50,000 salary (the latter is more financially dependent on this bank)
- Product count adds a relationship breadth dimension that balance alone misses
- By weighting salary, we identify customers who *could* have more deposits but don't — an early churn signal

In SHAP terms, FCI achieved 7.09% mean absolute importance vs Balance at 7.47% — comparable but complementary, suggesting they capture slightly different aspects of financial engagement.

---

### Q4: How would you deploy this model in a real bank?

**Answer:** A production deployment would involve:

1. **Model Serving:** LightGBM model serialised to pickle/ONNX format, served via a FastAPI REST endpoint — `POST /predict` accepts customer feature vector, returns `{"churn_probability": 0.73, "risk_tier": "HIGH"}`

2. **Integration:** Core banking system sends daily customer feature snapshots to the scoring API; results stored in the CRM as custom fields

3. **Monitoring:** MLflow or Evidently AI for tracking model performance drift. If AUC drops >3% from baseline, retrain triggers automatically

4. **Retraining Pipeline:** Quarterly retraining incorporating intervention outcomes — did customers who received a retention offer actually stay? This feedback loop continuously improves the model

5. **Governance:** Explainability reports generated for every high-risk prediction (regulatory requirement under GDPR Article 22 for automated decision-making)

---

### Q5: What were the most surprising findings in your EDA?

**Answer:** Three findings defied naive expectations:

1. **Zero-balance customers churn LESS than high-balance customers.** Counter-intuitive — one might expect high-balance customers to be stickier. But the data reveals that high-balance + inactive = flight risk. These customers are holding assets while evaluating competitors.

2. **3–4 product customers churn MORE than 1-product customers.** The expected "more products = more loyalty" hypothesis fails. Customers with 3–4 products exhibit >80% churn, suggesting the bank is force-cross-selling or that product complexity is causing dissatisfaction.

3. **Germany's churn rate (32%) vs Spain (17%).** The 15-percentage-point gap between geographies sharing the same product set suggests market-specific competitive dynamics, not product issues — requiring a geography-specific retention strategy rather than a portfolio-wide fix.

---

### Q6: How do you validate that your engineered features actually add value?

**Answer:** Three validation approaches:

1. **SHAP Importance:** Engineered features collectively account for ~22% of total SHAP importance (FCI at 7.09%, RSI at 5.43%, EngagementScore at 4.82%), confirming they capture signal beyond the raw features

2. **Ablation Study:** Retrained the LightGBM model on raw features only vs raw + engineered features. The engineered feature set improved ROC-AUC by ~8 percentage points (0.783 → 0.861)

3. **Business Validation:** Presented feature definitions to domain experts (relationship managers) who confirmed that Engagement Score and Loyalty Score aligned with their intuitive understanding of at-risk customers — critical for adoption in a real banking environment

---

### Q7: What are the limitations of this model?

**Answer:** I would disclose four key limitations to stakeholders:

1. **Static Snapshot:** The model uses point-in-time features. Banking churn is a dynamic process; ideally, we'd use rolling time-series features (e.g., 3-month engagement trend) to capture trajectory, not just current state

2. **No Transaction History:** The dataset lacks transaction frequency and channel usage data. A customer might have the same balance but very different product usage patterns — a critical signal we're missing

3. **Recall-Precision Trade-off:** At the current threshold (0.5), recall is 0.597 — meaning we miss ~40% of churners. Lowering the threshold improves recall but increases false positives (over-alerting frontline staff). The optimal threshold depends on the relative cost of each error type

4. **Temporal Stability:** The model was trained on a single time period. Economic changes, competitive moves, or product launches can shift the underlying churn dynamics, requiring continuous monitoring and retraining

---

## 🔷 Technical Skills Demonstrated

| Skill Category | Technologies Used |
|---------------|------------------|
| Programming | Python 3.11, OOP, Modular Architecture |
| Data Engineering | Pandas, PyArrow, Pipeline design, Schema validation |
| Machine Learning | Scikit-learn, XGBoost, LightGBM, imbalanced-learn |
| Explainable AI | SHAP, Feature Importance, Model Comparison |
| Statistics | Hypothesis testing, Correlation analysis, Distribution analysis |
| Visualisation | Matplotlib, Seaborn, Plotly (interactive) |
| Dashboarding | Streamlit, Custom CSS, Multi-tab layout |
| Deployment | Docker, Streamlit Cloud, REST API design |
| Domain Knowledge | Banking KPIs, Churn analysis, Retention strategy, CRM |

---

*This project demonstrates industry-level data science skills across the full analytics lifecycle: from raw data to deployed production system with executive reporting.*
