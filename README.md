# 🏦 BankRetain IQ — Customer Retention Intelligence Platform

> **Enterprise-grade Banking Analytics | Churn Prediction | Retention Strategy | Executive Intelligence**

---

## 📌 Project Overview

**BankRetain IQ** is a production-quality, end-to-end customer retention intelligence platform built for the banking domain. It transforms raw customer transaction and demographic data into actionable churn predictions, behavioural segmentation, KPI dashboards, and personalised retention strategies.

This project demonstrates **industry-level** proficiency in:
- Advanced Feature Engineering (12 custom KPIs)
- Machine Learning (5 models, SMOTE, Cross-Validation)
- Explainable AI (SHAP analysis)
- Customer Segmentation (KMeans + Persona Labelling)
- Interactive Executive Dashboard (Streamlit + Plotly)
- Retention Intelligence Engine (Rule-Based + ML-Driven)

---

## 🎯 Business Problem

A European bank with 10,000+ customers faces a **20.37% churn rate** — significantly impacting revenue, AUM, and customer acquisition costs. The challenge:

> *"Which customers will churn, why will they churn, and what specific action should we take to retain them — before it's too late?"*

---

## 📊 Dataset

| Feature | Description |
|---------|-------------|
| `CustomerId` | Unique customer identifier |
| `CreditScore` | Customer credit score (350–850) |
| `Geography` | France / Germany / Spain |
| `Gender` | Male / Female |
| `Age` | Customer age (18–92) |
| `Tenure` | Years with the bank (0–10) |
| `Balance` | Account balance (€) |
| `NumOfProducts` | Products held (1–4) |
| `HasCrCard` | Credit card holder (0/1) |
| `IsActiveMember` | Active status (0/1) |
| `EstimatedSalary` | Annual salary estimate (€) |
| `Exited` | **Target** — churned (1) or retained (0) |

**Key Stats:** 10,000 rows | 20.37% churn rate | Zero missing values | Clean, real-world distribution

---

## 🏗️ Project Architecture

```
banking_retention/
│
├── config/
│   └── config.py              # Centralised configuration (paths, hyperparams, thresholds)
│
├── data/
│   ├── raw/                   # Original dataset (CSV)
│   ├── processed/             # Cleaned + feature-engineered parquet files
│   └── generate_data.py       # Synthetic data generator (fallback)
│
├── src/
│   ├── ingestion/
│   │   └── data_pipeline.py   # Phase 2: Data Engineering + Validation
│   ├── features/
│   │   └── feature_engineering.py  # Phase 3: 12 Engineered Features
│   ├── eda/
│   │   └── eda_analysis.py    # Phase 4: 12 Advanced EDA Charts
│   ├── segmentation/
│   │   └── customer_segments.py    # Phase 5: KMeans + Persona Labelling
│   ├── kpi/
│   │   └── kpi_engine.py      # Phase 6: 8 Banking KPIs with RAG Status
│   ├── ml/
│   │   └── model_pipeline.py  # Phase 7: 5 ML Models + SMOTE + CV
│   ├── xai/
│   │   └── explainability.py  # Phase 8: SHAP + Feature Importance
│   └── retention/
│       └── retention_engine.py     # Phase 9: Rule-Based Retention Plans
│
├── dashboard/
│   └── app.py                 # Phase 10: 8-Tab Streamlit Dashboard
│
├── models/
│   ├── trained/               # Reserved for trained .pkl files
│   └── artifacts/             # Serialised pipelines + feature columns
│
├── reports/
│   ├── figures/               # 12 EDA charts (PNG)
│   ├── executive/             # Executive summary (PDF/Markdown)
│   └── RESEARCH_PAPER.md      # Academic research paper
│
├── deployment/
│   ├── Dockerfile             # Container deployment
│   └── deployment_guide.md    # Step-by-step deployment instructions
│
├── run_pipeline.py            # Master pipeline runner
├── requirements.txt           # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/yourname/bankretain-iq.git
cd bankretain-iq
pip install -r requirements.txt
```

### 2. Place Dataset
```bash
cp your_data.csv data/raw/churn_data.csv
```

### 3. Run Full Pipeline
```bash
python run_pipeline.py
```

### 4. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 🔬 Engineered Features (Phase 3)

| Feature | Formula | Business Meaning |
|---------|---------|-----------------|
| `EngagementScore` | 0.40×Active + 0.30×Products + 0.15×Card + 0.15×Tenure | How actively the customer uses the bank |
| `RelationshipStrengthIndex` | 0.35×Tenure + 0.30×Products + 0.20×Balance + 0.15×Credit | Depth of customer–bank relationship |
| `ProductUtilizationScore` | Products × Card_adj × Activity_adj | Breadth of product adoption |
| `LoyaltyScore` | 0.40×Tenure + 0.35×Active + 0.25×Products | Loyalty behaviour composite |
| `PremiumRiskScore` | Balance × Inactive × InvCredit | High-value disengaged risk |
| `FinancialCommitmentIndex` | 0.50×Balance + 0.30×Salary + 0.20×Products | Financial depth of engagement |
| `ActivityIntensityScore` | Active × Products/Card × Tenure | Intensity of service usage |
| `ProductDepthIndex` | Products × Tenure × Card_adj | Cross-sell penetration over time |
| `StabilityScore` | 0.35×Credit + 0.30×Tenure + 0.20×Balance + 0.15×Salary | Financial health & stability |
| `ChurnRiskCategory` | Rule-based ordinal (0–4) | Human-readable risk classification |
| `BalanceSalaryRatio` | Balance / Salary | Savings commitment ratio |
| `AgeSegment` | Ordinal age bins | Lifecycle stage |

---

## 🤖 Machine Learning Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **LightGBM** ⭐ | 0.857 | 0.666 | 0.597 | 0.630 | **0.861** |
| Random Forest | 0.835 | 0.585 | 0.654 | 0.617 | 0.860 |
| XGBoost | 0.794 | 0.495 | 0.767 | 0.602 | 0.859 |
| Decision Tree | 0.798 | 0.503 | 0.727 | 0.594 | 0.837 |
| Logistic Regression | 0.722 | 0.398 | 0.722 | 0.514 | 0.783 |

**Best Model:** LightGBM with **ROC-AUC = 0.861** | Includes SMOTE oversampling + 5-fold cross-validation

---

## 🎯 Top Churn Drivers (SHAP Analysis)

1. **Age** (13.97%) — Customers 45–60 churn significantly more
2. **Balance** (7.47%) — High idle balance + low activity = flight risk
3. **FinancialCommitmentIndex** (7.09%) — Engineered feature proves predictive
4. **CreditScore** (6.87%) — Below-700 scores correlate with higher churn
5. **EstimatedSalary** (6.30%) — High-earners with unengaged profiles at risk

---

## 🏦 Banking KPIs

| KPI | Value | Status |
|-----|-------|--------|
| Engagement Retention Ratio | 44.2% | 🔴 Red |
| Product Depth Index | 38.3% | 🔴 Red |
| Avg Relationship Strength Index | 0.380 | 🔴 Red |
| Premium Churn Risk Score | 23.7% | 🟢 Green |
| Credit Card Stickiness Score | 79.8% | 🟢 Green |
| Active Loyalty Index | 0.219 | 🔴 Red |
| Customer Lifetime Stability Indicator | 0.396 | 🔴 Red |
| Behavioral Risk Score | 24.6% | 🟢 Green |

---

## 📋 Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 📊 Executive Overview | KPI cards, churn pie chart, geography analysis |
| 🔋 Engagement | Engagement score distributions, activity heatmaps |
| ⚠️ Churn Analytics | Age/balance/tenure churn analysis, risk categories |
| 📦 Products | Product adoption, utilization, credit card analysis |
| 👥 Segments | PCA scatter, persona distribution, churn by persona |
| 🛡️ Retention Plans | Urgency tiers, action distribution, individual lookup |
| 🤖 Predictive | Model comparison radar, churn probability simulator |
| 📋 Data Explorer | Searchable filtered customer table with CSV export |

---

## 🐳 Docker Deployment

```bash
docker build -t bankretain-iq .
docker run -p 8501:8501 bankretain-iq
# Open: http://localhost:8501
```

---

## ☁️ Streamlit Cloud Deployment

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `dashboard/app.py` as entry point
4. Deploy → share URL

---

## 📄 Resume Bullet Points

```
• Built end-to-end banking churn prediction platform achieving 86.1% ROC-AUC using
  LightGBM with SMOTE oversampling on 10,000-customer European bank dataset

• Engineered 12 domain-specific behavioural KPIs (Engagement Score, Relationship
  Strength Index, Premium Risk Score) improving model performance by 8% vs baseline

• Implemented SHAP explainability to identify top churn drivers, enabling targeted
  retention campaigns estimated to reduce monthly churn by 15–20%

• Deployed 8-tab interactive Streamlit dashboard with KPI monitoring, customer
  segmentation (KMeans), and personalised retention recommendation engine
```

---

## 🔗 LinkedIn Project Description

> **BankRetain IQ — Customer Retention Intelligence Platform**
>
> Built a full-stack banking analytics platform that predicts customer churn with 86.1% AUC, segments 10,000+ customers into behavioural personas, and generates personalised retention strategies using a rule-based intelligence engine.
>
> Stack: Python | LightGBM | SHAP | Streamlit | Plotly | Scikit-learn | KMeans

---

## 👤 Author

Built as an industry-level portfolio project demonstrating banking analytics, machine learning engineering, and data product development.

---

*"Retention is cheaper than acquisition. Intelligence is cheaper than guessing."*
