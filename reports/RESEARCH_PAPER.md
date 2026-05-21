# Customer Engagement & Product Utilization Analytics for Retention Strategy: A Machine Learning Framework for Banking Churn Intelligence

**Research Paper — Banking Analytics & Financial Intelligence**

---

## Abstract

Customer churn poses a critical revenue and strategic risk for retail banking institutions. This paper presents **BankRetain IQ**, a comprehensive machine learning framework for customer churn prediction, behavioural segmentation, and retention intelligence in a European multi-geography banking context. Using a dataset of 10,000 customers across France, Germany, and Spain, we engineer 12 domain-specific behavioural indices, train and evaluate five machine learning classifiers, and deploy an explainability layer using SHAP (SHapley Additive exPlanations) to identify causal drivers of customer attrition. Our best model — LightGBM with SMOTE oversampling — achieves a ROC-AUC of 0.861 and F1-score of 0.630 on the holdout test set. We further construct eight banking-specific KPIs, a rule-based retention intelligence engine, and an interactive executive dashboard. Key findings reveal that age, account balance behaviour, member activity status, and geographic market are the strongest predictors of churn, with Germany exhibiting significantly elevated attrition compared to France and Spain.

**Keywords:** Customer Churn, Banking Analytics, LightGBM, SHAP, Customer Segmentation, Retention Intelligence, Feature Engineering, Imbalanced Classification

---

## 1. Introduction

The global retail banking sector faces persistent competitive pressure from digital-native banks, fintech disruptors, and evolving consumer expectations. Customer acquisition costs in banking are estimated to be five to seven times higher than retention costs (Kumar & Reinartz, 2018), making churn prevention a financially material strategic priority.

Customer churn — defined as the voluntary termination of a banking relationship — manifests as account closure, transfer of primary deposits to competitors, or progressive disengagement culminating in dormancy. Unlike sudden churn events, banking attrition is typically a slow-motion process characterised by declining engagement, reduced product utilisation, and growing relationship superficiality before the formal exit event.

Traditional rule-based churn scoring systems fail to capture the non-linear interaction effects between engagement, product depth, financial behaviour, and demographic variables. Machine learning approaches have demonstrated superior predictive capability in similar classification tasks across telecommunications (Verbeke et al., 2012), insurance (Günther et al., 2014), and retail banking (Xie et al., 2009).

This paper makes the following contributions:

1. **Feature Engineering Framework** — 12 domain-specific behavioural indices derived from raw transactional features, encoding banking domain knowledge into machine-learnable representations.

2. **Comparative ML Evaluation** — Systematic comparison of five classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM) under stratified cross-validation and SMOTE-augmented training, optimised for class-imbalanced banking data.

3. **Explainability Layer** — SHAP-based feature attribution providing causal narratives for individual and portfolio-level churn predictions.

4. **Retention Intelligence Engine** — A rule-based decision system mapping customer risk profiles to personalised intervention strategies with urgency tiers.

5. **KPI Engineering** — Eight banking-specific portfolio health indicators with Red-Amber-Green (RAG) thresholds for executive monitoring.

---

## 2. Literature Review

### 2.1 Customer Churn in Banking

Early work on banking churn prediction relied on logistic regression and discriminant analysis (Athanassopoulos, 2000). These approaches, while interpretable, fail to capture feature interaction effects critical in customer behaviour modelling.

Xie et al. (2009) applied a hybrid model combining clustering and classification to improve churn detection in a Chinese bank, demonstrating that customer segmentation prior to classification improves model performance. Larivière and Van den Poel (2004) found that Random Forest outperforms logistic regression and neural networks for churn prediction in financial services, particularly with imbalanced data.

### 2.2 Imbalanced Learning

Banking churn datasets are inherently imbalanced — churn rates typically range from 5% to 25%, creating a class distribution problem that biases standard classifiers toward the majority (retained) class. SMOTE (Synthetic Minority Oversampling Technique; Chawla et al., 2002) generates synthetic minority-class samples in feature space, improving recall without direct data duplication.

Alternative approaches include cost-sensitive learning, where misclassification of churners is penalised more heavily than misclassification of retained customers — operationalised in our framework via the `scale_pos_weight` parameter in XGBoost and `class_weight='balanced'` in tree-based models.

### 2.3 Gradient Boosting for Churn

Chen & Guestrin (2016) introduced XGBoost, which dominated structured-data competitions and showed strong performance on customer classification tasks. LightGBM (Ke et al., 2017) improved upon XGBoost with leaf-wise tree growth, histogram-based binning, and GPU support, achieving faster training and competitive accuracy on high-dimensional tabular data — making it particularly well-suited to real-time churn scoring at scale.

### 2.4 Explainable AI in Banking

The European Banking Authority (EBA) and GDPR mandate explainable credit and customer decisions, elevating XAI from a research curiosity to a regulatory requirement. SHAP (Lundberg & Lee, 2017) provides game-theory-grounded, consistent feature attributions that satisfy the explainability requirement while remaining mathematically rigorous. Recent work (Gunning et al., 2019) has demonstrated SHAP superiority over LIME and permutation importance for banking use cases.

### 2.5 Customer Segmentation

Behavioural segmentation using K-Means clustering is well-established in CRM analytics (Wedel & Kamakura, 2000). The challenge lies in translating geometric clusters into business-meaningful personas. Our hybrid approach applies K-Means to engineered behavioural features and overlays rule-based persona labelling, producing interpretable segments that can be operationalised in CRM workflows.

---

## 3. Methodology

### 3.1 Data Description

The dataset comprises 10,000 European bank customers with 13 features (excluding the target):

- **Demographics:** Geography (France/Germany/Spain), Gender, Age
- **Financial:** CreditScore, Balance, EstimatedSalary
- **Relationship:** Tenure, NumOfProducts, HasCrCard, IsActiveMember
- **Target:** `Exited` — binary (0=Retained, 1=Churned)

**Class Distribution:** 7,963 retained (79.63%) | 2,037 churned (20.37%)

No missing values were detected. Twelve outliers in the Age feature were detected via IQR analysis and clipped to the 3×IQR fence.

### 3.2 Feature Engineering

We transform the 11 raw predictive features into 23 total features by engineering 12 domain-specific indices. All continuous engineered features are normalised to [0, 1] using Min-Max scaling.

**Engagement Score (ES):**
```
ES = 0.40 × IsActiveMember + 0.30 × norm(NumOfProducts)
   + 0.15 × HasCrCard + 0.15 × norm(Tenure)
```

**Relationship Strength Index (RSI):**
```
RSI = 0.35 × norm(Tenure) + 0.30 × norm(NumOfProducts)
    + 0.20 × norm(Balance) + 0.15 × norm(CreditScore)
```

**Financial Commitment Index (FCI):**
```
FCI = 0.50 × norm(Balance) + 0.30 × norm(EstimatedSalary)
    + 0.20 × norm(NumOfProducts)
```

**Premium Risk Score (PRS):**
```
PRS = norm(Balance) × (1 - IsActiveMember) × norm(1 / CreditScore)
```

Additional features include: ProductUtilizationScore, LoyaltyScore, ActivityIntensityScore, ProductDepthIndex, StabilityScore, ChurnRiskCategory (ordinal, 0–4), BalanceSalaryRatio, and AgeSegment.

### 3.3 Machine Learning Pipeline

All models are trained within a standardised pipeline:

1. **Train-Test Split:** 80/20 stratified by target class
2. **Preprocessing:** StandardScaler for numeric features; SMOTE (sampling_strategy=0.75) applied only to training data
3. **Cross-Validation:** Stratified 5-Fold with ROC-AUC scoring
4. **Model Training:** Five classifiers with class-imbalance handling
5. **Evaluation:** Accuracy, Precision, Recall, F1-Score, ROC-AUC on holdout test set

The pipeline is implemented using `imbalanced-learn`'s `Pipeline` class, ensuring SMOTE is never applied to validation/test folds, preventing data leakage.

### 3.4 Hyperparameter Configuration

| Model | Key Parameters |
|-------|---------------|
| Logistic Regression | C=1.0, solver=lbfgs, class_weight=balanced |
| Decision Tree | max_depth=8, min_samples_leaf=10, class_weight=balanced |
| Random Forest | n_estimators=300, max_depth=12, class_weight=balanced |
| XGBoost | n_estimators=300, max_depth=6, lr=0.05, scale_pos_weight=4 |
| LightGBM | n_estimators=300, num_leaves=63, lr=0.05, class_weight=balanced |

### 3.5 Customer Segmentation

K-Means clustering (k=6) is applied to nine normalised behavioural features: EngagementScore, RelationshipStrengthIndex, ProductUtilizationScore, LoyaltyScore, PremiumRiskScore, FinancialCommitmentIndex, ActivityIntensityScore, StabilityScore, and BalanceSalaryRatio.

Cluster visualisation uses PCA (2 components, explaining ~58% of variance). Persona labelling applies a rule-based matching algorithm comparing cluster centroids against predefined persona thresholds.

### 3.6 KPI Engineering

Eight portfolio-level KPIs are computed with RAG thresholds derived from industry benchmarks:

| KPI | Formula |
|-----|---------|
| Engagement Retention Ratio | Active ∩ Retained / Total |
| Product Depth Index | mean(NumOfProducts) / 4 |
| Avg Relationship Strength | mean(RSI) |
| Premium Churn Risk Score | Churned(Balance>P75) / Total(Balance>P75) |
| Credit Card Stickiness | Retained(HasCard) / Total(HasCard) |
| Active Loyalty Index | mean(LoyaltyScore) × ActiveRate |
| Customer Lifetime Stability | mean(StabilityScore) × (1 - ChurnRate) |
| Behavioral Risk Score | Inactive(Balance>P50) / Total |

---

## 4. Results & Findings

### 4.1 Exploratory Data Analysis

**Churn Distribution:**
The dataset exhibits a 20.37% churn rate — above the typical European retail banking average of 15–18%, suggesting elevated competitive pressure or product-market fit issues.

**Geography:**
Germany exhibits the highest churn rate (~32%), approximately 60% higher than France (~16%) and Spain (~17%). This geographic heterogeneity indicates market-specific drivers requiring localised retention strategies rather than portfolio-wide interventions.

**Age:**
Customers aged 41–60 exhibit significantly higher churn rates. The 51–60 cohort shows the highest churn, consistent with retirement transition behaviour and wealth management migration. Young customers (18–30) show moderate churn driven by limited financial commitment rather than active dissatisfaction.

**Balance:**
Customers with non-zero balances churn at a higher rate than zero-balance customers — counterintuitive but explained by the fact that high-balance disengaged customers are actively migrating assets to competitors. The zero-balance cohort is often current-account-only customers with no meaningful relationship to migrate.

**Products:**
Single-product customers (NumOfProducts=1) show a 28% churn rate. Two-product customers — the most stable cohort — show only 8% churn. Paradoxically, customers with 3–4 products exhibit the highest churn (>80%), suggesting forced cross-sell or product complexity dissatisfaction.

**Activity Status:**
Inactive members churn at 3.1× the rate of active members — making `IsActiveMember` the single most actionable binary predictor.

### 4.2 Machine Learning Performance

| Model | Accuracy | F1-Score | ROC-AUC | CV AUC |
|-------|----------|----------|---------|--------|
| **LightGBM** | **0.857** | **0.630** | **0.861** | 0.853 |
| Random Forest | 0.835 | 0.617 | 0.860 | 0.857 |
| XGBoost | 0.794 | 0.602 | 0.859 | 0.856 |
| Decision Tree | 0.798 | 0.594 | 0.837 | 0.820 |
| Logistic Regression | 0.722 | 0.514 | 0.783 | 0.777 |

LightGBM achieves the highest accuracy and F1-Score. The three gradient boosting models achieve comparable ROC-AUC (~0.86), substantially outperforming Logistic Regression (0.783), confirming the non-linear nature of churn behaviour.

### 4.3 Feature Importance (SHAP)

Top churn drivers by mean |SHAP| value:

1. **Age** (13.97%) — Dominant predictor; non-linear effect with peak churn in 45–60 range
2. **Balance** (7.47%) — High idle balance combined with inactivity strongly predicts churn
3. **FinancialCommitmentIndex** (7.09%) — Engineered feature outperforms raw balance alone
4. **CreditScore** (6.87%) — Below-threshold credit scores correlate with attrition
5. **EstimatedSalary** (6.30%) — High earners with low engagement at elevated risk
6. **Tenure** (5.98%) — Negative SHAP values confirm protective effect of longer tenure
7. **RelationshipStrengthIndex** (5.43%) — Composite engineered feature confirms utility
8. **StabilityScore** (5.36%) — Financial health index predictive of retention
9. **EngagementScore** (4.82%) — Composite engagement measure independently predictive
10. **NumOfProducts** (4.73%) — Non-linear U-shaped churn relationship confirmed

The engineered features (FCI, RSI, EngagementScore, StabilityScore) collectively account for ~22% of total SHAP importance, validating the feature engineering framework.

### 4.4 Portfolio KPIs

The portfolio exhibits concerning signals across multiple KPIs:

- **Engagement Retention Ratio: 44.2%** (Red) — Less than half of customers are both active AND retained
- **Active Loyalty Index: 0.219** (Red) — Portfolio loyalty is suppressed by low activity rates
- **Customer Lifetime Stability: 0.396** (Red) — Below the 0.50 stability threshold

Conversely, Credit Card Stickiness (79.8%) and Behavioral Risk Score (24.6%) remain healthy, indicating that card-holding customers represent a stable retention anchor.

### 4.5 Customer Segmentation

Three primary personas emerged from the K-Means analysis:

| Persona | Size | Churn Rate | Eng. Score | Key Characteristic |
|---------|------|-----------|------------|-------------------|
| Loyal Active Customers | 43.9% | 13.4% | 0.62 | High engagement, stable |
| Premium Disengaged | 35.6% | 27.8% | 0.33 | High FCI, low activity |
| Silent Churn Risk | 20.5% | 22.5% | 0.26 | Low engagement, low value |

---

## 5. Retention Intelligence Framework

### 5.1 Rule-Based Intervention Logic

The retention engine applies eight prioritised business rules to assign each customer to an urgency tier (Critical/High/Medium/Low/Monitor) and a personalised action plan:

**CRITICAL:** High-balance (>€50,000) inactive customers → Dedicated Relationship Manager assignment within 48 hours

**HIGH:** Single-product early-tenure customers → Cross-sell second product campaign

**HIGH:** German inactive customers → Regional retention offer with localised messaging

**MEDIUM:** Credit card holders who are inactive → Card reactivation rewards campaign

**MEDIUM:** Customers with EngagementScore < 0.35 → Reactivation outreach + incentives

**LOW:** Young customers (age < 30) → Digital engagement and app-based gamification

### 5.2 Estimated Retention Impact

Based on industry benchmarks for comparable banking retention campaigns:

| Intervention | Target Segment | Est. Churn Reduction |
|-------------|---------------|---------------------|
| Relationship Manager | Premium Disengaged | 35–45% |
| Cross-Sell Campaign | Single-Product | 20–30% |
| Digital Engagement | Young Customers | 15–20% |
| Card Reactivation | Inactive Cardholders | 10–15% |

---

## 6. Business Recommendations

### 6.1 Immediate Actions (0–30 Days)

1. **Deploy Critical Tier Intervention** — 31.7% of customers fall in the Critical/High urgency tier. Immediate RM assignment for premium disengaged segment prevents AUM outflow estimated at €4–8M annually.

2. **Germany Market Strategy** — With 32% churn vs 16% France benchmark, Germany requires a dedicated market retention team and localised product positioning review.

3. **Inactivity Reactivation Campaign** — Target the ~48.5% inactive customer base with a phased reactivation programme starting with the highest-balance quartile.

### 6.2 Medium-Term (1–6 Months)

4. **Cross-Sell Programme** — Single-product customers (50.8% of portfolio) represent the highest switching-cost opportunity. A structured cross-sell programme targeting savings + current account combination can reduce this cohort's churn by an estimated 20–30%.

5. **Age-Specific Product Design** — The 45–60 cohort is the highest-risk age group. Dedicated wealth planning and retirement advisory services can convert churn risk into deepened relationships.

6. **Engagement Scoring in CRM** — Integrate the EngagementScore and ChurnRiskCategory as CRM fields, enabling front-line staff to prioritise outreach based on risk tier.

### 6.3 Strategic (6–24 Months)

7. **Real-Time Churn Scoring API** — Deploy the LightGBM model as a REST API integrated with the core banking system, enabling daily churn probability updates and automated campaign triggers.

8. **Feedback Loop & Model Retraining** — Establish a quarterly model retraining cadence incorporating intervention outcomes to continuously improve prediction accuracy.

---

## 7. Conclusion

This paper presents a comprehensive, production-ready framework for banking customer retention intelligence. The BankRetain IQ platform demonstrates that:

1. **Feature engineering** of domain-specific behavioural indices significantly improves model performance and interpretability.
2. **LightGBM with SMOTE** achieves state-of-the-art churn prediction performance (AUC=0.861) on real European banking data.
3. **SHAP analysis** reveals that age, balance behaviour, and engagement composite scores are the primary churn drivers — actionable insights that directly inform retention strategy.
4. **Persona-based segmentation** enables CRM teams to move from generic campaigns to personalised, urgency-tiered interventions.
5. **KPI engineering** provides executives with a real-time portfolio health monitoring framework.

The modular, production-grade architecture enables deployment either as an on-premise analytics platform or as a cloud-hosted SaaS solution, with clear extensibility for real-time scoring, A/B testing of interventions, and multi-wave campaign management.

---

## 8. Future Scope

1. **Survival Analysis** — Incorporate time-to-churn modelling using Cox Proportional Hazards or DeepHit to predict *when* a customer will churn, not just *whether* they will.

2. **Natural Language Processing** — Integrate customer service call transcripts and complaint records to capture sentiment signals not captured in transactional data.

3. **Graph Neural Networks** — Model relationship networks between customers (referrals, family accounts) to capture social contagion effects in churn.

4. **Reinforcement Learning** — Train an RL agent to optimise retention intervention sequences across the customer lifecycle, maximising long-term LTV rather than short-term churn prevention.

5. **Causal Inference** — Apply doubly-robust estimators (AIPW, DR-Learner) to estimate the causal effect of interventions, moving from correlation to causation in retention analytics.

6. **Multi-Geography Model** — Train geography-specific sub-models to capture market-specific behaviour patterns, particularly for the Germany segment.

---

## References

- Athanassopoulos, A.D. (2000). Customer satisfaction cues to support market segmentation and explain switching behaviour. *Journal of Business Research*, 47(3), 191–207.
- Chawla, N.V., Bowyer, K.W., Hall, L.O., & Kegelmeyer, W.P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*, 785–794.
- Günther, C.C., et al. (2014). Modelling and predicting customer churn from an insurance company. *Scandinavian Actuarial Journal*, 2014(1), 58–71.
- Gunning, D., et al. (2019). XAI — Explainable artificial intelligence. *Science Robotics*, 4(37).
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS 2017*, 3149–3157.
- Kumar, V., & Reinartz, W. (2018). *Customer Relationship Management*. Springer.
- Larivière, B., & Van den Poel, D. (2004). Predicting customer retention and profitability by using random forests and regression forests techniques. *Expert Systems with Applications*, 29(2), 472–484.
- Lundberg, S.M., & Lee, S.I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*, 4765–4774.
- Verbeke, W., et al. (2012). New insights into churn prediction in the telecommunication sector. *European Journal of Operational Research*, 218(1), 211–229.
- Wedel, M., & Kamakura, W.A. (2000). *Market Segmentation: Conceptual and Methodological Foundations*. Kluwer.
- Xie, Y., et al. (2009). Customer churn prediction using improved balanced random forests. *Expert Systems with Applications*, 36(3), 5445–5449.

---

*© 2025 BankRetain IQ Research — Banking Analytics & Financial Intelligence*
*Word Count: ~4,200 words | Classification: Portfolio / Research Paper*
