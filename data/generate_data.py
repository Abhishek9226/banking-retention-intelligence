"""
=============================================================================
SYNTHETIC DATA GENERATOR — Banking Churn Dataset
=============================================================================
Generates a realistic 10 000-row banking churn dataset that mirrors the
real-world Kaggle Bank Customer Churn dataset structure.  Run this once to
populate data/raw/churn_data.csv before executing any pipeline module.
=============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)


def generate_banking_churn_dataset(n: int = 10_000) -> pd.DataFrame:
    """
    Generate a statistically realistic banking churn dataset.

    The underlying data-generating process encodes realistic domain knowledge:
      - Older customers churn more (lifecycle effects).
      - Inactive members churn ~3× more than active ones.
      - Customers with 3–4 products have elevated churn (product overwhelm).
      - Germany customers have higher base-churn than France / Spain.
      - High-balance, low-activity customers are a prime risk segment.

    Parameters
    ----------
    n : int
        Number of customer records to generate.

    Returns
    -------
    pd.DataFrame
        Raw dataset with the same schema as the Kaggle source.
    """

    # ── Demographics ──────────────────────────────────────────────────────
    customer_ids  = np.arange(15_600_001, 15_600_001 + n)
    surnames      = _random_surnames(n)
    credit_scores = np.clip(np.random.normal(650, 97, n).astype(int), 350, 850)
    geographies   = np.random.choice(
        ["France", "Germany", "Spain"], n, p=[0.50, 0.25, 0.25]
    )
    genders       = np.random.choice(["Male", "Female"], n, p=[0.545, 0.455])
    ages          = np.clip(np.random.normal(38, 10, n).astype(int), 18, 92)
    tenures       = np.random.randint(0, 11, n)

    # ── Financial behaviour ───────────────────────────────────────────────
    # ~30 % of customers hold zero balance (current-account only)
    zero_balance_mask = np.random.rand(n) < 0.30
    balances          = np.where(
        zero_balance_mask,
        0.0,
        np.clip(np.random.normal(76_000, 62_000, n), 0, 250_861),
    )
    num_products      = np.random.choice([1, 2, 3, 4], n, p=[0.50, 0.46, 0.03, 0.01])
    has_credit_card   = np.random.choice([0, 1], n, p=[0.29, 0.71])
    is_active_member  = np.random.choice([0, 1], n, p=[0.485, 0.515])
    estimated_salary  = np.clip(np.random.normal(100_000, 57_460, n), 11, 199_993)

    # ── Churn label — logistic model ─────────────────────────────────────
    logit = (
        -3.5
        + 0.025  * (ages - 38)                       # age effect
        + 0.004  * (300 - credit_scores) / 10        # poor credit ↑ churn
        - 0.15   * tenures                            # longer tenure ↓ churn
        + 0.0000025 * balances                        # high idle balance ↑ churn
        + 0.90   * (geographies == "Germany")        # Germany effect
        - 0.70   * is_active_member                  # active ↓ churn
        + 1.10   * (num_products >= 3)               # product overwhelm
        - 0.30   * has_credit_card
        + 0.10   * np.random.randn(n)                # noise
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    exited     = (np.random.rand(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "CustomerId":       customer_ids,
        "Surname":          surnames,
        "CreditScore":      credit_scores,
        "Geography":        geographies,
        "Gender":           genders,
        "Age":              ages,
        "Tenure":           tenures,
        "Balance":          np.round(balances, 2),
        "NumOfProducts":    num_products,
        "HasCrCard":        has_credit_card,
        "IsActiveMember":   is_active_member,
        "EstimatedSalary":  np.round(estimated_salary, 2),
        "Exited":           exited,
    })

    return df


def _random_surnames(n: int) -> np.ndarray:
    pool = [
        "Smith", "Jones", "Williams", "Taylor", "Brown", "Davies", "Evans",
        "Wilson", "Thomas", "Roberts", "Johnson", "Walker", "Wright", "Robinson",
        "Thompson", "White", "Hughes", "Edwards", "Green", "Hall", "Lewis",
        "Harris", "Clarke", "Patel", "Jackson", "Wood", "Turner", "Martin",
        "Cooper", "Hill", "Ward", "Morris", "Moore", "Clark", "Lee", "King",
        "Baker", "Harrison", "Morgan", "Allen", "James", "Scott", "Phillips",
        "Watson", "Davis", "Parker", "Price", "Bennett", "Young", "Griffiths",
    ]
    return np.random.choice(pool, n)


if __name__ == "__main__":
    output_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "churn_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_banking_churn_dataset(10_000)
    df.to_csv(output_path, index=False)

    churn_rate = df["Exited"].mean()
    print(f"✅  Dataset generated → {output_path}")
    print(f"   Rows        : {len(df):,}")
    print(f"   Churn rate  : {churn_rate:.1%}")
    print(f"   Columns     : {list(df.columns)}")
    print(df.head(3).to_string(index=False))
