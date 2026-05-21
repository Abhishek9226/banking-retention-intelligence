"""
=============================================================================
PHASE 2 — DATA ENGINEERING PIPELINE
src/ingestion/data_pipeline.py
=============================================================================
Production-grade data ingestion, validation, cleaning, and preprocessing
pipeline.  Every function is stateless and independently testable.
=============================================================================
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. DATA INGESTION
# =============================================================================

def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """
    Ingest raw CSV with dtype hints and basic sanity checks.

    Parameters
    ----------
    filepath : str | Path
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with optimised dtypes.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data not found at: {filepath}")

    dtype_map = {
        "CustomerId":      "int64",
        "CreditScore":     "int16",
        "Age":             "int8",
        "Tenure":          "int8",
        "NumOfProducts":   "int8",
        "HasCrCard":       "int8",
        "IsActiveMember":  "int8",
        "Exited":          "int8",
    }

    df = pd.read_csv(filepath, dtype=dtype_map)
    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns from {filepath.name}")
    return df


# =============================================================================
# 2. DATA VALIDATION
# =============================================================================

class DataValidator:
    """
    Validates the raw dataframe against a schema contract.

    Checks
    ------
    - Required columns presence
    - Datatype compliance
    - Value-range constraints
    - Binary variable integrity
    - Null / duplicate thresholds
    """

    SCHEMA = {
        "required_columns": [
            "CustomerId", "Surname", "CreditScore", "Geography", "Gender",
            "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
            "IsActiveMember", "EstimatedSalary", "Exited",
        ],
        "ranges": {
            "CreditScore":     (300, 850),
            "Age":             (18,  100),
            "Tenure":          (0,   10),
            "NumOfProducts":   (1,   4),
            "Balance":         (0,   1_000_000),
            "EstimatedSalary": (0,   500_000),
        },
        "binary_columns": ["HasCrCard", "IsActiveMember", "Exited"],
        "categorical_values": {
            "Geography": {"France", "Germany", "Spain"},
            "Gender":    {"Male", "Female"},
        },
        "null_threshold":      0.05,   # flag if > 5 % missing
        "duplicate_threshold": 0.01,   # flag if > 1 % duplicates
    }

    def __init__(self, df: pd.DataFrame):
        self.df      = df
        self.issues  : List[str] = []
        self.warnings: List[str] = []

    def validate(self) -> Dict:
        self._check_required_columns()
        self._check_null_values()
        self._check_duplicates()
        self._check_value_ranges()
        self._check_binary_columns()
        self._check_categorical_values()

        report = {
            "passed":   len(self.issues) == 0,
            "issues":   self.issues,
            "warnings": self.warnings,
            "shape":    self.df.shape,
        }

        if self.issues:
            logger.error(f"Validation FAILED — {len(self.issues)} issue(s) found")
        else:
            logger.info(f"Validation PASSED — {len(self.warnings)} warning(s)")

        return report

    def _check_required_columns(self):
        missing = set(self.SCHEMA["required_columns"]) - set(self.df.columns)
        if missing:
            self.issues.append(f"Missing columns: {missing}")

    def _check_null_values(self):
        null_pct = self.df.isnull().mean()
        for col, pct in null_pct[null_pct > 0].items():
            if pct > self.SCHEMA["null_threshold"]:
                self.issues.append(f"High nulls in {col}: {pct:.1%}")
            else:
                self.warnings.append(f"Minor nulls in {col}: {pct:.1%}")

    def _check_duplicates(self):
        dup_pct = self.df.duplicated().mean()
        if dup_pct > self.SCHEMA["duplicate_threshold"]:
            self.issues.append(f"High duplicate rate: {dup_pct:.1%}")
        elif dup_pct > 0:
            self.warnings.append(f"Minor duplicates: {dup_pct:.1%}")

    def _check_value_ranges(self):
        for col, (lo, hi) in self.SCHEMA["ranges"].items():
            if col not in self.df.columns:
                continue
            out_of_range = ((self.df[col] < lo) | (self.df[col] > hi)).sum()
            if out_of_range > 0:
                self.warnings.append(f"{out_of_range} out-of-range values in {col}")

    def _check_binary_columns(self):
        for col in self.SCHEMA["binary_columns"]:
            if col not in self.df.columns:
                continue
            unique_vals = set(self.df[col].dropna().unique())
            if not unique_vals.issubset({0, 1}):
                self.issues.append(f"{col} contains non-binary values: {unique_vals}")

    def _check_categorical_values(self):
        for col, expected in self.SCHEMA["categorical_values"].items():
            if col not in self.df.columns:
                continue
            actual = set(self.df[col].dropna().unique())
            unexpected = actual - expected
            if unexpected:
                self.warnings.append(f"Unexpected {col} values: {unexpected}")


# =============================================================================
# 3. DATA CLEANING
# =============================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning transformations and return a sanitised dataframe.

    Operations
    ----------
    1. Drop exact duplicate rows
    2. Impute missing values (median for numeric, mode for categorical)
    3. Clip outliers using the IQR fence (Tukey method)
    4. Standardise categorical strings
    5. Optimise memory via category / int dtypes

    Returns
    -------
    pd.DataFrame
        Clean, analysis-ready dataframe.
    """
    df = df.copy()
    original_rows = len(df)

    # ── 1. Duplicates ─────────────────────────────────────────────────────
    df.drop_duplicates(inplace=True)
    dropped = original_rows - len(df)
    if dropped:
        logger.info(f"Dropped {dropped} duplicate rows")

    # ── 2. Missing value imputation ───────────────────────────────────────
    numeric_cols    = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
            logger.info(f"Imputed {col} with median")

    for col in categorical_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
            logger.info(f"Imputed {col} with mode")

    # ── 3. Outlier capping (IQR fence) ────────────────────────────────────
    clip_cols = ["CreditScore", "Age", "Balance", "EstimatedSalary"]
    for col in clip_cols:
        if col not in df.columns:
            continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR    = Q3 - Q1
        lo, hi = Q1 - 3.0 * IQR, Q3 + 3.0 * IQR
        clipped = ((df[col] < lo) | (df[col] > hi)).sum()
        if clipped:
            df[col] = df[col].clip(lo, hi)
            logger.info(f"Clipped {clipped} outliers in {col}")

    # ── 4. String standardisation ─────────────────────────────────────────
    for col in ["Geography", "Gender"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()

    # ── 5. Dtype optimisation ─────────────────────────────────────────────
    for col in ["Geography", "Gender"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    for col in ["HasCrCard", "IsActiveMember", "Exited", "NumOfProducts", "Tenure"]:
        if col in df.columns:
            df[col] = df[col].astype("int8")

    logger.info(f"Cleaning complete — {len(df):,} rows retained")
    return df


# =============================================================================
# 4. OUTLIER ANALYSIS REPORT
# =============================================================================

def outlier_report(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compute per-column outlier statistics using Z-score and IQR methods.

    Returns
    -------
    pd.DataFrame
        Report with count and percentage of outliers per method.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    records = []
    for col in numeric_cols:
        series   = df[col].dropna()
        z_scores = np.abs(stats.zscore(series))
        Q1, Q3   = series.quantile(0.25), series.quantile(0.75)
        IQR      = Q3 - Q1

        records.append({
            "column":          col,
            "z_outliers":      (z_scores > 3).sum(),
            "z_outlier_pct":   f"{(z_scores > 3).mean():.2%}",
            "iqr_outliers":    ((series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)).sum(),
            "iqr_outlier_pct": f"{((series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)).mean():.2%}",
            "mean":            round(series.mean(), 2),
            "std":             round(series.std(), 2),
            "min":             series.min(),
            "max":             series.max(),
            "skewness":        round(series.skew(), 3),
        })

    return pd.DataFrame(records)


# =============================================================================
# 5. DATA QUALITY SUMMARY
# =============================================================================

def data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a comprehensive per-column data quality report."""
    records = []
    for col in df.columns:
        records.append({
            "column":        col,
            "dtype":         str(df[col].dtype),
            "null_count":    df[col].isnull().sum(),
            "null_pct":      f"{df[col].isnull().mean():.2%}",
            "unique_values": df[col].nunique(),
            "sample_values": str(df[col].dropna().unique()[:5].tolist()),
        })
    return pd.DataFrame(records)


# =============================================================================
# 6. PREPROCESSING PIPELINE (Scikit-learn compatible)
# =============================================================================

from sklearn.pipeline import Pipeline
from sklearn.compose   import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, MinMaxScaler
)
from sklearn.impute import SimpleImputer


def build_preprocessing_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    scaler: str = "standard",
) -> ColumnTransformer:
    """
    Build a reusable Scikit-learn preprocessing pipeline.

    Parameters
    ----------
    numeric_cols : list
        Names of numeric feature columns.
    categorical_cols : list
        Names of categorical feature columns.
    scaler : str
        'standard' for StandardScaler, 'minmax' for MinMaxScaler.

    Returns
    -------
    ColumnTransformer
        Fitted-ready preprocessor.
    """
    _scaler = StandardScaler() if scaler == "standard" else MinMaxScaler()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  _scaler),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer,  numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    return preprocessor


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from config.config import RAW_DATA_FILE, CLEAN_DATA_FILE

    # Load
    df_raw = load_raw_data(RAW_DATA_FILE)

    # Validate
    validator = DataValidator(df_raw)
    report    = validator.validate()
    print("\n=== VALIDATION REPORT ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    # Quality summary
    print("\n=== DATA QUALITY SUMMARY ===")
    print(data_quality_summary(df_raw).to_string(index=False))

    # Outlier report
    print("\n=== OUTLIER REPORT ===")
    numeric = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
    print(outlier_report(df_raw, numeric).to_string(index=False))

    # Clean
    df_clean = clean_data(df_raw)

    # Save
    CLEAN_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(CLEAN_DATA_FILE, index=False)
    print(f"\n✅  Clean data saved → {CLEAN_DATA_FILE}")
