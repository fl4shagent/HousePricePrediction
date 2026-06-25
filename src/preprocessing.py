import re

import numpy as np
import pandas as pd


def parse_remaining_lease(series: pd.Series, month_col: pd.Series, lease_commence_col: pd.Series) -> pd.Series:
    """Parse remaining_lease into total months. Handles 3 formats:
    - null (2000-2014): compute from lease_commence_date + 99 - transaction_year
    - numeric string "70" (2015-2016): treat as years, convert to months
    - "X years Y months" (2017+): parse and convert
    """
    result = pd.Series(np.nan, index=series.index, dtype=float)

    for idx, val in series.items():
        if pd.isna(val):
            year = int(str(month_col[idx])[:4])
            month_num = int(str(month_col[idx])[5:7])
            lease_start = int(lease_commence_col[idx])
            remaining_years = lease_start + 99 - year
            remaining_months = 12 - month_num
            result[idx] = remaining_years * 12 + remaining_months
        elif isinstance(val, (int, float)):
            result[idx] = float(val) * 12
        elif isinstance(val, str):
            val = val.strip()
            if re.match(r"^\d+$", val):
                result[idx] = int(val) * 12
            else:
                years = 0
                months = 0
                y_match = re.search(r"(\d+)\s*year", val)
                m_match = re.search(r"(\d+)\s*month", val)
                if y_match:
                    years = int(y_match.group(1))
                if m_match:
                    months = int(m_match.group(1))
                result[idx] = years * 12 + months

    return result


def parse_storey_range(series: pd.Series) -> pd.Series:
    """Convert storey range "10 TO 12" to numeric median 11.0."""
    def to_median(s):
        try:
            parts = s.split(" TO ")
            return (int(parts[0]) + int(parts[1])) / 2
        except (ValueError, IndexError, AttributeError):
            return np.nan
    return series.apply(to_median)


def convert_yn_to_bool(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Convert Y/N string columns to boolean."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map({"Y": True, "N": False})
    return df


def extract_transaction_date(df: pd.DataFrame, month_col: str = "month") -> pd.DataFrame:
    """Extract year and month_numeric from transaction month column."""
    df = df.copy()
    df[month_col] = df[month_col].astype(str)
    df["transaction_year"] = df[month_col].str[:4].astype(int)
    df["transaction_month"] = df[month_col].str[5:7].astype(int)
    return df


def preprocess_raw_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing steps to raw transaction data."""
    df = df.copy()

    df = df.drop_duplicates()

    df = extract_transaction_date(df)

    df["remaining_lease_months"] = parse_remaining_lease(
        df["remaining_lease"], df["month"], df["lease_commence_date"]
    )

    df["storey_median"] = parse_storey_range(df["storey_range"])

    df = df.drop(columns=["remaining_lease", "storey_range", "lease_commence_date"], errors="ignore")

    return df
