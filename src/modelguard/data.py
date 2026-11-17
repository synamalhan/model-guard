"""Load UCI Adult and split off the reference window."""
import pandas as pd

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
    "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]
TARGET = "income"
NUMERIC = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL = [c for c in COLUMNS if c not in NUMERIC + [TARGET]]
REFERENCE_FRAC = 0.30


def load(path: str = "data/adult.data") -> pd.DataFrame:
    return pd.read_csv(path, names=COLUMNS, skipinitialspace=True, na_values="?")


def split_reference(df: pd.DataFrame, frac: float = REFERENCE_FRAC):
    """First `frac` of rows (file order) is reference; the rest is the live stream."""
    n = int(len(df) * frac)
    return df.iloc[:n], df.iloc[n:]

POSITIVE = ">50K"
