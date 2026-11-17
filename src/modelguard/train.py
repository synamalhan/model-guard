"""Train the baseline on the reference window; save model and metrics."""
import json
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modelguard.data import (
    CATEGORICAL,
    NUMERIC,
    POSITIVE,
    TARGET,
    load,
    split_reference,
)

SEED = 42
MODEL_PATH = Path("models/baseline.joblib")
METRICS_PATH = Path("results/baseline.json")


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("cat", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore"),
        ), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, random_state=SEED))])


def train(df, seed: int = SEED):
    """Fit on 80% of the reference window; return (model, holdout accuracy, holdout F1)."""
    ref, _ = split_reference(df)
    X, y = ref[NUMERIC + CATEGORICAL], ref[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    model = build_pipeline().fit(Xtr, ytr)
    f1 = f1_score(yte, model.predict(Xte), pos_label=POSITIVE)
    return model, float(model.score(Xte, yte)), float(f1)


def main() -> None:
    df = load()
    model, acc, f1 = train(df)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    ref, _ = split_reference(df)
    METRICS_PATH.write_text(json.dumps(
        {"holdout_accuracy": acc, "holdout_f1": f1, "seed": SEED, "reference_rows": len(ref)}, indent=2) + "\n")
    print(f"saved {MODEL_PATH}; holdout accuracy {acc:.4f}, F1 {f1:.4f}")


if __name__ == "__main__":
    main()
