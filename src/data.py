"""Data loading and preprocessing utilities."""
from pathlib import Path
from typing import Tuple

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"


def load_competition_data(competition: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, and sample submission for a competition."""
    comp_dir = DATA_DIR / competition

    train = pd.read_csv(comp_dir / "train.csv")
    test = pd.read_csv(comp_dir / "test.csv")
    sample_sub = pd.read_csv(comp_dir / "sample_submission.csv")

    return train, test, sample_sub


def save_submission(
    submission: pd.DataFrame,
    competition: str,
    name: str = "submission"
) -> Path:
    """Save a submission CSV."""
    sub_dir = SUBMISSIONS_DIR / competition
    sub_dir.mkdir(parents=True, exist_ok=True)

    path = sub_dir / f"{name}.csv"
    submission.to_csv(path, index=False)
    print(f"Saved submission to {path}")
    return path


def get_feature_types(df: pd.DataFrame) -> dict:
    """Identify numerical and categorical columns."""
    numerical = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()

    return {"numerical": numerical, "categorical": categorical}
