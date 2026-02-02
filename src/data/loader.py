"""Data loader for Kaggle Ethereum Fraud Detection Dataset."""

from pathlib import Path

import pandas as pd

from config.settings import settings


def load_kaggle_dataset(file_path: Path | None = None) -> pd.DataFrame:
    """
    Load the Kaggle Ethereum Fraud Detection dataset.

    Args:
        file_path: Optional path to CSV file. Defaults to data/raw/transaction_dataset.csv

    Returns:
        DataFrame with the fraud detection data.

    Raises:
        FileNotFoundError: If the dataset file doesn't exist.
    """
    if file_path is None:
        file_path = settings.data_raw_dir / "transaction_dataset.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. "
            "Please download from Kaggle: "
            "https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset"
        )

    df = pd.read_csv(file_path, index_col=0)
    return df


def get_feature_columns() -> list[str]:
    """
    Get the list of feature column names used for training.

    Returns:
        List of feature column names.
    """
    return [
        "Avg min between sent tnx",
        "Avg min between received tnx",
        "Time Diff between first and last (Mins)",
        "Sent tnx",
        "Received Tnx",
        "Unique Received From Addresses",
        "Unique Sent To Addresses",
        "avg val sent",
        "avg val received",
        "total Ether sent",
        "total ether received",
        "total ether balance",
        " ERC20 total Ether received",
        " ERC20 total ether sent",
        " ERC20 uniq sent addr",
        " ERC20 uniq rec addr",
        " ERC20 avg val rec",
        " ERC20 avg val sent",
        " ERC20 uniq sent token name",
        " ERC20 uniq rec token name",
    ]


def get_target_column() -> str:
    """Get the target column name."""
    return "FLAG"


def prepare_training_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for model training.

    Args:
        df: Raw DataFrame from Kaggle dataset.

    Returns:
        Tuple of (features DataFrame, target Series).
    """
    feature_cols = get_feature_columns()
    target_col = get_target_column()

    # Filter to only existing columns
    available_features = [col for col in feature_cols if col in df.columns]

    X = df[available_features].copy()
    y = df[target_col].copy()

    # Handle missing values
    X = X.fillna(0)

    return X, y
