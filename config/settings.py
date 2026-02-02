"""Application settings using Pydantic BaseSettings."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Etherscan API
    etherscan_api_key: str = ""
    etherscan_base_url: str = "https://api.etherscan.io/v2/api"
    etherscan_chain_id: int = 1  # Ethereum mainnet
    etherscan_rate_limit: int = 5  # calls per second (free tier)

    # Paths
    project_root: Path = Path(__file__).parent.parent
    data_raw_dir: Path = project_root / "data" / "raw"
    data_processed_dir: Path = project_root / "data" / "processed"
    models_dir: Path = project_root / "models" / "trained"

    # Model settings
    model_name: str = "xgboost_fraud_detector.joblib"
    scaler_name: str = "feature_scaler.joblib"
    feature_names_file: str = "feature_names.joblib"

    @property
    def model_path(self) -> Path:
        return self.models_dir / self.model_name

    @property
    def scaler_path(self) -> Path:
        return self.models_dir / self.scaler_name

    @property
    def feature_names_path(self) -> Path:
        return self.models_dir / self.feature_names_file


settings = Settings()
