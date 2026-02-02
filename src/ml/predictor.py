"""Model inference and risk scoring."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config.settings import settings


class RiskPredictor:
    """Load trained model and make risk predictions."""

    def __init__(self, model_dir: Path | None = None):
        """
        Initialize the predictor.

        Args:
            model_dir: Directory containing model files. Defaults to settings.
        """
        self.model_dir = model_dir or settings.models_dir
        self.model: XGBClassifier | None = None
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] | None = None
        self._loaded = False

    def load(self) -> None:
        """
        Load the trained model, scaler, and feature names.

        Raises:
            FileNotFoundError: If model files don't exist.
        """
        model_path = self.model_dir / settings.model_name
        scaler_path = self.model_dir / settings.scaler_name
        features_path = self.model_dir / settings.feature_names_file

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please run 'python scripts/train_model.py' first."
            )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(features_path)
        self._loaded = True

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if not self._loaded:
            self.load()

    def predict_risk(self, features: pd.DataFrame) -> tuple[int, float]:
        """
        Predict risk score for given features.

        Args:
            features: DataFrame with feature values (single row).

        Returns:
            Tuple of (risk_score 0-100, raw_probability).
        """
        self._ensure_loaded()

        # Align features with training columns
        features_aligned = self._align_features(features)

        # Scale features
        features_scaled = self.scaler.transform(features_aligned)

        # Get probability of fraud
        probability = self.model.predict_proba(features_scaled)[0, 1]

        # Convert to 0-100 risk score
        risk_score = int(round(probability * 100))

        return risk_score, float(probability)

    def _align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Align input features with expected training features.

        Args:
            features: Input feature DataFrame.

        Returns:
            DataFrame with columns matching training data.
        """
        # Create DataFrame with all expected features, defaulting to 0
        aligned = pd.DataFrame(0, index=[0], columns=self.feature_names)

        # Copy matching columns
        for col in features.columns:
            if col in aligned.columns:
                aligned[col] = features[col].values[0]

        return aligned

    def get_risk_level(self, risk_score: int) -> str:
        """
        Get risk level category from score.

        Args:
            risk_score: Score from 0-100.

        Returns:
            Risk level string.
        """
        if risk_score >= 70:
            return "High Risk"
        elif risk_score >= 40:
            return "Medium Risk"
        else:
            return "Low Risk"

    def get_risk_color(self, risk_score: int) -> str:
        """
        Get color for risk visualization.

        Args:
            risk_score: Score from 0-100.

        Returns:
            Color hex code.
        """
        if risk_score >= 70:
            return "#FF4444"  # Red
        elif risk_score >= 40:
            return "#FFAA00"  # Orange/Yellow
        else:
            return "#44AA44"  # Green
