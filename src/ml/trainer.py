"""XGBoost model training with SMOTE for fraud detection."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config.settings import settings


class FraudDetectorTrainer:
    """Train XGBoost model for Ethereum fraud detection."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ):
        """
        Initialize the trainer.

        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Boosting learning rate.
            random_state: Random seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.model: XGBClassifier | None = None
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] | None = None

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        apply_smote: bool = True,
    ) -> dict[str, float]:
        """
        Train the fraud detection model.

        Args:
            X: Feature DataFrame.
            y: Target Series (0 = legitimate, 1 = fraud).
            test_size: Proportion of data to use for testing.
            apply_smote: Whether to apply SMOTE for class balancing.

        Returns:
            Dictionary with training metrics.
        """
        self.feature_names = list(X.columns)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Apply SMOTE for class imbalance
        if apply_smote:
            smote = SMOTE(random_state=self.random_state)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

        # Train XGBoost
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric="logloss",
        )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        metrics = {
            "f1_score": f1_score(y_test, y_pred),
            "accuracy": self.model.score(X_test_scaled, y_test),
            "train_samples": len(X_train_scaled),
            "test_samples": len(X_test),
            "fraud_ratio_test": float(y_test.mean()),
        }

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

        return metrics

    def save(self, output_dir: Path | None = None) -> None:
        """
        Save trained model, scaler, and feature names.

        Args:
            output_dir: Directory to save files. Defaults to settings.models_dir.

        Raises:
            ValueError: If model hasn't been trained.
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be trained before saving")

        output_dir = output_dir or settings.models_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, output_dir / settings.model_name)
        joblib.dump(self.scaler, output_dir / settings.scaler_name)
        joblib.dump(self.feature_names, output_dir / settings.feature_names_file)

        print(f"\nModel saved to {output_dir}")

    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores.

        Raises:
            ValueError: If model hasn't been trained.
        """
        if self.model is None or self.feature_names is None:
            raise ValueError("Model must be trained first")

        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))
