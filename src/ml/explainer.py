"""SHAP-based model explanations."""

import numpy as np
import pandas as pd

from config.settings import settings


# Human-readable descriptions for features
FEATURE_DESCRIPTIONS = {
    "Avg min between sent tnx": "Average time between outgoing transactions",
    "Avg min between received tnx": "Average time between incoming transactions",
    "Time Diff between first and last (Mins)": "Total activity time span",
    "Sent tnx": "Number of outgoing transactions",
    "Received Tnx": "Number of incoming transactions",
    "Unique Received From Addresses": "Unique addresses that sent to this wallet",
    "Unique Sent To Addresses": "Unique addresses this wallet sent to",
    "avg val sent": "Average ETH value sent per transaction",
    "avg val received": "Average ETH value received per transaction",
    "total Ether sent": "Total ETH sent",
    "total ether received": "Total ETH received",
    "total ether balance": "Net ETH balance",
    " ERC20 total Ether received": "Total ERC20 tokens received",
    " ERC20 total ether sent": "Total ERC20 tokens sent",
    " ERC20 uniq sent addr": "Unique addresses for token sends",
    " ERC20 uniq rec addr": "Unique addresses for token receives",
    " ERC20 avg val rec": "Average token value received",
    " ERC20 avg val sent": "Average token value sent",
    " ERC20 uniq sent token name": "Variety of tokens sent",
    " ERC20 uniq rec token name": "Variety of tokens received",
}


class RiskExplainer:
    """Generate explanations for risk predictions using feature importance."""

    def __init__(self, model, feature_names: list[str]):
        """
        Initialize the explainer.

        Args:
            model: Trained XGBoost model.
            feature_names: List of feature names.
        """
        self.model = model
        self.feature_names = feature_names
        self._feature_importances = self._get_feature_importances()

    def _get_feature_importances(self) -> dict[str, float]:
        """Extract feature importances from XGBoost model."""
        try:
            # Get importance scores (gain-based)
            importance = self.model.get_booster().get_score(importance_type="gain")
            # Map feature indices to names
            result = {}
            for key, value in importance.items():
                # XGBoost uses 'f0', 'f1', etc. for feature names
                if key.startswith("f"):
                    idx = int(key[1:])
                    if idx < len(self.feature_names):
                        result[self.feature_names[idx]] = value
                else:
                    result[key] = value
            return result
        except Exception:
            # Fallback to equal importance
            return {name: 1.0 for name in self.feature_names}

    def explain_prediction(
        self,
        features: pd.DataFrame | np.ndarray,
        top_n: int = 5,
    ) -> list[dict]:
        """
        Generate explanation for a prediction.

        Args:
            features: Scaled feature DataFrame or array (single row).
            top_n: Number of top contributing factors to return.

        Returns:
            List of dicts with factor explanations.
        """
        # Convert to array if DataFrame
        if isinstance(features, pd.DataFrame):
            feature_values = features.values[0]
        else:
            feature_values = features[0] if features.ndim > 1 else features

        # Calculate impact as importance * normalized feature value
        impacts = []
        for i, name in enumerate(self.feature_names):
            importance = self._feature_importances.get(name, 0.0)
            value = float(feature_values[i]) if i < len(feature_values) else 0.0

            # Impact is proportional to importance and value magnitude
            # Positive values increase risk, negative decrease (simplified)
            impact = importance * (1 if value > 0 else -0.5) * min(abs(value), 1.0)
            impacts.append((name, impact, value, importance))

        # Sort by importance (not impact) to show most important features
        impacts.sort(key=lambda x: x[3], reverse=True)

        # Build explanations for top N
        explanations = []
        for name, impact, feature_value, importance in impacts[:top_n]:
            # Determine direction based on feature value relative to normal
            direction = "increases" if feature_value > 0 else "decreases"
            description = FEATURE_DESCRIPTIONS.get(name, name)

            explanations.append({
                "feature": name,
                "description": description,
                "value": float(feature_value),
                "impact": float(importance),
                "direction": direction,
                "explanation": self._generate_explanation(
                    description, feature_value, importance
                ),
            })

        return explanations

    def _generate_explanation(
        self, description: str, value: float, importance: float
    ) -> str:
        """
        Generate human-readable explanation for a feature.

        Args:
            description: Feature description.
            value: Feature value (scaled).
            importance: Feature importance score.

        Returns:
            Human-readable explanation string.
        """
        # Determine impact direction and intensity
        direction = "increases" if value > 0 else "decreases"

        # Normalize importance for intensity description
        max_importance = max(self._feature_importances.values()) if self._feature_importances else 1.0
        normalized = importance / max_importance if max_importance > 0 else 0

        if normalized > 0.5:
            intensity = "significantly"
        elif normalized > 0.2:
            intensity = "moderately"
        else:
            intensity = "slightly"

        return f"{description} {intensity} {direction} risk"

    def get_summary_plot_data(
        self, features: pd.DataFrame
    ) -> tuple[np.ndarray, list[str]]:
        """
        Get data for importance visualization.

        Args:
            features: Feature DataFrame.

        Returns:
            Tuple of (importance_values array, feature_names).
        """
        importances = np.array([
            self._feature_importances.get(name, 0.0)
            for name in self.feature_names
        ])
        return importances, self.feature_names
