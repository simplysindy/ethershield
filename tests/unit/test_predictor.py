"""Unit tests for the risk predictor."""

import pytest
import numpy as np
import pandas as pd

from src.ml.predictor import RiskPredictor


class TestRiskPredictor:
    """Tests for RiskPredictor class."""

    def test_get_risk_level_low(self):
        """Test low risk level classification."""
        predictor = RiskPredictor()
        assert predictor.get_risk_level(0) == "Low Risk"
        assert predictor.get_risk_level(20) == "Low Risk"
        assert predictor.get_risk_level(39) == "Low Risk"

    def test_get_risk_level_medium(self):
        """Test medium risk level classification."""
        predictor = RiskPredictor()
        assert predictor.get_risk_level(40) == "Medium Risk"
        assert predictor.get_risk_level(55) == "Medium Risk"
        assert predictor.get_risk_level(69) == "Medium Risk"

    def test_get_risk_level_high(self):
        """Test high risk level classification."""
        predictor = RiskPredictor()
        assert predictor.get_risk_level(70) == "High Risk"
        assert predictor.get_risk_level(85) == "High Risk"
        assert predictor.get_risk_level(100) == "High Risk"

    def test_get_risk_color_low(self):
        """Test color for low risk."""
        predictor = RiskPredictor()
        assert predictor.get_risk_color(20) == "#44AA44"

    def test_get_risk_color_medium(self):
        """Test color for medium risk."""
        predictor = RiskPredictor()
        assert predictor.get_risk_color(50) == "#FFAA00"

    def test_get_risk_color_high(self):
        """Test color for high risk."""
        predictor = RiskPredictor()
        assert predictor.get_risk_color(80) == "#FF4444"

    def test_load_missing_model_raises(self):
        """Test that loading missing model raises FileNotFoundError."""
        from pathlib import Path
        predictor = RiskPredictor(model_dir=Path("/nonexistent/path"))
        with pytest.raises(FileNotFoundError, match="Model not found"):
            predictor.load()


class TestRiskPredictorAlignment:
    """Tests for feature alignment in RiskPredictor."""

    def test_align_features_fills_missing(self):
        """Test that missing features are filled with zeros."""
        predictor = RiskPredictor()
        predictor.feature_names = ["feature_a", "feature_b", "feature_c"]
        predictor._loaded = True

        # Input has only one feature
        input_df = pd.DataFrame({"feature_a": [5.0]})
        aligned = predictor._align_features(input_df)

        assert list(aligned.columns) == ["feature_a", "feature_b", "feature_c"]
        assert aligned["feature_a"].iloc[0] == 5.0
        assert aligned["feature_b"].iloc[0] == 0
        assert aligned["feature_c"].iloc[0] == 0

    def test_align_features_ignores_extra(self):
        """Test that extra features are ignored."""
        predictor = RiskPredictor()
        predictor.feature_names = ["feature_a", "feature_b"]
        predictor._loaded = True

        # Input has extra feature
        input_df = pd.DataFrame({
            "feature_a": [5.0],
            "feature_b": [10.0],
            "feature_extra": [99.0],
        })
        aligned = predictor._align_features(input_df)

        assert list(aligned.columns) == ["feature_a", "feature_b"]
        assert "feature_extra" not in aligned.columns
