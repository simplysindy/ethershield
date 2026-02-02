"""Unit tests for feature engineering functions."""

import pytest

from src.data.feature_engineering import (
    TransactionFeatures,
    compute_address_features,
    compute_time_features,
    compute_value_features,
)


class TestComputeTimeFeatures:
    """Tests for compute_time_features function."""

    def test_empty_timestamps(self):
        """Test with empty timestamp list."""
        avg_sent, avg_received, time_diff = compute_time_features([], [])
        assert avg_sent == 0.0
        assert avg_received == 0.0
        assert time_diff == 0.0

    def test_single_timestamp(self):
        """Test with single timestamp."""
        avg_sent, avg_received, time_diff = compute_time_features([1000], [True])
        assert avg_sent == 0.0
        assert avg_received == 0.0
        assert time_diff == 0.0

    def test_two_sent_transactions(self):
        """Test average time between two sent transactions."""
        # 120 seconds = 2 minutes apart
        timestamps = [1000, 1120]
        is_sent = [True, True]
        avg_sent, avg_received, time_diff = compute_time_features(timestamps, is_sent)
        assert avg_sent == 2.0  # 2 minutes
        assert avg_received == 0.0
        assert time_diff == 2.0

    def test_mixed_transactions(self):
        """Test with both sent and received transactions."""
        timestamps = [1000, 1060, 1120, 1180]  # 60 seconds apart each
        is_sent = [True, False, True, False]
        avg_sent, avg_received, time_diff = compute_time_features(timestamps, is_sent)
        assert avg_sent == 2.0  # 120 seconds / 60 = 2 minutes between sent
        assert avg_received == 2.0  # 120 seconds / 60 = 2 minutes between received
        assert time_diff == 3.0  # 180 seconds / 60 = 3 minutes total span


class TestComputeValueFeatures:
    """Tests for compute_value_features function."""

    def test_empty_values(self):
        """Test with empty values."""
        result = compute_value_features([], [])
        assert result == (0.0, 0.0, 0, 0, 0)

    def test_only_sent(self):
        """Test with only sent transactions."""
        values = [1.0, 2.0, 3.0]
        is_sent = [True, True, True]
        avg_sent, avg_received, total_sent, total_received, balance = compute_value_features(
            values, is_sent
        )
        assert avg_sent == 2.0
        assert avg_received == 0.0
        assert total_sent == 6.0
        assert total_received == 0.0
        assert balance == -6.0

    def test_only_received(self):
        """Test with only received transactions."""
        values = [1.0, 2.0, 3.0]
        is_sent = [False, False, False]
        avg_sent, avg_received, total_sent, total_received, balance = compute_value_features(
            values, is_sent
        )
        assert avg_sent == 0.0
        assert avg_received == 2.0
        assert total_sent == 0.0
        assert total_received == 6.0
        assert balance == 6.0

    def test_mixed_transactions(self):
        """Test with mixed sent/received."""
        values = [1.0, 2.0, 3.0, 4.0]
        is_sent = [True, False, True, False]
        avg_sent, avg_received, total_sent, total_received, balance = compute_value_features(
            values, is_sent
        )
        assert avg_sent == 2.0  # (1 + 3) / 2
        assert avg_received == 3.0  # (2 + 4) / 2
        assert total_sent == 4.0
        assert total_received == 6.0
        assert balance == 2.0


class TestComputeAddressFeatures:
    """Tests for compute_address_features function."""

    def test_empty_addresses(self):
        """Test with empty address lists."""
        unique_from, unique_to = compute_address_features([], [], "0x123")
        assert unique_from == 0
        assert unique_to == 0

    def test_excludes_wallet_address(self):
        """Test that wallet's own address is excluded."""
        from_addresses = ["0xabc", "0x123", "0xdef"]
        to_addresses = ["0x123", "0xghi", "0xjkl"]
        wallet = "0x123"

        unique_from, unique_to = compute_address_features(
            from_addresses, to_addresses, wallet
        )
        # Excludes 0x123 from counts
        assert unique_from == 2  # 0xabc, 0xdef
        assert unique_to == 2  # 0xghi, 0xjkl

    def test_unique_counts_deduplication(self):
        """Test that duplicate addresses are counted once."""
        from_addresses = ["0xabc", "0xabc", "0xdef"]
        to_addresses = ["0xghi", "0xghi", "0xghi"]
        wallet = "0x123"

        unique_from, unique_to = compute_address_features(
            from_addresses, to_addresses, wallet
        )
        assert unique_from == 2  # 0xabc, 0xdef
        assert unique_to == 1  # 0xghi

    def test_case_insensitive(self):
        """Test case-insensitive address comparison."""
        from_addresses = ["0xABC", "0xabc"]
        to_addresses = ["0xDEF"]
        wallet = "0x123"

        unique_from, unique_to = compute_address_features(
            from_addresses, to_addresses, wallet
        )
        assert unique_from == 1  # 0xabc counted once


class TestTransactionFeatures:
    """Tests for TransactionFeatures dataclass."""

    def test_default_values(self):
        """Test default feature values."""
        features = TransactionFeatures()
        assert features.sent_tnx == 0
        assert features.received_tnx == 0
        assert features.total_ether_balance == 0.0

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        features = TransactionFeatures(
            sent_tnx=10,
            received_tnx=5,
            total_ether_sent=100.0,
        )
        df = features.to_dataframe()

        assert len(df) == 1
        assert df["Sent tnx"].iloc[0] == 10
        assert df["Received Tnx"].iloc[0] == 5
        assert df["total Ether sent"].iloc[0] == 100.0
