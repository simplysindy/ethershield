"""Unit tests for Ethereum address validators."""

import pytest

from src.utils.validators import is_valid_eth_address, normalize_address


class TestIsValidEthAddress:
    """Tests for is_valid_eth_address function."""

    def test_valid_address_lowercase(self):
        """Test valid lowercase address."""
        assert is_valid_eth_address("0xd8da6bf26964af9d7eed9e03e53415d37aa96045")

    def test_valid_address_uppercase(self):
        """Test valid uppercase address."""
        assert is_valid_eth_address("0xD8DA6BF26964AF9D7EED9E03E53415D37AA96045")

    def test_valid_address_mixed_case(self):
        """Test valid mixed case address (checksum format)."""
        assert is_valid_eth_address("0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045")

    def test_invalid_address_too_short(self):
        """Test address that's too short."""
        assert not is_valid_eth_address("0xd8da6bf26964af9d7eed9e03e53415d37aa9604")

    def test_invalid_address_too_long(self):
        """Test address that's too long."""
        assert not is_valid_eth_address("0xd8da6bf26964af9d7eed9e03e53415d37aa960451")

    def test_invalid_address_no_prefix(self):
        """Test address without 0x prefix."""
        assert not is_valid_eth_address("d8da6bf26964af9d7eed9e03e53415d37aa96045")

    def test_invalid_address_wrong_prefix(self):
        """Test address with wrong prefix."""
        assert not is_valid_eth_address("1xd8da6bf26964af9d7eed9e03e53415d37aa96045")

    def test_invalid_address_non_hex(self):
        """Test address with non-hex characters."""
        assert not is_valid_eth_address("0xd8da6bf26964af9d7eed9e03e53415d37aa9604g")

    def test_empty_address(self):
        """Test empty string."""
        assert not is_valid_eth_address("")

    def test_none_address(self):
        """Test None value."""
        assert not is_valid_eth_address(None)


class TestNormalizeAddress:
    """Tests for normalize_address function."""

    def test_normalize_uppercase(self):
        """Test normalizing uppercase address."""
        result = normalize_address("0xD8DA6BF26964AF9D7EED9E03E53415D37AA96045")
        assert result == "0xd8da6bf26964af9d7eed9e03e53415d37aa96045"

    def test_normalize_mixed_case(self):
        """Test normalizing mixed case address."""
        result = normalize_address("0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045")
        assert result == "0xd8da6bf26964af9d7eed9e03e53415d37aa96045"

    def test_normalize_already_lowercase(self):
        """Test already lowercase address."""
        result = normalize_address("0xd8da6bf26964af9d7eed9e03e53415d37aa96045")
        assert result == "0xd8da6bf26964af9d7eed9e03e53415d37aa96045"

    def test_normalize_invalid_raises(self):
        """Test that invalid address raises ValueError."""
        with pytest.raises(ValueError, match="Invalid Ethereum address"):
            normalize_address("invalid")

    def test_normalize_empty_raises(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid Ethereum address"):
            normalize_address("")
