"""Ethereum address validation utilities."""

import re


def is_valid_eth_address(address: str) -> bool:
    """
    Validate an Ethereum address format.

    Args:
        address: The address string to validate.

    Returns:
        True if valid Ethereum address format, False otherwise.
    """
    if not address:
        return False

    # Check basic format: 0x followed by 40 hex characters
    pattern = r"^0x[a-fA-F0-9]{40}$"
    return bool(re.match(pattern, address))


def normalize_address(address: str) -> str:
    """
    Normalize an Ethereum address to lowercase with 0x prefix.

    Args:
        address: The address to normalize.

    Returns:
        Normalized address string.

    Raises:
        ValueError: If the address is invalid.
    """
    if not is_valid_eth_address(address):
        raise ValueError(f"Invalid Ethereum address: {address}")

    return address.lower()
