"""Feature engineering for Ethereum transaction data."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TransactionFeatures:
    """Container for computed transaction features."""

    avg_min_between_sent_tnx: float = 0.0
    avg_min_between_received_tnx: float = 0.0
    time_diff_first_last_mins: float = 0.0
    sent_tnx: int = 0
    received_tnx: int = 0
    unique_received_from_addresses: int = 0
    unique_sent_to_addresses: int = 0
    avg_val_sent: float = 0.0
    avg_val_received: float = 0.0
    total_ether_sent: float = 0.0
    total_ether_received: float = 0.0
    total_ether_balance: float = 0.0
    erc20_total_ether_received: float = 0.0
    erc20_total_ether_sent: float = 0.0
    erc20_uniq_sent_addr: int = 0
    erc20_uniq_rec_addr: int = 0
    erc20_avg_val_rec: float = 0.0
    erc20_avg_val_sent: float = 0.0
    erc20_uniq_sent_token_name: int = 0
    erc20_uniq_rec_token_name: int = 0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert features to DataFrame matching training format."""
        return pd.DataFrame(
            [
                {
                    "Avg min between sent tnx": self.avg_min_between_sent_tnx,
                    "Avg min between received tnx": self.avg_min_between_received_tnx,
                    "Time Diff between first and last (Mins)": self.time_diff_first_last_mins,
                    "Sent tnx": self.sent_tnx,
                    "Received Tnx": self.received_tnx,
                    "Unique Received From Addresses": self.unique_received_from_addresses,
                    "Unique Sent To Addresses": self.unique_sent_to_addresses,
                    "avg val sent": self.avg_val_sent,
                    "avg val received": self.avg_val_received,
                    "total Ether sent": self.total_ether_sent,
                    "total ether received": self.total_ether_received,
                    "total ether balance": self.total_ether_balance,
                    " ERC20 total Ether received": self.erc20_total_ether_received,
                    " ERC20 total ether sent": self.erc20_total_ether_sent,
                    " ERC20 uniq sent addr": self.erc20_uniq_sent_addr,
                    " ERC20 uniq rec addr": self.erc20_uniq_rec_addr,
                    " ERC20 avg val rec": self.erc20_avg_val_rec,
                    " ERC20 avg val sent": self.erc20_avg_val_sent,
                    " ERC20 uniq sent token name": self.erc20_uniq_sent_token_name,
                    " ERC20 uniq rec token name": self.erc20_uniq_rec_token_name,
                }
            ]
        )


def compute_time_features(
    timestamps: list[int], is_sent: list[bool]
) -> tuple[float, float, float]:
    """
    Compute time-based features from transaction timestamps.

    Args:
        timestamps: List of Unix timestamps.
        is_sent: List of booleans indicating if transaction was sent.

    Returns:
        Tuple of (avg_min_between_sent, avg_min_between_received, time_diff_first_last).
    """
    if not timestamps:
        return 0.0, 0.0, 0.0

    sent_times = sorted([t for t, s in zip(timestamps, is_sent) if s])
    received_times = sorted([t for t, s in zip(timestamps, is_sent) if not s])

    avg_sent = _compute_avg_time_diff(sent_times)
    avg_received = _compute_avg_time_diff(received_times)

    time_diff = (max(timestamps) - min(timestamps)) / 60.0 if len(timestamps) > 1 else 0.0

    return avg_sent, avg_received, time_diff


def _compute_avg_time_diff(sorted_times: list[int]) -> float:
    """Compute average time difference between consecutive timestamps in minutes."""
    if len(sorted_times) < 2:
        return 0.0

    diffs = [
        (sorted_times[i + 1] - sorted_times[i]) / 60.0
        for i in range(len(sorted_times) - 1)
    ]
    return float(np.mean(diffs)) if diffs else 0.0


def compute_value_features(
    values: list[float], is_sent: list[bool]
) -> tuple[float, float, float, float, float]:
    """
    Compute value-based features.

    Args:
        values: List of transaction values in ETH.
        is_sent: List of booleans indicating if transaction was sent.

    Returns:
        Tuple of (avg_sent, avg_received, total_sent, total_received, balance).
    """
    sent_values = [v for v, s in zip(values, is_sent) if s]
    received_values = [v for v, s in zip(values, is_sent) if not s]

    avg_sent = float(np.mean(sent_values)) if sent_values else 0.0
    avg_received = float(np.mean(received_values)) if received_values else 0.0
    total_sent = sum(sent_values)
    total_received = sum(received_values)
    balance = total_received - total_sent

    return avg_sent, avg_received, total_sent, total_received, balance


def compute_address_features(
    from_addresses: list[str], to_addresses: list[str], wallet_address: str
) -> tuple[int, int]:
    """
    Compute unique address features.

    Args:
        from_addresses: List of sender addresses.
        to_addresses: List of recipient addresses.
        wallet_address: The wallet being analyzed.

    Returns:
        Tuple of (unique_received_from, unique_sent_to).
    """
    wallet_lower = wallet_address.lower()

    # Addresses that sent TO this wallet
    unique_received_from = len(
        set(addr.lower() for addr in from_addresses if addr.lower() != wallet_lower)
    )

    # Addresses this wallet sent TO
    unique_sent_to = len(
        set(addr.lower() for addr in to_addresses if addr.lower() != wallet_lower)
    )

    return unique_received_from, unique_sent_to
