"""Transform Etherscan API data into model features."""

import numpy as np

from src.data.feature_engineering import (
    TransactionFeatures,
    compute_address_features,
    compute_time_features,
    compute_value_features,
)

from .models import ERC20Transfer, Transaction


class TransactionTransformer:
    """Transform raw transaction data into ML features."""

    def __init__(self, wallet_address: str):
        """
        Initialize transformer for a specific wallet.

        Args:
            wallet_address: The wallet address being analyzed.
        """
        self.wallet_address = wallet_address.lower()

    def compute_features(
        self,
        transactions: list[Transaction],
        erc20_transfers: list[ERC20Transfer] | None = None,
    ) -> TransactionFeatures:
        """
        Compute all features from transaction data.

        Args:
            transactions: List of normal transactions.
            erc20_transfers: Optional list of ERC20 transfers.

        Returns:
            TransactionFeatures with computed values.
        """
        features = TransactionFeatures()

        if not transactions:
            return features

        # Determine sent vs received for each transaction
        is_sent = [
            tx.from_address.lower() == self.wallet_address for tx in transactions
        ]

        # Time features
        timestamps = [tx.timestamp_int for tx in transactions]
        (
            features.avg_min_between_sent_tnx,
            features.avg_min_between_received_tnx,
            features.time_diff_first_last_mins,
        ) = compute_time_features(timestamps, is_sent)

        # Count features
        features.sent_tnx = sum(is_sent)
        features.received_tnx = len(transactions) - features.sent_tnx

        # Address features
        from_addresses = [tx.from_address for tx in transactions]
        to_addresses = [tx.to_address or "" for tx in transactions]
        (
            features.unique_received_from_addresses,
            features.unique_sent_to_addresses,
        ) = compute_address_features(from_addresses, to_addresses, self.wallet_address)

        # Value features
        values = [tx.value_in_eth for tx in transactions]
        (
            features.avg_val_sent,
            features.avg_val_received,
            features.total_ether_sent,
            features.total_ether_received,
            features.total_ether_balance,
        ) = compute_value_features(values, is_sent)

        # ERC20 features
        if erc20_transfers:
            self._compute_erc20_features(features, erc20_transfers)

        return features

    def _compute_erc20_features(
        self, features: TransactionFeatures, transfers: list[ERC20Transfer]
    ) -> None:
        """
        Compute ERC20-specific features.

        Args:
            features: TransactionFeatures object to update.
            transfers: List of ERC20 transfers.
        """
        is_sent = [
            tx.from_address.lower() == self.wallet_address for tx in transfers
        ]

        sent_values = [
            tx.value_normalized for tx, s in zip(transfers, is_sent) if s
        ]
        received_values = [
            tx.value_normalized for tx, s in zip(transfers, is_sent) if not s
        ]

        # Total values (in token units, treated as ETH-equivalent for scoring)
        features.erc20_total_ether_sent = sum(sent_values)
        features.erc20_total_ether_received = sum(received_values)

        # Average values
        features.erc20_avg_val_sent = float(np.mean(sent_values)) if sent_values else 0.0
        features.erc20_avg_val_rec = (
            float(np.mean(received_values)) if received_values else 0.0
        )

        # Unique addresses
        sent_addresses = set(
            tx.to_address.lower() for tx, s in zip(transfers, is_sent) if s
        )
        received_addresses = set(
            tx.from_address.lower() for tx, s in zip(transfers, is_sent) if not s
        )
        features.erc20_uniq_sent_addr = len(sent_addresses)
        features.erc20_uniq_rec_addr = len(received_addresses)

        # Unique token names
        sent_tokens = set(
            tx.token_name for tx, s in zip(transfers, is_sent) if s and tx.token_name
        )
        received_tokens = set(
            tx.token_name for tx, s in zip(transfers, is_sent) if not s and tx.token_name
        )
        features.erc20_uniq_sent_token_name = len(sent_tokens)
        features.erc20_uniq_rec_token_name = len(received_tokens)

    def compute_balance_history(
        self, transactions: list[Transaction], current_balance: float | None = None
    ) -> list[tuple[int, float]]:
        """
        Compute running balance history from transactions.

        If current_balance is provided, works backwards from current balance
        to compute accurate historical balances.

        Args:
            transactions: List of transactions.
            current_balance: Current balance from API (optional but recommended).

        Returns:
            List of (timestamp, balance) tuples.
        """
        if not transactions:
            return []

        # Sort by timestamp (oldest first)
        sorted_txs = sorted(transactions, key=lambda tx: tx.timestamp_int)

        if current_balance is not None:
            # Work backwards from current balance to find starting balance
            # before these transactions
            starting_balance = current_balance
            for tx in reversed(sorted_txs):
                if tx.from_address.lower() == self.wallet_address:
                    # We sent ETH, so before this tx we had more
                    starting_balance += tx.value_in_eth
                else:
                    # We received ETH, so before this tx we had less
                    starting_balance -= tx.value_in_eth
        else:
            starting_balance = 0.0

        # Now compute forward from starting balance
        balance = starting_balance
        history = []

        for tx in sorted_txs:
            if tx.from_address.lower() == self.wallet_address:
                balance -= tx.value_in_eth
            else:
                balance += tx.value_in_eth

            history.append((tx.timestamp_int, balance))

        return history
