"""Etherscan API client with rate limiting and retry logic."""

import asyncio
from typing import Any

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings

from .models import BalanceResponse, ERC20Transfer, EtherscanResponse, Transaction


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, calls_per_second: int = 5):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait if necessary to respect rate limit."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            time_since_last = now - self.last_call
            if time_since_last < self.min_interval:
                await asyncio.sleep(self.min_interval - time_since_last)
            self.last_call = asyncio.get_event_loop().time()


class EtherscanClient:
    """Async client for Etherscan API with rate limiting."""

    def __init__(self, api_key: str | None = None):
        """
        Initialize the Etherscan client.

        Args:
            api_key: Etherscan API key. Defaults to settings.
        """
        self.api_key = api_key or settings.etherscan_api_key
        self.base_url = settings.etherscan_base_url
        self.rate_limiter = RateLimiter(settings.etherscan_rate_limit)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def _make_request(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Make a rate-limited request to Etherscan API.

        Args:
            params: Query parameters for the API call.

        Returns:
            JSON response as dictionary.

        Raises:
            aiohttp.ClientError: On request failure.
        """
        await self.rate_limiter.acquire()

        params["apikey"] = self.api_key
        params["chainid"] = settings.etherscan_chain_id
        session = await self._get_session()

        async with session.get(self.base_url, params=params) as response:
            response.raise_for_status()
            return await response.json()

    async def get_transactions(
        self,
        address: str,
        start_block: int = 0,
        end_block: int = 99999999,
        sort: str = "asc",
    ) -> list[Transaction]:
        """
        Get normal transactions for an address.

        Args:
            address: Ethereum wallet address.
            start_block: Starting block number.
            end_block: Ending block number.
            sort: Sort order ('asc' or 'desc').

        Returns:
            List of Transaction objects.
        """
        params = {
            "module": "account",
            "action": "txlist",
            "address": address,
            "startblock": start_block,
            "endblock": end_block,
            "sort": sort,
        }

        data = await self._make_request(params)
        response = EtherscanResponse(**data)

        if not response.is_success or not isinstance(response.result, list):
            return []

        return [Transaction(**tx) for tx in response.result]

    async def get_erc20_transfers(
        self,
        address: str,
        start_block: int = 0,
        end_block: int = 99999999,
        sort: str = "asc",
    ) -> list[ERC20Transfer]:
        """
        Get ERC20 token transfers for an address.

        Args:
            address: Ethereum wallet address.
            start_block: Starting block number.
            end_block: Ending block number.
            sort: Sort order ('asc' or 'desc').

        Returns:
            List of ERC20Transfer objects.
        """
        params = {
            "module": "account",
            "action": "tokentx",
            "address": address,
            "startblock": start_block,
            "endblock": end_block,
            "sort": sort,
        }

        data = await self._make_request(params)
        response = EtherscanResponse(**data)

        if not response.is_success or not isinstance(response.result, list):
            return []

        return [ERC20Transfer(**tx) for tx in response.result]

    async def get_balance(self, address: str) -> float:
        """
        Get ETH balance for an address.

        Args:
            address: Ethereum wallet address.

        Returns:
            Balance in ETH.
        """
        params = {
            "module": "account",
            "action": "balance",
            "address": address,
            "tag": "latest",
        }

        data = await self._make_request(params)
        response = BalanceResponse(**data)

        return response.balance_eth

    async def __aenter__(self) -> "EtherscanClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
