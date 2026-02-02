"""Pydantic models for Etherscan API responses."""

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """Model for an Ethereum transaction."""

    block_number: str = Field(alias="blockNumber")
    timestamp: str = Field(alias="timeStamp")
    hash: str
    from_address: str = Field(alias="from")
    to_address: str | None = Field(alias="to", default=None)
    value: str
    gas: str
    gas_price: str = Field(alias="gasPrice")
    gas_used: str = Field(alias="gasUsed")
    is_error: str = Field(alias="isError", default="0")
    contract_address: str = Field(alias="contractAddress", default="")

    @property
    def value_in_eth(self) -> float:
        """Convert Wei value to ETH."""
        return int(self.value) / 1e18

    @property
    def gas_price_gwei(self) -> float:
        """Convert gas price to Gwei."""
        return int(self.gas_price) / 1e9

    @property
    def timestamp_int(self) -> int:
        """Get timestamp as integer."""
        return int(self.timestamp)


class ERC20Transfer(BaseModel):
    """Model for an ERC20 token transfer."""

    block_number: str = Field(alias="blockNumber")
    timestamp: str = Field(alias="timeStamp")
    hash: str
    from_address: str = Field(alias="from")
    to_address: str = Field(alias="to")
    value: str
    token_name: str = Field(alias="tokenName", default="")
    token_symbol: str = Field(alias="tokenSymbol", default="")
    token_decimal: str = Field(alias="tokenDecimal", default="18")
    contract_address: str = Field(alias="contractAddress")

    @property
    def value_normalized(self) -> float:
        """Convert value using token decimals."""
        decimals = int(self.token_decimal) if self.token_decimal else 18
        return int(self.value) / (10**decimals)


class EtherscanResponse(BaseModel):
    """Generic Etherscan API response."""

    status: str
    message: str
    result: list | str | None = None

    @property
    def is_success(self) -> bool:
        """Check if the API call was successful."""
        return self.status == "1"


class BalanceResponse(BaseModel):
    """Etherscan balance API response."""

    status: str
    message: str
    result: str | None = None

    @property
    def balance_eth(self) -> float:
        """Get balance in ETH."""
        if self.result is None:
            return 0.0
        return int(self.result) / 1e18
