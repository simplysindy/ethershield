"""EtherShield - Ethereum Transaction Risk Classifier Dashboard."""

import asyncio
import sys
from pathlib import Path

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.etherscan.client import EtherscanClient
from src.etherscan.transformer import TransactionTransformer
from src.ml.explainer import RiskExplainer
from src.ml.predictor import RiskPredictor
from src.utils.validators import is_valid_eth_address

from app.components.balance_chart import render_balance_chart
from app.components.explanation_panel import (
    render_explanation_panel,
    render_feature_summary,
    render_risk_interpretation,
)
from app.components.risk_gauge import render_risk_gauge

# Page config
st.set_page_config(
    page_title="EtherShield - ETH Risk Analyzer",
    page_icon="\U0001F6E1",
    layout="wide",
)


def check_model_exists() -> bool:
    """Check if the trained model exists."""
    return settings.model_path.exists()


def check_api_key() -> bool:
    """Check if Etherscan API key is configured."""
    return bool(settings.etherscan_api_key)


async def analyze_wallet(address: str) -> dict:
    """
    Fetch data from Etherscan and analyze the wallet.

    Args:
        address: Ethereum wallet address.

    Returns:
        Dictionary with analysis results.
    """
    async with EtherscanClient() as client:
        # Fetch transactions in parallel
        transactions, erc20_transfers, balance = await asyncio.gather(
            client.get_transactions(address, sort="desc"),
            client.get_erc20_transfers(address, sort="desc"),
            client.get_balance(address),
        )

    if not transactions:
        return {
            "error": "No transactions found for this address.",
            "has_data": False,
        }

    # Transform to features
    transformer = TransactionTransformer(address)
    features = transformer.compute_features(transactions, erc20_transfers)
    features_df = features.to_dataframe()

    # Get prediction
    predictor = RiskPredictor()
    risk_score, probability = predictor.predict_risk(features_df)
    risk_level = predictor.get_risk_level(risk_score)

    # Get explanations
    predictor._ensure_loaded()
    features_aligned = predictor._align_features(features_df)
    features_scaled = predictor.scaler.transform(features_aligned)

    explainer = RiskExplainer(predictor.model, predictor.feature_names)
    explanations = explainer.explain_prediction(features_scaled, top_n=5)

    # Get balance history (using current balance for accurate calculation)
    balance_history = transformer.compute_balance_history(transactions, balance)

    return {
        "has_data": True,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "probability": probability,
        "explanations": explanations,
        "features": features_df.iloc[0].to_dict(),
        "balance_history": balance_history,
        "current_balance": balance,
        "transaction_count": len(transactions),
        "erc20_count": len(erc20_transfers),
    }


def main():
    # Header
    st.title("\U0001F6E1 EtherShield")
    st.markdown("**Ethereum Transaction Risk Classifier**")
    st.markdown("Analyze wallet addresses to identify potential illicit activity patterns.")

    # Sidebar with status
    with st.sidebar:
        st.header("Status")

        # Model status
        if check_model_exists():
            st.success("Model loaded")
        else:
            st.error("Model not found")
            st.markdown(
                "Run `python scripts/train_model.py` to train the model first."
            )

        # API status
        if check_api_key():
            st.success("API key configured")
        else:
            st.warning("API key missing")
            st.markdown(
                "Add `ETHERSCAN_API_KEY` to your `.env` file."
            )

        st.divider()
        st.markdown("### Risk Levels")
        st.markdown("""
        - **Low (0-39)**: Normal patterns
        - **Medium (40-69)**: Some concerns
        - **High (70-100)**: Suspicious activity
        """)

    # Main input
    col1, col2 = st.columns([4, 1])

    with col1:
        address = st.text_input(
            "Ethereum Wallet Address",
            placeholder="0x...",
            help="Enter a valid Ethereum address (42 characters starting with 0x)",
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button("Analyze Wallet", type="primary", use_container_width=True)

    # Example wallets section
    with st.expander("Example Wallets to Test"):
        st.markdown("""
        | Name | Address | Description |
        |------|---------|-------------|
        | Binance Hot Wallet | `0x28C6c06298d514Db089934071355E5743bf21d60` | Major exchange hot wallet with extremely high transaction volume. Patterns consistent with exchange operations. |
        | Binance Cold Wallet | `0x21a31Ee1afC51d94C2eFcCAa2092aD1028285549` | Exchange cold storage wallet. Large infrequent transfers typical of institutional custody operations. |
        """)
        st.caption("Click an address to copy, then paste above to analyze.")

    # Validation and analysis
    if analyze_button:
        if not address:
            st.error("Please enter an Ethereum address.")
            return

        if not is_valid_eth_address(address):
            st.error(
                "Invalid Ethereum address format. "
                "Address should be 42 characters starting with '0x'."
            )
            return

        if not check_model_exists():
            st.error(
                "Model not found. Please run `python scripts/train_model.py` first."
            )
            return

        if not check_api_key():
            st.error(
                "Etherscan API key not configured. "
                "Please add ETHERSCAN_API_KEY to your .env file."
            )
            return

        # Run analysis
        with st.spinner("Fetching transaction data and analyzing..."):
            try:
                results = asyncio.run(analyze_wallet(address))
            except Exception as e:
                st.error(f"Error analyzing wallet: {str(e)}")
                return

        if not results.get("has_data"):
            st.warning(results.get("error", "No data available for this address."))
            return

        # Display results
        st.divider()

        # Top row: Risk gauge and summary
        col1, col2 = st.columns([1, 1])

        with col1:
            render_risk_gauge(results["risk_score"], results["risk_level"])

        with col2:
            st.metric("Current Balance", f"{results['current_balance']:.4f} ETH")
            st.metric("Total Transactions", results["transaction_count"])
            st.metric("ERC20 Transfers", results["erc20_count"])

        st.divider()

        # Middle row: Charts
        col1, col2 = st.columns([1, 1])

        with col1:
            render_balance_chart(results["balance_history"])

        with col2:
            render_feature_summary(results["features"], results["current_balance"])

        st.divider()

        # Bottom row: Explanations
        col1, col2 = st.columns([1, 1])

        with col1:
            render_explanation_panel(results["explanations"])

        with col2:
            render_risk_interpretation(results["risk_score"], results["risk_level"])

    # Footer
    st.divider()
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "EtherShield uses machine learning to identify suspicious patterns. "
        "Results should be used for informational purposes only."
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
