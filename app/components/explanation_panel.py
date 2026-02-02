"""Risk explanation panel component."""

import streamlit as st


def render_explanation_panel(explanations: list[dict]) -> None:
    """
    Render the risk explanation panel showing top contributing factors.

    Args:
        explanations: List of explanation dicts from RiskExplainer.
    """
    st.subheader("Risk Factors")

    if not explanations:
        st.info("No explanation data available.")
        return

    for i, exp in enumerate(explanations, 1):
        impact = exp["impact"]
        direction = exp["direction"]

        # Determine styling based on impact direction
        if impact > 0:
            icon = "\u2191"  # Up arrow
            color = "#FF6B6B"
        else:
            icon = "\u2193"  # Down arrow
            color = "#4ECDC4"

        # Format the impact magnitude
        magnitude = abs(impact)
        if magnitude > 0.3:
            strength = "Strong"
        elif magnitude > 0.1:
            strength = "Moderate"
        else:
            strength = "Weak"

        with st.container():
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(
                    f"""
                    **{i}. {exp['description']}**

                    {exp['explanation']}
                    """,
                )

            with col2:
                st.markdown(
                    f"""
                    <div style="
                        background-color: {color};
                        color: white;
                        padding: 5px 10px;
                        border-radius: 3px;
                        text-align: center;
                        font-size: 12px;
                    ">
                        {icon} {strength}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.divider()


def render_feature_summary(features_dict: dict, current_balance: float | None = None) -> None:
    """
    Render a summary of key feature values in a table format.

    Args:
        features_dict: Dictionary of feature names to values.
        current_balance: Actual current balance from API (optional).
    """
    st.subheader("Transaction Summary")

    def format_value(value, is_eth: bool = False) -> str:
        """Format a value for display."""
        if isinstance(value, float):
            if is_eth:
                return f"{value:,.4f} ETH"
            elif abs(value) >= 1000:
                return f"{value:,.0f}"
            else:
                return f"{value:.4f}"
        return f"{int(value):,}"

    # Build table data
    table_data = [
        ("Transactions Sent", format_value(features_dict.get("Sent tnx", 0))),
        ("Transactions Received", format_value(features_dict.get("Received Tnx", 0))),
        ("Total ETH Sent", format_value(features_dict.get("total Ether sent", 0), is_eth=True)),
        ("Total ETH Received", format_value(features_dict.get("total ether received", 0), is_eth=True)),
        ("Unique Recipients", format_value(features_dict.get("Unique Sent To Addresses", 0))),
        ("Unique Senders", format_value(features_dict.get("Unique Received From Addresses", 0))),
    ]

    # Use actual balance from API if available
    if current_balance is not None:
        table_data.append(("Current Balance (API)", format_value(current_balance, is_eth=True)))

    # Display as markdown table
    table_md = "| Metric | Value |\n|--------|-------|\n"
    for label, value in table_data:
        table_md += f"| {label} | {value} |\n"

    st.markdown(table_md)


def render_risk_interpretation(risk_score: int, risk_level: str) -> None:
    """
    Render human-readable interpretation of the risk score.

    Args:
        risk_score: Risk score from 0-100.
        risk_level: Risk level category.
    """
    st.subheader("Risk Interpretation")

    if risk_score >= 70:
        interpretation = """
        **High Risk Detected**

        This wallet exhibits transaction patterns commonly associated with suspicious activity.
        Patterns may include:
        - Unusual transaction timing or frequency
        - Interaction with known risky addresses
        - Abnormal value distributions

        **Recommendation**: Exercise extreme caution when interacting with this address.
        """
    elif risk_score >= 40:
        interpretation = """
        **Medium Risk Detected**

        This wallet shows some patterns that warrant attention, but doesn't exhibit
        clearly malicious behavior. This could indicate:
        - Automated trading activity
        - High-volume legitimate operations
        - Some interaction with flagged addresses

        **Recommendation**: Verify the source and purpose before large transactions.
        """
    else:
        interpretation = """
        **Low Risk Detected**

        This wallet's transaction patterns appear consistent with normal Ethereum usage.
        Indicators include:
        - Regular transaction timing
        - Typical value distributions
        - Limited interaction with flagged addresses

        **Recommendation**: Standard precautions apply for any crypto transaction.
        """

    st.markdown(interpretation)
