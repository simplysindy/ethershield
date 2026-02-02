"""ETH balance time-series chart component."""

from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_balance_chart(balance_history: list[tuple[int, float]]) -> None:
    """
    Render a time-series line chart of ETH balance.

    Args:
        balance_history: List of (timestamp, balance) tuples.
    """
    if not balance_history:
        st.info("No transaction history available for balance chart.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(balance_history, columns=["timestamp", "balance"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="s")

    # Get date range for title and axis
    min_date = df["date"].min()
    max_date = df["date"].max()
    date_range_str = f"{min_date.strftime('%b %d, %Y')} - {max_date.strftime('%b %d, %Y')}"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["balance"],
            mode="lines+markers",
            name="ETH Balance",
            line=dict(color="#627EEA", width=2),
            marker=dict(size=4),
            fill="tozeroy",
            fillcolor="rgba(98, 126, 234, 0.1)",
        )
    )

    fig.update_layout(
        title=f"ETH Balance Over Time ({date_range_str})",
        xaxis_title="Date",
        yaxis_title="Balance (ETH)",
        hovermode="x unified",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128, 128, 128, 0.2)",
            range=[min_date, max_date],  # Explicitly set range to data bounds
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128, 128, 128, 0.2)",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("Note: Chart based on last 10,000 transactions (API limit), calibrated to current balance.")


def render_transaction_volume_chart(
    timestamps: list[int], values: list[float], is_sent: list[bool]
) -> None:
    """
    Render a chart showing transaction volumes over time.

    Args:
        timestamps: List of transaction timestamps.
        values: List of transaction values in ETH.
        is_sent: List indicating if each transaction was sent.
    """
    if not timestamps:
        st.info("No transactions to display.")
        return

    df = pd.DataFrame({
        "timestamp": timestamps,
        "value": values,
        "type": ["Sent" if s else "Received" for s in is_sent],
    })
    df["date"] = pd.to_datetime(df["timestamp"], unit="s")

    fig = px.scatter(
        df,
        x="date",
        y="value",
        color="type",
        color_discrete_map={"Sent": "#FF6B6B", "Received": "#4ECDC4"},
        title="Transaction History",
        labels={"value": "Value (ETH)", "date": "Date", "type": "Type"},
    )

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="closest",
    )

    st.plotly_chart(fig, use_container_width=True)
