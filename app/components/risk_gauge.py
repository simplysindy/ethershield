"""Risk score gauge visualization component."""

import plotly.graph_objects as go
import streamlit as st


def render_risk_gauge(risk_score: int, risk_level: str) -> None:
    """
    Render a gauge chart showing the risk score.

    Args:
        risk_score: Risk score from 0-100.
        risk_level: Risk level string (Low/Medium/High).
    """
    # Determine color based on score
    if risk_score >= 70:
        bar_color = "#FF4444"
    elif risk_score >= 40:
        bar_color = "#FFAA00"
    else:
        bar_color = "#44AA44"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"Risk Score - {risk_level}", "font": {"size": 20}},
            number={"font": {"size": 48}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": bar_color},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 40], "color": "#E8F5E9"},  # Light green
                    {"range": [40, 70], "color": "#FFF3E0"},  # Light orange
                    {"range": [70, 100], "color": "#FFEBEE"},  # Light red
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": risk_score,
                },
            },
        )
    )

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_risk_badge(risk_score: int, risk_level: str) -> None:
    """
    Render a colored badge showing the risk level.

    Args:
        risk_score: Risk score from 0-100.
        risk_level: Risk level string.
    """
    if risk_score >= 70:
        color = "red"
        icon = "\u26a0"  # Warning sign
    elif risk_score >= 40:
        color = "orange"
        icon = "\u2139"  # Info sign
    else:
        color = "green"
        icon = "\u2713"  # Check mark

    st.markdown(
        f"""
        <div style="
            background-color: {color};
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        ">
            {icon} {risk_level}: {risk_score}/100
        </div>
        """,
        unsafe_allow_html=True,
    )
