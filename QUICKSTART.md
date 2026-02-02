# Quick Start Guide

## Prerequisites
- Python 3.11+
- [Etherscan API Key](https://etherscan.io/apis) (free tier works)

## Installation

```bash
# Clone the repository
git clone https://github.com/simplysindy/ethershield
cd ethershield

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Add your ETHERSCAN_API_KEY to .env
```

## Training the Model

Download the [Ethereum Fraud Detection Dataset](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset) from Kaggle and place `transaction_dataset.csv` in `data/raw/`.

```bash
uv run python scripts/train_model.py
```

## Running the Dashboard

```bash
uv run streamlit run app/main.py
```

Open http://localhost:8501 in your browser.

## Running Tests

```bash
uv run pytest tests/ -v
```
