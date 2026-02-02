# EtherShield

**Real-Time Ethereum Wallet Risk Classifier**

An ML-powered dashboard that analyzes Ethereum wallet addresses and calculates a "Risk Score" based on transaction history, identifying potential illicit activity such as money laundering, phishing, or bot behavior.

**[Live Demo](https://shareappio-ahlwqgisfysvjdbzehjmky.streamlit.app/)** · **[Quick Start Guide](QUICKSTART.md)**

![Dashboard Overview](images/image1.png)

## The Problem

Blockchain's transparency is a double-edged sword. While all transactions are public, the sheer volume makes manual analysis impossible. Bad actors exploit this by:
- Layering funds through multiple wallets to obscure origins
- Using automated bots for wash trading or market manipulation
- Running phishing operations that drain victim wallets

EtherShield automates the detection of these patterns using machine learning trained on known fraud cases.

## What It Does

EtherShield helps identify suspicious Ethereum wallets by analyzing their on-chain transaction patterns. Enter any wallet address, and within seconds you'll receive:

- **Risk Score (0-100)**: An instant assessment of how likely the wallet is involved in fraudulent activity
- **Balance History**: Visual chart showing ETH balance changes over time
- **Transaction Summary**: Key metrics including total sent/received, unique counterparties, and ERC-20 token activity
- **Explainability Report**: Understand *why* a wallet was flagged, with the specific factors contributing to the score

![Transaction Analysis](images/image2.png)

## Web3 Integration

EtherShield connects to the Ethereum blockchain via the **Etherscan API** to fetch real-time on-chain data:

- **ETH Transactions**: Complete transaction history including sender, recipient, value, gas fees, and timestamps
- **ERC-20 Token Transfers**: Token movement patterns across different assets
- **Wallet Balance**: Current holdings for context

The async API client implements rate limiting (5 calls/sec for free tier) and exponential backoff retry logic to handle API constraints gracefully.

## Machine Learning Pipeline

### Training Data
Model trained on the [Ethereum Fraud Detection Dataset](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset) from Kaggle containing 10,000+ labeled wallet addresses (fraudulent vs. legitimate).

### Feature Engineering
20 behavioral features extracted from on-chain transaction data:

| Category | Features |
|----------|----------|
| **Temporal** | Avg time between sent/received transactions, activity duration |
| **Volume** | Transaction counts (sent/received), total ETH moved |
| **Network** | Unique sender/recipient addresses (counterparty diversity) |
| **Value** | Avg transaction values, total balance |
| **ERC-20** | Token transfer counts, unique tokens, token value distributions |

### Model Architecture
- **Algorithm**: XGBoost gradient boosting classifier
- **Class Balancing**: SMOTE (Synthetic Minority Over-sampling) to handle fraud/legitimate imbalance
- **Preprocessing**: StandardScaler for feature normalization

### Model Performance

| Metric | Score |
|--------|-------|
| **F1 Score** | 0.916 |
| **Accuracy** | 0.963 |
| **Precision (Fraud)** | 0.94 |
| **Recall (Fraud)** | 0.90 |

Evaluated on held-out test set (1,969 samples, 22% fraud ratio).

### Explainability
Gain-based feature importance extraction generates human-readable explanations, showing users exactly which behavioral patterns contributed to a high-risk classification.

## Risk Levels

| Score | Level | Interpretation |
|-------|-------|----------------|
| 0-39 | Low Risk | Normal transaction patterns |
| 40-69 | Medium Risk | Some unusual patterns detected |
| 70-100 | High Risk | Strong indicators of suspicious activity |

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Machine Learning** | XGBoost, scikit-learn, SMOTE (imbalanced-learn) |
| **Explainability** | SHAP-inspired feature importance |
| **Blockchain Data** | Etherscan API |
| **Async I/O** | aiohttp, tenacity (retry logic) |
| **Frontend** | Streamlit, Plotly |
| **Validation** | Pydantic |

## Project Structure

```
eth-classifier/
├── app/                    # Streamlit dashboard
│   ├── main.py
│   └── components/         # UI components (gauge, charts, panels)
├── src/
│   ├── etherscan/          # API client & data transformation
│   ├── ml/                 # Model training, prediction, explainer
│   └── data/               # Feature engineering pipeline
├── models/trained/         # Serialized model files
├── scripts/                # Training scripts
└── tests/                  # Unit tests
```

## License

[MIT](LICENSE)
