#!/usr/bin/env python
"""CLI script to train the fraud detection model."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_kaggle_dataset, prepare_training_data
from src.ml.trainer import FraudDetectorTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train the Ethereum fraud detection model"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to training CSV file (default: data/raw/transaction_dataset.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory to save model (default: models/trained/)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of XGBoost estimators (default: 200)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum tree depth (default: 6)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE class balancing",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("EtherShield - Fraud Detection Model Training")
    print("=" * 60)

    # Load data
    print("\nLoading dataset...")
    try:
        df = load_kaggle_dataset(args.data)
        print(f"Loaded {len(df)} records")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo download the dataset:")
        print("1. Visit https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset")
        print("2. Download and extract to data/raw/transaction_dataset.csv")
        sys.exit(1)

    # Prepare training data
    print("\nPreparing features...")
    X, y = prepare_training_data(df)
    print(f"Features: {len(X.columns)}")
    print(f"Fraud ratio: {y.mean():.2%}")

    # Train model
    print("\nTraining XGBoost model...")
    trainer = FraudDetectorTrainer(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )

    metrics = trainer.train(X, y, apply_smote=not args.no_smote)

    print("\n" + "=" * 60)
    print("Training Results")
    print("=" * 60)
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Training samples: {metrics['train_samples']}")
    print(f"Test samples: {metrics['test_samples']}")

    # Check F1 threshold
    if metrics["f1_score"] < 0.8:
        print("\nWarning: F1 score below 0.8 threshold")

    # Show feature importance
    print("\nTop 10 Feature Importances:")
    importance = trainer.get_feature_importance()
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for name, score in sorted_importance[:10]:
        print(f"  {name}: {score:.4f}")

    # Save model
    print("\nSaving model...")
    trainer.save(args.output)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
