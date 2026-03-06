#!/usr/bin/env python3
"""
Unified Router Training Script

Training script for the Unified Router's multi-tier quality predictors.

This script trains separate quality predictors for each model tier:
- draft_8b: Predicts quality for 8B parameter models
- draft_70b: Predicts quality for 70B draft models (Groq)
- target_70b: Predicts quality for 70B target models (GPT-4o, Claude)
- ultra_400b: Predicts quality for frontier models (GPT-5.2, Claude Opus)

Training data can be loaded from a JSON/CSV file or generated synthetically.

Usage:
    python3 train_unified.py [--data-file PATH] [--output-dir DIR] [--model-type rf|gb|lr]
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import warnings

warnings.filterwarnings("ignore")


# Feature columns for training
FEATURE_COLUMNS = [
    "prompt_length_norm",
    "word_count_norm",
    "has_code_request",
    "has_reasoning_request",
    "has_creative_request",
    "has_analysis_request",
    "question_count_norm",
    "complexity_score",
]

# Model tiers to train
MODEL_TIERS = ["draft_8b", "draft_70b", "target_70b", "ultra_400b"]

# Quality threshold for binary classification
QUALITY_THRESHOLD = 0.7  # Above this = "good quality"


def load_training_data_from_file(data_file: str) -> Optional[pd.DataFrame]:
    """
    Load training data from a JSON or CSV file.

    JSON format: list of dicts with keys matching expected columns.
    CSV format: standard CSV with header row.

    Expected columns:
        - task_type: str (general, code, writing, reasoning, analysis)
        - model_tier: str (draft_8b, draft_70b, target_70b, ultra_400b)
        - prompt_length: float
        - word_count: float (optional, derived from prompt_length if missing)
        - has_code_request: bool
        - has_reasoning_request: bool
        - has_creative_request: bool (optional)
        - has_analysis_request: bool (optional)
        - question_count: int (optional)
        - complexity_score: float
        - quality_score: float
        - was_accepted: bool (optional, derived from quality_score if missing)

    Args:
        data_file: Path to JSON or CSV file.

    Returns:
        DataFrame or None if loading fails.
    """
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return None

    try:
        if data_file.endswith(".csv"):
            df = pd.read_csv(data_file)
        else:
            with open(data_file, "r") as f:
                records = json.load(f)
            df = pd.DataFrame(records)

        print(f"Loaded {len(df)} training samples from {data_file}")
        return df

    except Exception as e:
        print(f"Failed to load training data from {data_file}: {e}")
        return None


def generate_synthetic_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic training data for initial model training.

    This simulates the kind of data we'd collect from real usage.
    The quality scores are based on empirical observations of model capabilities.
    """
    np.random.seed(42)

    # Task types and their characteristics
    task_types = ["general", "code", "writing", "reasoning", "analysis"]

    # Quality distributions by tier and task (mean, std)
    quality_distributions = {
        "draft_8b": {
            "general": (0.72, 0.12),
            "code": (0.58, 0.15),
            "writing": (0.68, 0.10),
            "reasoning": (0.45, 0.18),
            "analysis": (0.55, 0.14),
        },
        "draft_70b": {
            "general": (0.82, 0.08),
            "code": (0.75, 0.12),
            "writing": (0.80, 0.08),
            "reasoning": (0.65, 0.15),
            "analysis": (0.75, 0.10),
        },
        "target_70b": {
            "general": (0.90, 0.06),
            "code": (0.88, 0.08),
            "writing": (0.90, 0.05),
            "reasoning": (0.85, 0.10),
            "analysis": (0.88, 0.07),
        },
        "ultra_400b": {
            "general": (0.96, 0.03),
            "code": (0.95, 0.04),
            "writing": (0.95, 0.03),
            "reasoning": (0.94, 0.05),
            "analysis": (0.95, 0.04),
        },
    }

    data = []

    for _ in range(n_samples):
        # Random task type
        task_type = np.random.choice(task_types, p=[0.35, 0.25, 0.15, 0.15, 0.10])

        # Random model tier
        tier = np.random.choice(MODEL_TIERS, p=[0.40, 0.20, 0.25, 0.15])

        # Generate features
        prompt_length = np.random.exponential(200) + 20
        word_count = prompt_length / 5

        has_code = task_type == "code" or np.random.random() < 0.1
        has_reasoning = task_type == "reasoning" or np.random.random() < 0.15
        has_creative = task_type == "writing" or np.random.random() < 0.1
        has_analysis = task_type == "analysis" or np.random.random() < 0.1

        question_count = np.random.poisson(1)

        # Complexity based on features
        complexity = 0.0
        complexity += min(prompt_length / 1000, 1.0) * 0.3
        complexity += 0.2 if has_reasoning else 0
        complexity += 0.15 if has_code else 0
        complexity += 0.1 if question_count > 2 else 0
        complexity = min(complexity, 1.0)

        # Generate quality score based on tier and task
        mean, std = quality_distributions[tier][task_type]

        # Adjust quality based on complexity
        if complexity > 0.6:
            if tier == "draft_8b":
                mean *= 0.85  # Draft struggles with complex
            elif tier == "ultra_400b":
                mean = min(mean * 1.02, 1.0)  # Ultra handles complex well

        quality = np.clip(np.random.normal(mean, std), 0, 1)

        # Was it accepted? (quality > threshold)
        was_accepted = quality > QUALITY_THRESHOLD

        data.append(
            {
                "task_type": task_type,
                "model_tier": tier,
                "prompt_length": prompt_length,
                "word_count": word_count,
                "has_code_request": has_code,
                "has_reasoning_request": has_reasoning,
                "has_creative_request": has_creative,
                "has_analysis_request": has_analysis,
                "question_count": question_count,
                "complexity_score": complexity,
                "quality_score": quality,
                "was_accepted": was_accepted,
            }
        )

    df = pd.DataFrame(data)
    print(f"Generated {len(df)} synthetic training samples")
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare feature matrix and labels from DataFrame.

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Binary labels (1 = good quality, 0 = poor quality)
    """
    # Normalize features
    df["prompt_length_norm"] = df["prompt_length"] / 1000
    df["word_count_norm"] = df["word_count"] / 200
    df["question_count_norm"] = df["question_count"] / 5

    # Ensure boolean columns are numeric
    for col in [
        "has_code_request",
        "has_reasoning_request",
        "has_creative_request",
        "has_analysis_request",
    ]:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].astype(float)

    # Extract features
    X = df[FEATURE_COLUMNS].values

    # Binary labels: good quality (1) vs poor quality (0)
    y = (df["quality_score"] >= QUALITY_THRESHOLD).astype(int).values

    return X, y


def train_tier_predictor(
    df: pd.DataFrame, tier: str, model_type: str = "rf"
) -> Tuple[any, Dict[str, float]]:
    """
    Train a quality predictor for a specific model tier.

    Args:
        df: Training data
        tier: Model tier to train for
        model_type: 'rf' (RandomForest), 'gb' (GradientBoosting), 'lr' (LogisticRegression)

    Returns:
        Trained model and metrics dict
    """
    # Filter data for this tier
    tier_df = df[df["model_tier"] == tier].copy()

    if len(tier_df) < 100:
        print(f"Insufficient data for {tier}: {len(tier_df)} samples")
        return None, {}

    print(f"\n{'=' * 60}")
    print(f"Training predictor for: {tier}")
    print(f"Samples: {len(tier_df)}")
    print(f"{'=' * 60}")

    # Prepare features
    X, y = prepare_features(tier_df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Select model
    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "gb":
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
    else:
        model = LogisticRegression(max_iter=1000, random_state=42)

    # Train
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0,
        "samples": len(tier_df),
    }

    print(f"\nMetrics for {tier}:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

    # Feature importance (for tree-based models)
    if hasattr(model, "feature_importances_"):
        print("\nFeature Importance:")
        for feat, imp in sorted(
            zip(FEATURE_COLUMNS, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        ):
            print(f"  {feat}: {imp:.4f}")

    # Wrap model with scaler for inference
    class ScaledModel:
        def __init__(self, model, scaler):
            self.model = model
            self.scaler = scaler

        def predict(self, X):
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)

        def predict_proba(self, X):
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)

    return ScaledModel(model, scaler), metrics


def train_all_predictors(
    df: pd.DataFrame, output_dir: str = "models", model_type: str = "rf"
) -> Dict[str, Dict[str, float]]:
    """
    Train quality predictors for all model tiers.

    Args:
        df: Training data
        output_dir: Directory to save trained models
        model_type: Type of classifier to use

    Returns:
        Dict of metrics for each tier
    """
    os.makedirs(output_dir, exist_ok=True)

    all_metrics = {}

    for tier in MODEL_TIERS:
        model, metrics = train_tier_predictor(df, tier, model_type)

        if model is not None:
            # Save model
            model_path = os.path.join(output_dir, f"{tier}_predictor.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"Saved {tier} predictor to {model_path}")

            all_metrics[tier] = metrics

    return all_metrics


def evaluate_routing_accuracy(df: pd.DataFrame, models_dir: str = "models"):
    """
    Evaluate end-to-end routing accuracy.

    Simulates routing decisions and checks if we would have picked
    the optimal model (highest quality at lowest cost).
    """
    print(f"\n{'=' * 60}")
    print("Evaluating End-to-End Routing Accuracy")
    print(f"{'=' * 60}")

    # Load trained models
    predictors = {}
    for tier in MODEL_TIERS:
        model_path = os.path.join(models_dir, f"{tier}_predictor.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                predictors[tier] = pickle.load(f)

    if not predictors:
        print("No trained models found")
        return

    # Cost per tier (simplified)
    tier_costs = {
        "draft_8b": 0.05,
        "draft_70b": 0.59,
        "target_70b": 5.00,
        "ultra_400b": 7.875,
    }

    correct_routes = 0
    total_routes = 0
    cost_savings = 0

    # Group by unique prompts (simulate routing decisions)
    for task_type in df["task_type"].unique():
        task_df = df[df["task_type"] == task_type]

        # Sample some prompts
        for _, row in task_df.sample(min(100, len(task_df))).iterrows():
            # Get actual quality for each tier
            actual_qualities = {}
            for tier in MODEL_TIERS:
                tier_rows = df[
                    (df["task_type"] == task_type) & (df["model_tier"] == tier)
                ]
                if len(tier_rows) > 0:
                    actual_qualities[tier] = tier_rows["quality_score"].mean()

            if not actual_qualities:
                continue

            # Predict quality for each tier
            features = np.array(
                [
                    [
                        row["prompt_length"] / 1000,
                        row["word_count"] / 200,
                        float(row.get("has_code_request", False)),
                        float(row.get("has_reasoning_request", False)),
                        float(row.get("has_creative_request", False)),
                        float(row.get("has_analysis_request", False)),
                        row.get("question_count", 0) / 5,
                        row["complexity_score"],
                    ]
                ]
            )

            predicted_utilities = {}
            for tier, predictor in predictors.items():
                try:
                    quality_prob = predictor.predict_proba(features)[0][1]
                    utility = quality_prob / tier_costs[tier]
                    predicted_utilities[tier] = utility
                except Exception:
                    pass

            if not predicted_utilities:
                continue

            # Our routing decision
            predicted_best = max(predicted_utilities, key=predicted_utilities.get)

            # Optimal decision (highest quality that meets threshold, lowest cost)
            valid_tiers = [
                t for t, q in actual_qualities.items() if q >= QUALITY_THRESHOLD
            ]
            if valid_tiers:
                optimal_best = min(valid_tiers, key=lambda t: tier_costs[t])
            else:
                optimal_best = max(actual_qualities, key=actual_qualities.get)

            # Check if correct
            if predicted_best == optimal_best:
                correct_routes += 1

            # Calculate cost savings vs always using ultra
            if predicted_best != "ultra_400b":
                cost_savings += tier_costs["ultra_400b"] - tier_costs[predicted_best]

            total_routes += 1

    if total_routes > 0:
        accuracy = correct_routes / total_routes
        avg_savings = cost_savings / total_routes

        print("\nRouting Results:")
        print(f"  Total decisions: {total_routes}")
        print(f"  Correct routes:  {correct_routes}")
        print(f"  Accuracy:        {accuracy:.2%}")
        print(f"  Avg cost saved:  ${avg_savings:.4f}/request")


def main():
    """Main training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train unified router quality predictors"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to JSON or CSV training data file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained model pickles",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="rf",
        choices=["rf", "gb", "lr"],
        help="Classifier type: rf=RandomForest, gb=GradientBoosting, lr=LogisticRegression",
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=10000,
        help="Number of synthetic samples if no data file provided",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("UNIFIED ROUTER TRAINING PIPELINE")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    # Try to load real data from file, fall back to synthetic
    df = None
    if args.data_file:
        df = load_training_data_from_file(args.data_file)

    if df is None or len(df) < 500:
        print("\nUsing synthetic data for training...")
        df = generate_synthetic_data(n_samples=args.synthetic_samples)

    # Data summary
    print("\nData Summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Tiers: {df['model_tier'].value_counts().to_dict()}")
    print(f"  Tasks: {df['task_type'].value_counts().to_dict()}")
    print(f"  Avg quality: {df['quality_score'].mean():.4f}")

    # Train all predictors
    print("\nTraining predictors...")
    metrics = train_all_predictors(
        df, output_dir=args.output_dir, model_type=args.model_type
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 60}")

    for tier, m in metrics.items():
        print(f"\n{tier}:")
        print(f"  Samples:  {m.get('samples', 0)}")
        print(f"  Accuracy: {m.get('accuracy', 0):.4f}")
        print(f"  F1 Score: {m.get('f1', 0):.4f}")
        print(f"  ROC AUC:  {m.get('roc_auc', 0):.4f}")

    # Evaluate routing
    evaluate_routing_accuracy(df, models_dir=args.output_dir)

    print("\nTraining complete!")
    print(f"Models saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
