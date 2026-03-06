#!/usr/bin/env python3
"""
ARBITRAGE Router Training Script

Trains a classifier to predict when the draft model is sufficient vs when
the target model is necessary, based on collected training data.

Usage:
    python3 train_router.py [--data-file PATH]

Output:
    arbitrage_router_v1.pkl - Trained classifier model
"""

import os
import json
import pickle
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ML imports
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    print("scikit-learn not installed. Run: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier

    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("LightGBM not available, will use sklearn classifiers")

OUTPUT_MODEL_PATH = os.getenv("ARBITRAGE_ROUTER_PATH", "arbitrage_router_v1.pkl")


def extract_features(row: Dict[str, Any]) -> List[float]:
    """
    Extract features from a training data row.
    Must match the features used in the arbitrage router.
    """
    prompt = row.get("prompt", "")
    prompt_length = row.get("prompt_length") or len(prompt)
    word_count = len(prompt.split())

    # Code detection
    code_keywords = [
        "python",
        "javascript",
        "function",
        "def ",
        "class ",
        "import ",
        "return",
        "const ",
        "let ",
        "var ",
        "```",
        "async",
        "await",
        "try:",
        "except:",
        "if __name__",
    ]
    num_code_keywords = sum(1 for kw in code_keywords if kw.lower() in prompt.lower())
    has_code = 1.0 if (num_code_keywords > 2 or "```" in prompt) else 0.0

    # Question detection
    is_question = 1.0 if prompt.strip().endswith("?") else 0.0
    has_multiple_questions = 1.0 if prompt.count("?") > 1 else 0.0
    has_numbered_list = (
        1.0
        if any(f"{i}." in prompt or f"{i})" in prompt for i in range(1, 10))
        else 0.0
    )

    # Simple question detection
    is_simple_question = (
        1.0 if (is_question and word_count < 20 and not has_code) else 0.0
    )

    # Task type detection
    has_code_task = (
        1.0
        if any(
            kw in prompt.lower()
            for kw in ["code", "implement", "function", "debug", "fix"]
        )
        else 0.0
    )
    has_creative_task = (
        1.0
        if any(
            kw in prompt.lower()
            for kw in ["write", "create", "generate", "story", "poem"]
        )
        else 0.0
    )
    has_analysis_task = (
        1.0
        if any(
            kw in prompt.lower() for kw in ["analyze", "compare", "evaluate", "assess"]
        )
        else 0.0
    )

    return [
        float(prompt_length),
        float(word_count),
        float(num_code_keywords),
        has_code,
        is_question,
        has_multiple_questions,
        has_numbered_list,
        is_simple_question,
        has_code_task,
        has_creative_task,
        has_analysis_task,
    ]


def load_training_data(data_file: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data from a JSON file.

    The JSON file should contain a list of dicts, each with at minimum:
        - prompt: str
        - was_regenerated: bool

    Optional fields: prompt_length (int)

    Args:
        data_file: Path to JSON file with training data.
                   Defaults to TRAINING_DATA_PATH env var or 'training_data.json'.

    Returns:
        X: Feature matrix
        y: Target labels (0 = draft sufficient, 1 = target needed)
    """
    if data_file is None:
        data_file = os.getenv("TRAINING_DATA_PATH", "training_data.json")

    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"Training data file not found: {data_file}\n"
            f"Provide a JSON file with a list of training records."
        )

    print(f"Loading training data from {data_file}...")

    with open(data_file, "r") as f:
        records = json.load(f)

    if not records:
        raise ValueError("No training data found in file")

    print(f"   Found {len(records)} records")

    X = []
    y = []

    for row in records:
        # Extract features
        features = extract_features(row)
        X.append(features)

        # Define target variable
        # y = 1 (target needed) if user regenerated the response
        # y = 0 (draft sufficient) if user accepted the response
        was_regenerated = row.get("was_regenerated", False)
        y.append(1 if was_regenerated else 0)

    X = np.array(X)
    y = np.array(y)

    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Target distribution: {np.bincount(y)} (0=draft_ok, 1=target_needed)")

    return X, y


def train_model(X: np.ndarray, y: np.ndarray) -> Any:
    """
    Train the router classifier.

    Tries multiple classifiers and picks the best one.
    """
    print("\nTraining classifier...")

    # Handle class imbalance
    class_counts = np.bincount(y)
    if len(class_counts) < 2:
        print("Warning: Only one class in training data!")
        print("   Need both regenerated and non-regenerated examples")

        # Create a dummy model that always predicts the majority class
        class DummyModel:
            def predict_proba(self, X):
                return np.array([[1.0, 0.0]] * len(X))

            def predict(self, X):
                return np.zeros(len(X))

        return DummyModel()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Try multiple classifiers
    classifiers = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
    }

    if LGBM_AVAILABLE:
        classifiers["LightGBM"] = LGBMClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, verbose=-1
        )

    best_model = None
    best_score = 0
    best_name = ""

    for name, clf in classifiers.items():
        print(f"\n   Training {name}...")

        try:
            # Use scaled data for LogisticRegression, raw for tree-based
            X_tr = X_train_scaled if name == "LogisticRegression" else X_train
            X_te = X_test_scaled if name == "LogisticRegression" else X_test

            clf.fit(X_tr, y_train)

            # Evaluate
            y_pred = clf.predict(X_te)
            y_prob = clf.predict_proba(X_te)[:, 1]

            # Use ROC-AUC as primary metric
            try:
                auc = roc_auc_score(y_test, y_prob)
            except (ValueError, TypeError):
                auc = 0.5  # Default if AUC can't be computed

            accuracy = (y_pred == y_test).mean()

            print(f"      Accuracy: {accuracy:.3f}")
            print(f"      ROC-AUC: {auc:.3f}")

            if auc > best_score:
                best_score = auc
                best_model = clf
                best_name = name

        except Exception as e:
            print(f"      Failed: {e}")

    print(f"\nBest model: {best_name} (AUC={best_score:.3f})")

    # Print detailed metrics for best model
    if best_model is not None:
        X_te = X_test_scaled if best_name == "LogisticRegression" else X_test
        y_pred = best_model.predict(X_te)

        print("\nClassification Report:")
        print(
            classification_report(
                y_test, y_pred, target_names=["draft_ok", "target_needed"]
            )
        )

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Feature importance (if available)
        if hasattr(best_model, "feature_importances_"):
            feature_names = [
                "prompt_length",
                "word_count",
                "num_code_keywords",
                "has_code",
                "is_question",
                "has_multiple_questions",
                "has_numbered_list",
                "is_simple_question",
                "has_code_task",
                "has_creative_task",
                "has_analysis_task",
            ]
            importances = best_model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]

            print("\nFeature Importance:")
            for idx in sorted_idx[:5]:
                print(f"      {feature_names[idx]}: {importances[idx]:.3f}")

    # Wrap model with scaler if needed
    if best_name == "LogisticRegression":

        class ScaledModel:
            def __init__(self, model, scaler):
                self.model = model
                self.scaler = scaler

            def predict_proba(self, X):
                X_scaled = self.scaler.transform(X)
                return self.model.predict_proba(X_scaled)

            def predict(self, X):
                X_scaled = self.scaler.transform(X)
                return self.model.predict(X_scaled)

        return ScaledModel(best_model, scaler)

    return best_model


def save_model(model: Any, path: str):
    """Save trained model to disk"""
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {path}")


def main():
    """Main training pipeline"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the arbitrage router classifier"
    )
    parser.add_argument(
        "--data-file", type=str, default=None, help="Path to JSON training data file"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save trained model pickle"
    )
    args = parser.parse_args()

    output_path = args.output or OUTPUT_MODEL_PATH

    print("=" * 60)
    print("ARBITRAGE Router Training")
    print(f"Started at: {datetime.now().isoformat()}")
    print("=" * 60)

    if not SKLEARN_AVAILABLE:
        print("scikit-learn is required. Install with: pip install scikit-learn")
        return

    try:
        # Load data
        X, y = load_training_data(data_file=args.data_file)

        # Check minimum data requirements
        if len(X) < 10:
            print(
                f"Warning: Only {len(X)} training samples. Need more data for reliable model."
            )

        # Train model
        model = train_model(X, y)

        # Save model
        save_model(model, output_path)

        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"   Model saved to: {output_path}")
        print("   To enable router mode, set: ARBITRAGE_MODE=router")
        print("=" * 60)

    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
