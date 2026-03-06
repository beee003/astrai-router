import asyncio
import json
import logging
import os
import pickle
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .storage import get_storage

logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import LogisticRegression

    SKLEARN_AVAILABLE = True
except Exception:
    LogisticRegression = None
    SKLEARN_AVAILABLE = False

ASTRAI_HOME = Path(os.path.expanduser("~/.astrai"))
INTEL_DIR = ASTRAI_HOME / "routing_intelligence"
QUALITY_MATRIX_PATH = INTEL_DIR / "quality_matrix.json"
CLASSIFIER_PATH = INTEL_DIR / "route_classifier.pkl"
TRAINING_DATA_PATH = INTEL_DIR / "route_training_data.jsonl"


class QualityLearner:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        INTEL_DIR.mkdir(parents=True, exist_ok=True)
        self.outcomes: List[Dict[str, Any]] = []
        self.quality_matrix: Dict[str, Dict[str, Dict[str, float]]] = (
            self._load_quality_matrix()
        )
        self.classifier: Optional[Any] = self._load_classifier()
        self.training_samples: int = self._count_training_samples()

    def _load_quality_matrix(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        matrix = self._load_quality_matrix_from_storage()
        if matrix is not None:
            return matrix

        if not QUALITY_MATRIX_PATH.exists():
            return {}
        try:
            return json.loads(QUALITY_MATRIX_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _load_quality_matrix_from_storage(
        self,
    ) -> Optional[Dict[str, Dict[str, Dict[str, float]]]]:
        try:
            storage = get_storage()
            rows = storage.get("quality_matrix", limit=10000)
            if not rows:
                return None
            matrix: Dict[str, Dict[str, Dict[str, float]]] = {}
            for row in rows:
                task_type = row.get("task_type")
                model = row.get("model")
                if not task_type or not model:
                    continue
                matrix.setdefault(task_type, {})
                matrix[task_type][model] = {
                    "avg_latency": float(row.get("avg_latency") or 0.0),
                    "p95_latency": float(row.get("p95_latency") or 0.0),
                    "error_rate": float(row.get("error_rate") or 0.0),
                    "avg_quality_score": float(row.get("avg_quality_score") or 0.0),
                    "sample_count": int(row.get("sample_count") or 0),
                    "confidence": float(row.get("confidence") or 0.0),
                }
            logger.info(f"Loaded quality matrix from storage ({len(rows)} rows)")
            return matrix
        except Exception as e:
            logger.warning(f"Failed to load quality matrix from storage: {e}")
            return None

    def _persist_quality_matrix(
        self, matrix: Dict[str, Dict[str, Dict[str, float]]]
    ) -> bool:
        try:
            storage = get_storage()
            now_iso = datetime.now(timezone.utc).isoformat()
            for task_type, model_data in matrix.items():
                for model, stats in model_data.items():
                    storage.upsert(
                        "quality_matrix",
                        {
                            "task_type": task_type,
                            "model": model,
                            "avg_latency": float(stats.get("avg_latency", 0.0)),
                            "p95_latency": float(stats.get("p95_latency", 0.0)),
                            "error_rate": float(stats.get("error_rate", 0.0)),
                            "avg_quality_score": float(
                                stats.get("avg_quality_score", 0.0)
                            ),
                            "sample_count": int(stats.get("sample_count", 0)),
                            "confidence": float(stats.get("confidence", 0.0)),
                            "updated_at": now_iso,
                        },
                        on_conflict="task_type,model",
                    )
            return True
        except Exception as e:
            logger.warning(f"Failed to persist quality matrix to storage: {e}")
            return False

    def _load_classifier(self) -> Optional[Any]:
        if not CLASSIFIER_PATH.exists():
            return None
        try:
            with CLASSIFIER_PATH.open("rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _count_training_samples(self) -> int:
        if not TRAINING_DATA_PATH.exists():
            return 0
        with TRAINING_DATA_PATH.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    def _append_training_sample(self, sample: Dict[str, Any]) -> None:
        with TRAINING_DATA_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(sample) + "\n")
        self.training_samples += 1

    def record_outcome(self, metadata: Dict[str, Any]) -> None:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "task_type": metadata.get("task_type", "general"),
            "model": metadata.get("model", "unknown"),
            "provider": metadata.get("provider", "unknown"),
            "first_token_latency_ms": float(
                metadata.get("first_token_latency_ms")
                or metadata.get("latency_ms")
                or 0
            ),
            "tokens_per_second": float(metadata.get("tokens_per_second") or 0),
            "prompt_length": int(metadata.get("prompt_length") or 0),
            "completion_length": int(metadata.get("completion_length") or 0),
            "error": bool(metadata.get("error", False)),
            "user_continued_conversation": bool(
                metadata.get("user_continued_conversation", False)
            ),
            "explicit_feedback": metadata.get("explicit_feedback"),
            "cost": float(metadata.get("cost", 0.0)),
            "features": metadata.get("features", {}),
        }
        ratio = payload["completion_length"] / max(payload["prompt_length"], 1)
        base_quality = 0.5
        if not payload["error"]:
            base_quality += 0.2
        base_quality += min(ratio, 2.0) * 0.1
        base_quality += 0.1 if payload["user_continued_conversation"] else 0.0
        if payload["explicit_feedback"] is not None:
            base_quality += (float(payload["explicit_feedback"]) - 3.0) / 10.0
        payload["quality_score"] = float(max(0.0, min(1.0, base_quality)))

        self.outcomes.append(payload)
        self._append_training_sample(payload)

    async def update_quality_matrix(self) -> Dict[str, Any]:
        async with self._lock:
            grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for item in self.outcomes:
                grouped[item["task_type"]][item["model"]].append(item)

            matrix: Dict[str, Dict[str, Dict[str, float]]] = {}
            for task_type, model_data in grouped.items():
                matrix[task_type] = {}
                for model, rows in model_data.items():
                    latencies = [r["first_token_latency_ms"] for r in rows]
                    errors = [1.0 if r["error"] else 0.0 for r in rows]
                    qualities = [r["quality_score"] for r in rows]
                    sample_count = len(rows)
                    confidence = min(1.0, sample_count / 100.0)
                    matrix[task_type][model] = {
                        "avg_latency": float(np.mean(latencies)) if latencies else 0.0,
                        "p95_latency": float(np.percentile(latencies, 95))
                        if latencies
                        else 0.0,
                        "error_rate": float(np.mean(errors)) if errors else 0.0,
                        "avg_quality_score": float(np.mean(qualities))
                        if qualities
                        else 0.0,
                        "sample_count": sample_count,
                        "confidence": confidence,
                    }

            self.quality_matrix = matrix
            persisted = self._persist_quality_matrix(matrix)
            if not persisted:
                QUALITY_MATRIX_PATH.write_text(
                    json.dumps(matrix, indent=2), encoding="utf-8"
                )
            return {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "task_types": len(matrix),
                "samples": len(self.outcomes),
            }

    def get_strong_recommendation(
        self, task_type: str, candidate_models: List[str]
    ) -> Optional[Dict[str, Any]]:
        task_matrix = self.quality_matrix.get(task_type, {})
        best_model = None
        best = None
        for model in candidate_models:
            data = task_matrix.get(model)
            if not data:
                continue
            if data.get("confidence", 0) < 0.8 or data.get("sample_count", 0) <= 50:
                continue
            if best is None or data.get("avg_quality_score", 0) > best.get(
                "avg_quality_score", 0
            ):
                best = data
                best_model = model
        if not best_model or not best:
            return None
        return {"model": best_model, **best}

    def should_override(
        self,
        task_type: str,
        chosen: str,
        alternative: str,
        chosen_cost: float,
        alt_cost: float,
    ) -> bool:
        task_matrix = self.quality_matrix.get(task_type, {})
        chosen_data = task_matrix.get(chosen)
        alt_data = task_matrix.get(alternative)
        if not chosen_data or not alt_data:
            return False
        quality_delta = alt_data.get("avg_quality_score", 0) - chosen_data.get(
            "avg_quality_score", 0
        )
        if chosen_data.get("avg_quality_score", 0) <= 0:
            return False
        return (
            quality_delta >= chosen_data["avg_quality_score"] * 0.2
            and alt_cost < chosen_cost
        )

    async def retrain_classifier(self) -> Dict[str, Any]:
        async with self._lock:
            if not SKLEARN_AVAILABLE:
                return {"trained": False, "reason": "sklearn_unavailable"}
            if self.training_samples < 50:
                return {
                    "trained": False,
                    "reason": "insufficient_samples",
                    "samples": self.training_samples,
                }

            X: List[List[float]] = []
            y: List[str] = []
            with TRAINING_DATA_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    feat = self._feature_vector(row)
                    X.append(feat)
                    y.append(row.get("model", "unknown"))

            if not X:
                return {"trained": False, "reason": "empty_training_data"}

            clf = LogisticRegression(max_iter=500, multi_class="auto")
            clf.fit(np.array(X), np.array(y))
            with CLASSIFIER_PATH.open("wb") as f:
                pickle.dump(clf, f)
            self.classifier = clf
            return {"trained": True, "samples": len(X)}

    def _feature_vector(self, row: Dict[str, Any]) -> List[float]:
        task_map = {
            "general": 0,
            "code": 1,
            "reasoning": 2,
            "analysis": 3,
            "writing": 4,
        }
        task_type = row.get("task_type", "general")
        features = row.get("features") or {}
        return [
            float(task_map.get(task_type, 0)),
            float(row.get("prompt_length", 0)),
            float(features.get("has_code", 0)),
            float(features.get("has_math", 0)),
            float(features.get("language_code", 0)),
            float(features.get("complexity_score", 0)),
            float(features.get("time_of_day", 0)),
            float(features.get("user_tier", 0)),
        ]

    def classifier_prediction(self, features: Dict[str, Any]) -> Optional[str]:
        if self.classifier is None:
            return None
        vec = np.array(
            [
                [
                    float(features.get("task_type_code", 0)),
                    float(features.get("prompt_length", 0)),
                    float(features.get("has_code", 0)),
                    float(features.get("has_math", 0)),
                    float(features.get("language_code", 0)),
                    float(features.get("complexity_score", 0)),
                    float(features.get("time_of_day", 0)),
                    float(features.get("user_tier", 0)),
                ]
            ]
        )
        return str(self.classifier.predict(vec)[0])

    def get_quality_matrix(self) -> Dict[str, Any]:
        return self.quality_matrix


_QUALITY_LEARNER: Optional[QualityLearner] = None


def get_quality_learner() -> QualityLearner:
    global _QUALITY_LEARNER
    if _QUALITY_LEARNER is None:
        _QUALITY_LEARNER = QualityLearner()
    return _QUALITY_LEARNER
