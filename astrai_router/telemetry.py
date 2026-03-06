"""
Routing Telemetry - Persistent storage for routing decisions and execution metrics.

Stores detailed telemetry for each request including:
- Request identification (request_id, timestamp, customer_id)
- Workflow context (workflow_id, step_id)
- Routing decisions (chosen_provider, chosen_model, candidates_considered)
- Pricing (prices for each candidate)
- Performance (latency_ms, tokens_in, tokens_out)
- Costs (cost_usd_est, cost_usd_actual)
- Quality (gate_results, fallback_count, final_status)
"""

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pydantic import BaseModel

from .storage import get_storage

# Fallback in-memory cache when storage is unavailable (e.g., local dev / demo)
TELEMETRY_CACHE: Dict[str, Dict[str, Any]] = {}
TELEMETRY_CACHE_LIMIT = 500  # simple cap to avoid unbounded growth


def _telemetry_backend() -> str:
    backend = os.getenv("ASTRAI_TELEMETRY_BACKEND")
    if backend:
        return backend.lower()
    if os.getenv("ASTRAI_LOCAL_MODE") == "1":
        return "sqlite"
    # Default to pluggable storage backend
    return "storage"


def _telemetry_mode() -> str:
    mode = os.getenv("ASTRAI_TELEMETRY_MODE")
    if mode:
        return mode.strip().lower()
    if os.getenv("ASTRAI_LOCAL_MODE") == "1":
        return "metadata"
    return "metadata"


def _sqlite_db_path() -> str:
    path = os.getenv("ASTRAI_TELEMETRY_DB", "./data/astrai_telemetry.sqlite")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except PermissionError:
        path = "/tmp/astrai_data/astrai_telemetry.sqlite"
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _ensure_sqlite_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS routing_telemetry (
            request_id TEXT PRIMARY KEY,
            timestamp TEXT,
            customer_id TEXT,
            workflow_id TEXT,
            step_id TEXT,
            chosen_provider TEXT,
            chosen_model TEXT,
            candidates_considered TEXT,
            prices TEXT,
            latency_ms REAL,
            tokens_in INTEGER,
            tokens_out INTEGER,
            cost_usd_est REAL,
            cost_usd_actual REAL,
            gate_results TEXT,
            fallback_count INTEGER,
            final_status TEXT,
            audit_hash TEXT,
            prompt_text TEXT,
            response_text TEXT,
            retention_mode TEXT
        )
        """
    )
    # Add new columns for existing DBs (safe no-op if already present)
    for column in ("prompt_text", "response_text", "retention_mode", "audit_hash"):
        try:
            conn.execute(f"ALTER TABLE routing_telemetry ADD COLUMN {column} TEXT")
        except Exception:
            pass
    conn.commit()


@dataclass
class RoutingTelemetryRecord:
    """A single routing telemetry record."""

    request_id: str
    timestamp: str
    customer_id: str
    workflow_id: Optional[str]
    step_id: Optional[str]
    chosen_provider: str
    chosen_model: str
    candidates_considered: List[Dict[str, Any]]
    prices: Dict[str, float]
    latency_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd_est: float
    cost_usd_actual: float
    gate_results: Dict[str, Any]
    fallback_count: int
    final_status: str


class TelemetryQueryParams(BaseModel):
    """Query parameters for telemetry endpoint."""

    limit: int = 50
    offset: int = 0
    customer_id: Optional[str] = None
    workflow_id: Optional[str] = None
    status: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"req-{uuid.uuid4().hex[:16]}"


async def store_routing_telemetry(
    request_id: str,
    customer_id: str,
    chosen_provider: str,
    chosen_model: str,
    candidates_considered: List[Dict[str, Any]],
    prices: Dict[str, float],
    latency_ms: float,
    tokens_in: int,
    tokens_out: int,
    cost_usd_est: float,
    cost_usd_actual: float,
    gate_results: Dict[str, Any],
    fallback_count: int,
    final_status: str,
    workflow_id: Optional[str] = None,
    step_id: Optional[str] = None,
    prompt_hash: Optional[str] = None,
    prompt_text: Optional[str] = None,
    response_text: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Store a routing telemetry record to the database.

    Args:
        request_id: Unique identifier for the request
        customer_id: Customer/account ID
        chosen_provider: The provider that was selected
        chosen_model: The model that was executed
        candidates_considered: List of candidate providers/models considered
        prices: Price quotes from each candidate
        latency_ms: Total latency in milliseconds
        tokens_in: Input token count
        tokens_out: Output token count
        cost_usd_est: Estimated cost before execution
        cost_usd_actual: Actual cost after execution
        gate_results: Quality gate verification results
        fallback_count: Number of fallback attempts
        final_status: Final status (success, failed, escalated, etc.)
        workflow_id: Optional workflow identifier
        step_id: Optional workflow step identifier

    Returns:
        The stored record or None if storage failed
    """
    mode = _telemetry_mode()
    if mode == "none":
        return None

    gate_results = gate_results or {}
    audit_payload = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "customer_id": customer_id,
        "chosen_provider": chosen_provider,
        "chosen_model": chosen_model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd_est": cost_usd_est,
        "cost_usd_actual": cost_usd_actual,
        "prompt_hash": prompt_hash,
    }
    audit_hash = hashlib.sha256(
        json.dumps(audit_payload, sort_keys=True).encode()
    ).hexdigest()
    gate_results.setdefault("audit", {})
    gate_results["audit"].update(
        {
            "prompt_hash": prompt_hash,
            "record_hash": audit_hash,
        }
    )

    retention_mode = "metadata"
    prompt_blob = None
    response_blob = None
    if mode == "full":
        try:
            # Try to import encryption utilities if available
            from .encryption import encrypt_api_key, get_encryption_key

            _ = get_encryption_key()
            if prompt_text:
                prompt_blob = encrypt_api_key(prompt_text)
            if response_text:
                response_blob = encrypt_api_key(response_text)
            retention_mode = "full_encrypted"
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                f"Telemetry full retention requested but encryption not configured: {e}"
            )
            retention_mode = "metadata"

    record = {
        "request_id": request_id,
        "timestamp": audit_payload["timestamp"],
        "customer_id": customer_id,
        "workflow_id": workflow_id,
        "step_id": step_id,
        "chosen_provider": chosen_provider,
        "chosen_model": chosen_model,
        "candidates_considered": candidates_considered,
        "prices": prices,
        "latency_ms": latency_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd_est": cost_usd_est,
        "cost_usd_actual": cost_usd_actual,
        "gate_results": gate_results,
        "fallback_count": fallback_count,
        "final_status": final_status,
        "audit_hash": audit_hash,
        "prompt_text": prompt_blob,
        "response_text": response_blob,
        "retention_mode": retention_mode,
    }

    # Always store in memory for fast access / smoke tests
    if len(TELEMETRY_CACHE) >= TELEMETRY_CACHE_LIMIT:
        TELEMETRY_CACHE.pop(next(iter(TELEMETRY_CACHE)))
    TELEMETRY_CACHE[request_id] = record

    backend = _telemetry_backend()
    if backend == "sqlite":
        try:
            db_path = _sqlite_db_path()
            dir_path = os.path.dirname(db_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            conn = sqlite3.connect(db_path)
            _ensure_sqlite_schema(conn)
            conn.execute(
                """
                INSERT OR REPLACE INTO routing_telemetry (
                    request_id, timestamp, customer_id, workflow_id, step_id,
                    chosen_provider, chosen_model, candidates_considered, prices,
                    latency_ms, tokens_in, tokens_out, cost_usd_est, cost_usd_actual,
                    gate_results, fallback_count, final_status, audit_hash,
                    prompt_text, response_text, retention_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["request_id"],
                    record["timestamp"],
                    record["customer_id"],
                    record["workflow_id"],
                    record["step_id"],
                    record["chosen_provider"],
                    record["chosen_model"],
                    json.dumps(record["candidates_considered"]),
                    json.dumps(record["prices"]),
                    record["latency_ms"],
                    record["tokens_in"],
                    record["tokens_out"],
                    record["cost_usd_est"],
                    record["cost_usd_actual"],
                    json.dumps(record["gate_results"]),
                    record["fallback_count"],
                    record["final_status"],
                    record["audit_hash"],
                    record["prompt_text"],
                    record["response_text"],
                    record["retention_mode"],
                ),
            )
            conn.commit()
            conn.close()
            return record
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Telemetry SQLite Error: {e}")
            return None

    if backend == "storage":
        try:
            storage = get_storage()
            record_db = dict(record)
            record_db.pop("audit_hash", None)
            record_db.pop("prompt_text", None)
            record_db.pop("response_text", None)
            record_db.pop("retention_mode", None)
            storage.insert("routing_telemetry", record_db)
            return record
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Telemetry Storage Error: {e}")
            return None

    # No persistent store; in-memory only
    return record


async def query_routing_telemetry(
    limit: int = 50,
    offset: int = 0,
    customer_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    status: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Query routing telemetry records.

    Args:
        limit: Maximum number of records to return (default 50)
        offset: Number of records to skip (for pagination)
        customer_id: Filter by customer ID
        workflow_id: Filter by workflow ID
        status: Filter by final status
        provider: Filter by chosen provider
        model: Filter by chosen model

    Returns:
        List of telemetry records matching the query
    """
    if _telemetry_mode() == "none":
        return []
    backend = _telemetry_backend()
    if backend == "sqlite":
        try:
            db_path = _sqlite_db_path()
            if not os.path.exists(db_path):
                return []
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            _ensure_sqlite_schema(conn)
            clauses = []
            params: List[Any] = []
            if customer_id:
                clauses.append("customer_id = ?")
                params.append(customer_id)
            if workflow_id:
                clauses.append("workflow_id = ?")
                params.append(workflow_id)
            if status:
                clauses.append("final_status = ?")
                params.append(status)
            if provider:
                clauses.append("chosen_provider = ?")
                params.append(provider)
            if model:
                clauses.append("chosen_model = ?")
                params.append(model)
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            query = f"SELECT * FROM routing_telemetry {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            rows = conn.execute(query, params).fetchall()
            conn.close()
            results = []
            for row in rows:
                item = dict(row)
                item["candidates_considered"] = json.loads(
                    item.get("candidates_considered") or "[]"
                )
                item["prices"] = json.loads(item.get("prices") or "{}")
                item["gate_results"] = json.loads(item.get("gate_results") or "{}")
                results.append(item)
            return results
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Telemetry SQLite Query Error: {e}")
            return []

    if backend == "storage":
        try:
            storage = get_storage()
            filters: Dict[str, Any] = {}
            if customer_id:
                filters["customer_id"] = customer_id
            if workflow_id:
                filters["workflow_id"] = workflow_id
            if status:
                filters["final_status"] = status
            if provider:
                filters["chosen_provider"] = provider
            if model:
                filters["chosen_model"] = model

            rows = storage.get(
                "routing_telemetry",
                filters=filters if filters else None,
                order_by="timestamp",
                desc=True,
                limit=limit,
            )
            # Handle offset manually since storage.get doesn't support offset
            if offset:
                rows = rows[offset:]
            return rows if rows else []
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Telemetry Query Error: {e}")
            return []

    # Return newest first from in-memory cache
    return list(TELEMETRY_CACHE.values())[::-1][offset : offset + limit]


async def get_telemetry_by_request_id(request_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a single telemetry record by request ID.

    Args:
        request_id: The request ID to look up

    Returns:
        The telemetry record or None if not found
    """
    # Fallback to cache first (fast path)
    cached = TELEMETRY_CACHE.get(request_id)
    if cached:
        return cached
    if _telemetry_mode() == "none":
        return None
    backend = _telemetry_backend()
    if backend == "sqlite":
        try:
            db_path = _sqlite_db_path()
            if not os.path.exists(db_path):
                return None
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            _ensure_sqlite_schema(conn)
            row = conn.execute(
                "SELECT * FROM routing_telemetry WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            conn.close()
            if not row:
                return None
            record = dict(row)
            record["candidates_considered"] = json.loads(
                record.get("candidates_considered") or "[]"
            )
            record["prices"] = json.loads(record.get("prices") or "{}")
            record["gate_results"] = json.loads(record.get("gate_results") or "{}")
            return record
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Telemetry SQLite Lookup Error: {e}")
            return None

    if backend != "storage":
        return None

    try:
        storage = get_storage()
        rows = storage.get(
            "routing_telemetry",
            filters={"request_id": request_id},
            limit=1,
        )
        if rows:
            return rows[0]
        return TELEMETRY_CACHE.get(request_id)
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(f"Telemetry Lookup Error: {e}")
        return TELEMETRY_CACHE.get(request_id)
