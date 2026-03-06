"""
Pluggable Storage Backend for Astrai Router.

Replaces Supabase dependency with a clean abstraction.
Ships with MemoryStorage (default) and SQLiteStorage.
PostgresStorage available with `pip install astrai-router[postgres]`.

Usage:
    from astrai_router.storage import get_storage, configure_storage

    # Default: in-memory (no persistence)
    storage = get_storage()

    # SQLite (persists to disk)
    configure_storage("sqlite", path="./router.db")

    # Postgres (production)
    configure_storage("postgres", dsn="postgresql://user:pass@host/db")
"""

from __future__ import annotations

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


# ============================================================================
# ABSTRACT BASE
# ============================================================================

class StorageBackend(ABC):
    """Abstract storage backend for routing data persistence."""

    @abstractmethod
    def get(
        self,
        table: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        gte: Optional[Dict[str, Any]] = None,
        lte: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        desc: bool = False,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query rows from a table with optional filtering."""
        ...

    @abstractmethod
    def insert(self, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert a row. Returns the inserted data."""
        ...

    @abstractmethod
    def upsert(
        self, table: str, data: Dict[str, Any], on_conflict: str = "id"
    ) -> Dict[str, Any]:
        """Insert or update a row. Returns the upserted data."""
        ...

    @abstractmethod
    def update(
        self, table: str, data: Dict[str, Any], filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Update rows matching filters. Returns updated rows."""
        ...

    @abstractmethod
    def delete(self, table: str, filters: Dict[str, Any]) -> int:
        """Delete rows matching filters. Returns count deleted."""
        ...

    @abstractmethod
    def count(self, table: str, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count rows matching filters."""
        ...

    def rpc(self, function_name: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Call a stored procedure/function. Override in backends that support it."""
        logger.warning(f"RPC '{function_name}' not supported by {type(self).__name__}")
        return None

    def close(self) -> None:
        """Clean up resources."""
        pass


# ============================================================================
# MEMORY STORAGE (default, no persistence)
# ============================================================================

class MemoryStorage(StorageBackend):
    """
    In-memory storage backend. Fast, no dependencies, no persistence.

    Data is lost when the process exits. Perfect for:
    - Testing
    - Short-lived scripts
    - Stateless deployments where you don't need routing history
    """

    def __init__(self) -> None:
        self._tables: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._lock = threading.Lock()
        self._id_counters: Dict[str, int] = defaultdict(int)

    def _match(self, row: Dict[str, Any], filters: Optional[Dict[str, Any]],
               gte: Optional[Dict[str, Any]] = None, lte: Optional[Dict[str, Any]] = None) -> bool:
        if filters:
            for k, v in filters.items():
                if row.get(k) != v:
                    return False
        if gte:
            for k, v in gte.items():
                if row.get(k) is None or row.get(k) < v:
                    return False
        if lte:
            for k, v in lte.items():
                if row.get(k) is None or row.get(k) > v:
                    return False
        return True

    def get(
        self,
        table: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        gte: Optional[Dict[str, Any]] = None,
        lte: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        desc: bool = False,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            rows = [
                deepcopy(r) for r in self._tables[table]
                if self._match(r, filters, gte, lte)
            ]
        if order_by:
            rows.sort(key=lambda r: r.get(order_by, ""), reverse=desc)
        if limit:
            rows = rows[:limit]
        return rows

    def insert(self, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        row = deepcopy(data)
        with self._lock:
            if "id" not in row:
                self._id_counters[table] += 1
                row["id"] = self._id_counters[table]
            self._tables[table].append(row)
        return row

    def upsert(
        self, table: str, data: Dict[str, Any], on_conflict: str = "id"
    ) -> Dict[str, Any]:
        conflict_val = data.get(on_conflict)
        with self._lock:
            for i, row in enumerate(self._tables[table]):
                if row.get(on_conflict) == conflict_val:
                    updated = {**row, **data}
                    self._tables[table][i] = updated
                    return deepcopy(updated)
            # No existing row — insert
            row = deepcopy(data)
            if "id" not in row:
                self._id_counters[table] += 1
                row["id"] = self._id_counters[table]
            self._tables[table].append(row)
            return deepcopy(row)

    def update(
        self, table: str, data: Dict[str, Any], filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        updated = []
        with self._lock:
            for i, row in enumerate(self._tables[table]):
                if self._match(row, filters):
                    self._tables[table][i] = {**row, **data}
                    updated.append(deepcopy(self._tables[table][i]))
        return updated

    def delete(self, table: str, filters: Dict[str, Any]) -> int:
        with self._lock:
            before = len(self._tables[table])
            self._tables[table] = [
                r for r in self._tables[table] if not self._match(r, filters)
            ]
            return before - len(self._tables[table])

    def count(self, table: str, filters: Optional[Dict[str, Any]] = None) -> int:
        with self._lock:
            if not filters:
                return len(self._tables[table])
            return sum(1 for r in self._tables[table] if self._match(r, filters))


# ============================================================================
# SQLITE STORAGE
# ============================================================================

class SQLiteStorage(StorageBackend):
    """
    SQLite-backed storage. Persists to a single file.

    Tables are created on-the-fly using a flexible JSON column approach:
    each table has (id INTEGER PRIMARY KEY, key TEXT, data JSON, created_at REAL).
    """

    def __init__(self, path: str = "astrai_router.db") -> None:
        import sqlite3
        self._path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.Lock()
        self._ensured_tables: set = set()

    def _ensure_table(self, table: str) -> None:
        if table in self._ensured_tables:
            return
        with self._lock:
            self._conn.execute(f"""
                CREATE TABLE IF NOT EXISTS [{table}] (
                    _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                    data JSON NOT NULL,
                    created_at REAL DEFAULT (unixepoch('now'))
                )
            """)
            self._conn.commit()
            self._ensured_tables.add(table)

    def _row_to_dict(self, data_json: str) -> Dict[str, Any]:
        return json.loads(data_json)

    def get(
        self,
        table: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        gte: Optional[Dict[str, Any]] = None,
        lte: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        desc: bool = False,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        self._ensure_table(table)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT data FROM [{table}]"
            ).fetchall()

        result = []
        for (data_json,) in rows:
            row = self._row_to_dict(data_json)
            match = True
            if filters:
                for k, v in filters.items():
                    if row.get(k) != v:
                        match = False
                        break
            if match and gte:
                for k, v in gte.items():
                    if row.get(k) is None or row.get(k) < v:
                        match = False
                        break
            if match and lte:
                for k, v in lte.items():
                    if row.get(k) is None or row.get(k) > v:
                        match = False
                        break
            if match:
                result.append(row)

        if order_by:
            result.sort(key=lambda r: r.get(order_by, ""), reverse=desc)
        if limit:
            result = result[:limit]
        return result

    def insert(self, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_table(table)
        row = deepcopy(data)
        if "created_at" not in row:
            row["created_at"] = time.time()
        with self._lock:
            cursor = self._conn.execute(
                f"INSERT INTO [{table}] (data) VALUES (?)",
                (json.dumps(row),)
            )
            self._conn.commit()
            if "id" not in row:
                row["id"] = cursor.lastrowid
                # Update stored data with the id
                self._conn.execute(
                    f"UPDATE [{table}] SET data = ? WHERE _rowid = ?",
                    (json.dumps(row), cursor.lastrowid)
                )
                self._conn.commit()
        return row

    def upsert(
        self, table: str, data: Dict[str, Any], on_conflict: str = "id"
    ) -> Dict[str, Any]:
        self._ensure_table(table)
        conflict_val = data.get(on_conflict)
        if conflict_val is not None:
            existing = self.get(table, filters={on_conflict: conflict_val}, limit=1)
            if existing:
                merged = {**existing[0], **data}
                with self._lock:
                    # Find and update the row
                    rows = self._conn.execute(
                        f"SELECT _rowid, data FROM [{table}]"
                    ).fetchall()
                    for rowid, data_json in rows:
                        row = json.loads(data_json)
                        if row.get(on_conflict) == conflict_val:
                            self._conn.execute(
                                f"UPDATE [{table}] SET data = ? WHERE _rowid = ?",
                                (json.dumps(merged), rowid)
                            )
                            self._conn.commit()
                            return merged
        return self.insert(table, data)

    def update(
        self, table: str, data: Dict[str, Any], filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        self._ensure_table(table)
        updated = []
        with self._lock:
            rows = self._conn.execute(
                f"SELECT _rowid, data FROM [{table}]"
            ).fetchall()
            for rowid, data_json in rows:
                row = json.loads(data_json)
                match = all(row.get(k) == v for k, v in filters.items())
                if match:
                    merged = {**row, **data}
                    self._conn.execute(
                        f"UPDATE [{table}] SET data = ? WHERE _rowid = ?",
                        (json.dumps(merged), rowid)
                    )
                    updated.append(merged)
            if updated:
                self._conn.commit()
        return updated

    def delete(self, table: str, filters: Dict[str, Any]) -> int:
        self._ensure_table(table)
        count = 0
        with self._lock:
            rows = self._conn.execute(
                f"SELECT _rowid, data FROM [{table}]"
            ).fetchall()
            for rowid, data_json in rows:
                row = json.loads(data_json)
                if all(row.get(k) == v for k, v in filters.items()):
                    self._conn.execute(
                        f"DELETE FROM [{table}] WHERE _rowid = ?", (rowid,)
                    )
                    count += 1
            if count:
                self._conn.commit()
        return count

    def count(self, table: str, filters: Optional[Dict[str, Any]] = None) -> int:
        return len(self.get(table, filters=filters))

    def close(self) -> None:
        self._conn.close()


# ============================================================================
# POSTGRES STORAGE (optional)
# ============================================================================

class PostgresStorage(StorageBackend):
    """
    PostgreSQL-backed storage. Requires `pip install astrai-router[postgres]`.

    Uses JSONB columns for flexible schema, similar to SQLiteStorage.
    """

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            raise ImportError(
                "PostgresStorage requires psycopg2. "
                "Install with: pip install astrai-router[postgres]"
            )
        self._dsn = dsn
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = True
        self._lock = threading.Lock()
        self._ensured_tables: set = set()

    def _ensure_table(self, table: str) -> None:
        if table in self._ensured_tables:
            return
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        _rowid SERIAL PRIMARY KEY,
                        data JSONB NOT NULL,
                        created_at DOUBLE PRECISION DEFAULT EXTRACT(EPOCH FROM NOW())
                    )
                """)
            self._ensured_tables.add(table)

    def get(
        self,
        table: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        gte: Optional[Dict[str, Any]] = None,
        lte: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        desc: bool = False,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        self._ensure_table(table)
        import psycopg2.extras

        conditions = []
        params: list = []
        if filters:
            for k, v in filters.items():
                conditions.append(f"data->>%s = %s")
                params.extend([k, str(v)])
        if gte:
            for k, v in gte.items():
                conditions.append(f"data->>%s >= %s")
                params.extend([k, str(v)])
        if lte:
            for k, v in lte.items():
                conditions.append(f"data->>%s <= %s")
                params.extend([k, str(v)])

        where = " AND ".join(conditions) if conditions else "TRUE"
        query = f"SELECT data FROM {table} WHERE {where}"
        if order_by:
            direction = "DESC" if desc else "ASC"
            query += f" ORDER BY data->>'{order_by}' {direction}"
        if limit:
            query += f" LIMIT {limit}"

        with self._lock:
            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, params)
                return [row["data"] for row in cur.fetchall()]

    def insert(self, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_table(table)
        row = deepcopy(data)
        if "created_at" not in row:
            row["created_at"] = time.time()
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {table} (data) VALUES (%s) RETURNING _rowid",
                    (json.dumps(row),)
                )
                rowid = cur.fetchone()[0]
                if "id" not in row:
                    row["id"] = rowid
                    cur.execute(
                        f"UPDATE {table} SET data = %s WHERE _rowid = %s",
                        (json.dumps(row), rowid)
                    )
        return row

    def upsert(
        self, table: str, data: Dict[str, Any], on_conflict: str = "id"
    ) -> Dict[str, Any]:
        conflict_val = data.get(on_conflict)
        if conflict_val is not None:
            existing = self.get(table, filters={on_conflict: conflict_val}, limit=1)
            if existing:
                merged = {**existing[0], **data}
                self.update(table, merged, {on_conflict: conflict_val})
                return merged
        return self.insert(table, data)

    def update(
        self, table: str, data: Dict[str, Any], filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        self._ensure_table(table)
        conditions = []
        params: list = []
        for k, v in filters.items():
            conditions.append(f"data->>%s = %s")
            params.extend([k, str(v)])

        where = " AND ".join(conditions) if conditions else "TRUE"
        with self._lock:
            with self._conn.cursor() as cur:
                # Get existing rows
                cur.execute(f"SELECT _rowid, data FROM {table} WHERE {where}", params)
                rows = cur.fetchall()
                updated = []
                for rowid, existing_data in rows:
                    merged = {**existing_data, **data}
                    cur.execute(
                        f"UPDATE {table} SET data = %s WHERE _rowid = %s",
                        (json.dumps(merged), rowid)
                    )
                    updated.append(merged)
        return updated

    def delete(self, table: str, filters: Dict[str, Any]) -> int:
        self._ensure_table(table)
        conditions = []
        params: list = []
        for k, v in filters.items():
            conditions.append(f"data->>%s = %s")
            params.extend([k, str(v)])

        where = " AND ".join(conditions) if conditions else "FALSE"
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(f"DELETE FROM {table} WHERE {where} RETURNING _rowid", params)
                return cur.rowcount

    def count(self, table: str, filters: Optional[Dict[str, Any]] = None) -> int:
        return len(self.get(table, filters=filters))

    def close(self) -> None:
        self._conn.close()


# ============================================================================
# GLOBAL SINGLETON
# ============================================================================

_storage: Optional[StorageBackend] = None
_storage_lock = threading.Lock()


def configure_storage(
    backend: str = "memory",
    *,
    path: str = "astrai_router.db",
    dsn: str = "",
    instance: Optional[StorageBackend] = None,
) -> StorageBackend:
    """
    Configure the global storage backend.

    Args:
        backend: "memory", "sqlite", or "postgres"
        path: File path for SQLite backend
        dsn: Connection string for Postgres backend
        instance: Pre-built StorageBackend instance (overrides backend arg)

    Returns:
        The configured storage backend.
    """
    global _storage
    with _storage_lock:
        if instance is not None:
            _storage = instance
        elif backend == "memory":
            _storage = MemoryStorage()
        elif backend == "sqlite":
            _storage = SQLiteStorage(path=path)
        elif backend == "postgres":
            _storage = PostgresStorage(dsn=dsn)
        else:
            raise ValueError(f"Unknown backend: {backend!r}. Use 'memory', 'sqlite', or 'postgres'.")
        logger.info(f"Storage configured: {type(_storage).__name__}")
        return _storage


def get_storage() -> StorageBackend:
    """Get the global storage backend. Defaults to MemoryStorage if not configured."""
    global _storage
    if _storage is None:
        with _storage_lock:
            if _storage is None:
                _storage = MemoryStorage()
    return _storage
