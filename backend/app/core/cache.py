import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable, Optional


class DiskCache:
    """
    Supports both Zero-Storage Stateless Mode (enabled=False, 0 bytes written to disk)
    and Offline CI Caching Mode.
    """

    def __init__(self, db_path: str = ".cache/rag_doctor_cache.sqlite", enabled: bool = False):
        self.enabled = enabled
        self._lock = threading.Lock()
        if self.enabled:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kv_cache (
                    cache_key TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    @staticmethod
    def make_key(namespace: str, payload: Any) -> str:
        serialized = json.dumps(payload, sort_keys=True, default=str)
        digest = hashlib.sha256(f"{namespace}:{serialized}".encode("utf-8")).hexdigest()
        return digest

    def get(self, namespace: str, payload: Any) -> Optional[Any]:
        if not self.enabled:
            return None
        key = self.make_key(namespace, payload)
        with self._lock, sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT payload_json FROM kv_cache WHERE cache_key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            return json.loads(row[0])

    def set(self, namespace: str, payload: Any, value: Any) -> None:
        if not self.enabled:
            return
        key = self.make_key(namespace, payload)
        serialized_val = json.dumps(value, default=str)
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO kv_cache (cache_key, namespace, payload_json)
                VALUES (?, ?, ?)
                """,
                (key, namespace, serialized_val),
            )
            conn.commit()

    def get_or_compute(self, namespace: str, payload: Any, compute_fn: Callable[[], Any]) -> Any:
        if not self.enabled:
            return compute_fn()
        cached = self.get(namespace, payload)
        if cached is not None:
            return cached
        result = compute_fn()
        self.set(namespace, payload, result)
        return result
