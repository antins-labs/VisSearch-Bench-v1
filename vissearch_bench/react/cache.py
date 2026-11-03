"""Persistent, content-addressed cache for successful ReAct tool responses."""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple


_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: Dict[str, threading.Lock] = {}


def _lock_for(path: Path) -> threading.Lock:
    key = str(path)
    with _LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _PATH_LOCKS[key] = lock
        return lock


@contextmanager
def _process_lock(path: Path):
    # Keep the inode: unlinking a lock file can split waiting processes across locks.
    lock_path = path.parent / '.locks' / (path.stem + '.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class JsonToolCache:
    """Store successful JSON-compatible responses without expiry."""

    VERSION = 1

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()

    @staticmethod
    def _fingerprint(namespace: str, request: Dict[str, Any]) -> str:
        encoded = json.dumps(
            {
                "namespace": namespace,
                "request": request,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def path_for(self, namespace: str, request: Dict[str, Any]) -> Path:
        digest = self._fingerprint(namespace, request)
        return self.root / namespace / "{}.json".format(digest)

    @staticmethod
    def _read(
        path: Path,
        *,
        namespace: str,
        request: Dict[str, Any],
    ) -> Any:
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(record, dict)
            or record.get("version") != JsonToolCache.VERSION
            or record.get("namespace") != namespace
            or record.get("request") != request
            or "response" not in record
        ):
            raise ValueError("invalid tool cache record")
        return record["response"]

    @staticmethod
    def _write(
        path: Path,
        *,
        namespace: str,
        request: Dict[str, Any],
        response: Any,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "version": JsonToolCache.VERSION,
            "namespace": namespace,
            "request": request,
            "response": response,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        temporary = path.with_name(
            ".{}.{}.{}.tmp".format(
                path.name,
                os.getpid(),
                threading.get_ident(),
            )
        )
        try:
            temporary.write_text(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(str(temporary), str(path))
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def get_or_fetch(
        self,
        namespace: str,
        request: Dict[str, Any],
        fetch: Callable[[], Any],
        validate: Optional[Callable[[Any], None]] = None,
    ) -> Tuple[Any, bool]:
        """Return a cached response or fetch and atomically persist it."""

        path = self.path_for(namespace, request)
        with _lock_for(path), _process_lock(path):
            if path.is_file():
                try:
                    response = self._read(
                        path,
                        namespace=namespace,
                        request=request,
                    )
                    if validate is not None:
                        validate(response)
                    return response, True
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
            response = fetch()
            if validate is not None:
                validate(response)
            self._write(
                path,
                namespace=namespace,
                request=request,
                response=response,
            )
            return response, False
