# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import fcntl
import os
import time
from multiprocessing import shared_memory as shm_pkg
from typing import Any

from vllm_omni.entrypoints.stage_utils import shm_read_bytes, shm_write_bytes
from vllm_omni.utils.nvtx import nvtx_mark, nvtx_range

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)


class SharedMemoryConnector(OmniConnectorBase):
    """Key-addressed local shared-memory connector.

    SHM is a local-only transport: it reads/writes POSIX shared memory
    segments identified purely by *key*.  It does **not** understand
    remote-transport metadata such as ``source_host`` / ``source_port``
    (that is the RDMA connector's job).  When such metadata is passed in,
    the connector silently falls back to key-based lookup.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.stage_id = config.get("stage_id", -1)
        self.device = config.get("device", "cuda:0")
        self.threshold = int(config.get("shm_threshold_bytes", 65536))
        self._cleanup_retention_seconds = float(config.get("shm_cleanup_retention_seconds", 30.0))
        self._pending_keys: set[str] = set()
        self._retired_keys: dict[str, float] = {}
        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "shm_writes": 0,
            "inline_writes": 0,
            "large_puts": 0,
            "retired_keys": 0,
            "purged_keys": 0,
            "last_put_ms": 0.0,
            "last_serialize_ms": 0.0,
            "last_write_ms": 0.0,
            "last_get_ms": 0.0,
            "last_read_ms": 0.0,
            "last_deserialize_ms": 0.0,
        }

    def put(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: Any,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        with nvtx_range("SHM.put", color="purple"):
            try:
                # Always serialize first to check size (and for SHM writing)
                # Note: For extremely large objects in "inline" mode (e.g. Ray),
                # we might double-serialize if we're not careful, but here we assume
                # if it's huge we use SHM, or if Ray, threshold is maxsize.
                self._purge_retired_keys()
                put_start = time.perf_counter()
                serialize_start = time.perf_counter()
                payload = self.serialize_obj(data)
                serialize_ms = (time.perf_counter() - serialize_start) * 1000
                size = len(payload)

                # Currently, we always use SHM.
                if True:
                    # Use Shared Memory
                    lock_file = f"/dev/shm/shm_{put_key}_lockfile.lock"
                    write_start = time.perf_counter()
                    with open(lock_file, "wb+") as lockf:
                        fcntl.flock(lockf, fcntl.LOCK_EX)
                        meta = shm_write_bytes(payload, name=put_key)
                        fcntl.flock(lockf, fcntl.LOCK_UN)
                    write_ms = (time.perf_counter() - write_start) * 1000

                    # meta contains {'name': ..., 'size': ...}
                    metadata = {"shm": meta, "size": size}
                    self._pending_keys.add(put_key)
                    self._retired_keys.pop(put_key, None)
                    self._metrics["shm_writes"] += 1
                else:
                    # Inline - pass bytes directly to avoid double serialization of the object
                    # We already serialized it to check size, so we pass the bytes.
                    # The Queue will pickle these bytes (fast), avoiding re-serializing the complex object.
                    metadata = {"inline_bytes": payload, "size": size}
                    self._metrics["inline_writes"] += 1

                self._metrics["puts"] += 1
                self._metrics["bytes_transferred"] += size
                put_ms = (time.perf_counter() - put_start) * 1000
                self._metrics["last_put_ms"] = put_ms
                self._metrics["last_serialize_ms"] = serialize_ms
                self._metrics["last_write_ms"] = write_ms

                if size >= 1024 * 1024 or put_ms >= 500:
                    self._metrics["large_puts"] += 1
                    logger.info(
                        "omni:connector:shm_put key=%s edge=%s->%s bytes=%d "
                        "serialize_ms=%.2f write_ms=%.2f total_ms=%.2f",
                        put_key,
                        from_stage,
                        to_stage,
                        size,
                        serialize_ms,
                        write_ms,
                        put_ms,
                    )

                return True, size, metadata

            except Exception as e:
                logger.error(f"SharedMemoryConnector put failed for req {put_key}: {e}")
                return False, 0, None

    def _get_data_with_lock(self, lock_file: str, shm_handle: dict, get_key: str | None = None):
        obj = None
        received = False
        try:
            get_start = time.perf_counter()
            read_start = time.perf_counter()
            with open(lock_file, "rb+") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX)
                data_bytes = shm_read_bytes(shm_handle)
                fcntl.flock(lockf, fcntl.LOCK_UN)
            read_ms = (time.perf_counter() - read_start) * 1000
            deserialize_start = time.perf_counter()
            obj = self.deserialize_obj(data_bytes)
            received = True
            deserialize_ms = (time.perf_counter() - deserialize_start) * 1000
            get_ms = (time.perf_counter() - get_start) * 1000
            size = int(shm_handle.get("size", 0))
            self._metrics["gets"] += 1
            self._metrics["last_get_ms"] = get_ms
            self._metrics["last_read_ms"] = read_ms
            self._metrics["last_deserialize_ms"] = deserialize_ms
            if size >= 1024 * 1024 or get_ms >= 500:
                logger.info(
                    "omni:connector:shm_get key=%s bytes=%d "
                    "read_ms=%.2f deserialize_ms=%.2f total_ms=%.2f",
                    get_key or shm_handle.get("name"),
                    size,
                    read_ms,
                    deserialize_ms,
                    get_ms,
                )
            return obj, int(shm_handle.get("size", 0))
        except Exception as e:
            logger.error(f"SharedMemoryConnector shm get failed for req : {e}")
            return None
        finally:
            # If data has been received, delete lock_file.
            if received and os.path.exists(lock_file):
                os.remove(lock_file)

    def _get_by_key(self, get_key: str) -> tuple[Any, int] | None:
        """Read a SHM segment addressed purely by *get_key*."""
        shm = None
        try:
            shm = shm_pkg.SharedMemory(name=get_key)
            if shm is None or shm.size == 0:
                return None
            lock_file = f"/dev/shm/shm_{get_key}_lockfile.lock"
            shm_handle = {"name": get_key, "size": shm.size}
            result = self._get_data_with_lock(lock_file, shm_handle, get_key=get_key)
            if result is not None:
                self._pending_keys.discard(get_key)
            return result
        except FileNotFoundError:
            return None
        except Exception:
            logger.debug("_get_by_key: unexpected error reading SHM segment %s", get_key, exc_info=True)
            return None
        finally:
            if shm:
                shm.close()

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata=None,
    ) -> tuple[Any, int] | None:
        with nvtx_range("SHM.get", color="purple"):
            if metadata is not None:
                if isinstance(metadata, dict) and get_key in metadata:
                    metadata = metadata.get(get_key)

                if not isinstance(metadata, dict):
                    return self._get_by_key(get_key)

                if "inline_bytes" in metadata:
                    try:
                        obj = self.deserialize_obj(metadata["inline_bytes"])
                        self._pending_keys.discard(get_key)
                        return obj, int(metadata.get("size", 0))
                    except Exception as e:
                        logger.error(f"SharedMemoryConnector inline get failed for req {get_key}: {e}")
                        return None

                if "shm" in metadata:
                    shm_handle = metadata["shm"]
                    lock_file = f"/dev/shm/shm_{shm_handle['name']}_lockfile.lock"
                    result = self._get_data_with_lock(lock_file, shm_handle, get_key=get_key)
                    if result is not None:
                        self._pending_keys.discard(get_key)
                    return result

                # Metadata is a dict but has no SHM-specific handle (e.g. RDMA-
                # style source_host/source_port).  Fall back to key-based read.
                return self._get_by_key(get_key)

            return self._get_by_key(get_key)

    def cleanup(self, request_id: str) -> None:
        """Best-effort cleanup of unconsumed SHM segments for *request_id*.

        Matches pending keys where *request_id* appears as the full key,
        as a ``_``-delimited prefix, or as a ``_``-delimited suffix.
        If ``get()`` was never called, we unlink it here so /dev/shm
        doesn't leak.
        Sender-side cleanup can race with the downstream stage reading the
        final sentinel.  Retire keys first and unlink after a short TTL; if the
        receiver reads the key in that window, ``shm_read_bytes`` unlinks it.
        """
        self._purge_retired_keys()
        stale = [
            k
            for k in self._pending_keys
            if k == request_id or k.startswith(request_id + "_") or k.endswith("_" + request_id)
        ]
        if self._cleanup_retention_seconds > 0:
            expires_at = time.monotonic() + self._cleanup_retention_seconds
            for key in stale:
                self._pending_keys.discard(key)
                self._retired_keys[key] = expires_at
            self._metrics["retired_keys"] += len(stale)
            return

        for key in stale:
            self._pending_keys.discard(key)
            self._unlink_key(key)

    def close(self) -> None:
        """Unlink all remaining tracked SHM segments."""
        for key in list(self._pending_keys) + list(self._retired_keys):
            self._unlink_key(key)
        self._pending_keys.clear()
        self._retired_keys.clear()

    def _purge_retired_keys(self) -> None:
        if not self._retired_keys:
            return
        now = time.monotonic()
        expired = [key for key, expires_at in self._retired_keys.items() if expires_at <= now]
        for key in expired:
            self._retired_keys.pop(key, None)
            self._unlink_key(key)
            self._metrics["purged_keys"] += 1

    @staticmethod
    def _unlink_key(key: str) -> None:
        try:
            seg = shm_pkg.SharedMemory(name=key)
            seg.close()
            seg.unlink()
            logger.debug("cleanup: unlinked unconsumed SHM segment %s", key)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.debug("cleanup: failed to unlink SHM segment %s: %s", key, e)
        lock_file = f"/dev/shm/shm_{key}_lockfile.lock"
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except OSError:
                pass
    def health(self) -> dict[str, Any]:
        return {"status": "healthy", "threshold": self.threshold, **self._metrics}
