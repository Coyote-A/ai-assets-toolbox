"""
Network volume file operations.

Utility functions for reading, writing, and managing files on the RunPod
Network Volume mounted at ``/runpod-volume/``.
"""

import base64
import logging
import os
import shutil
from typing import Optional

logger = logging.getLogger(__name__)

VOLUME_ROOT = "/runpod-volume"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def volume_path(*parts: str) -> str:
    """
    Build an absolute path under the network volume root.

    Example
    -------
    >>> volume_path("models", "loras", "my-lora.safetensors")
    '/runpod-volume/models/loras/my-lora.safetensors'
    """
    return os.path.join(VOLUME_ROOT, *parts)


def ensure_dir(path: str) -> str:
    """Create ``path`` (and all parents) if it does not exist, then return it."""
    os.makedirs(path, exist_ok=True)
    return path


def safe_volume_path(rel_path: str) -> str:
    """
    Resolve a relative path under the volume root and guard against traversal.

    Raises
    ------
    ValueError
        If the resolved path escapes the volume root.
    """
    abs_path = os.path.normpath(os.path.join(VOLUME_ROOT, rel_path))
    if not abs_path.startswith(VOLUME_ROOT):
        raise ValueError(
            f"Path '{rel_path}' resolves outside the volume root '{VOLUME_ROOT}'"
        )
    return abs_path


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def write_bytes(path: str, data: bytes, append: bool = False) -> int:
    """
    Write raw bytes to a file on the volume.

    Parameters
    ----------
    path:
        Absolute destination path.
    data:
        Bytes to write.
    append:
        If ``True``, append to an existing file instead of overwriting.

    Returns
    -------
    int
        Number of bytes written.
    """
    ensure_dir(os.path.dirname(path))
    mode = "ab" if append else "wb"
    with open(path, mode) as fh:
        fh.write(data)
    logger.debug("Wrote %d bytes to '%s' (append=%s)", len(data), path, append)
    return len(data)


def read_bytes(path: str) -> bytes:
    """Read and return the raw bytes of a file."""
    with open(path, "rb") as fh:
        data = fh.read()
    logger.debug("Read %d bytes from '%s'", len(data), path)
    return data


def write_b64(path: str, b64_data: str, append: bool = False) -> int:
    """
    Decode a base64 string and write the result to a file.

    Returns the number of decoded bytes written.
    """
    raw = base64.b64decode(b64_data)
    return write_bytes(path, raw, append=append)


def read_b64(path: str) -> str:
    """Read a file and return its contents as a base64 string."""
    raw = read_bytes(path)
    return base64.b64encode(raw).decode("utf-8")


# ---------------------------------------------------------------------------
# Directory listing
# ---------------------------------------------------------------------------

def list_files(
    directory: str,
    extensions: Optional[tuple[str, ...]] = None,
    recursive: bool = False,
) -> list[dict]:
    """
    List files in a directory on the volume.

    Parameters
    ----------
    directory:
        Absolute path to the directory.
    extensions:
        Optional tuple of allowed file extensions (e.g. ``(".safetensors", ".pt")``).
        If ``None``, all files are returned.
    recursive:
        If ``True``, recurse into sub-directories.

    Returns
    -------
    list[dict]
        Each entry has keys: ``name``, ``path`` (absolute), ``size_bytes``,
        ``size_mb``.
    """
    if not os.path.isdir(directory):
        logger.warning("Directory '%s' does not exist", directory)
        return []

    results: list[dict] = []

    def _scan(dirpath: str) -> None:
        for entry in sorted(os.scandir(dirpath), key=lambda e: e.name):
            if entry.is_file():
                if extensions is None or entry.name.lower().endswith(extensions):
                    size_bytes = entry.stat().st_size
                    results.append(
                        {
                            "name": entry.name,
                            "path": entry.path,
                            "size_bytes": size_bytes,
                            "size_mb": round(size_bytes / (1024 * 1024), 2),
                        }
                    )
            elif entry.is_dir() and recursive:
                _scan(entry.path)

    _scan(directory)
    logger.debug("Listed %d file(s) in '%s'", len(results), directory)
    return results


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

def delete_path(path: str) -> None:
    """
    Delete a file or directory tree from the volume.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If ``path`` escapes the volume root.
    """
    abs_path = os.path.normpath(path)
    if not abs_path.startswith(VOLUME_ROOT):
        raise ValueError(f"Refusing to delete path outside volume root: '{path}'")

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Path not found: '{abs_path}'")

    if os.path.isfile(abs_path):
        os.remove(abs_path)
        logger.info("Deleted file '%s'", abs_path)
    else:
        shutil.rmtree(abs_path)
        logger.info("Deleted directory tree '%s'", abs_path)


# ---------------------------------------------------------------------------
# Disk usage
# ---------------------------------------------------------------------------

def disk_usage_gb(path: str = VOLUME_ROOT) -> dict[str, float]:
    """
    Return disk usage statistics for the given path.

    Returns
    -------
    dict
        Keys: ``total_gb``, ``used_gb``, ``free_gb``.
    """
    if not os.path.exists(path):
        return {"total_gb": 0.0, "used_gb": 0.0, "free_gb": 0.0}

    stat = shutil.disk_usage(path)
    return {
        "total_gb": round(stat.total / (1024 ** 3), 2),
        "used_gb": round(stat.used / (1024 ** 3), 2),
        "free_gb": round(stat.free / (1024 ** 3), 2),
    }
