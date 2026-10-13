"""Content identities and atomic evidence writes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from xpolicy.core.contracts import ArtifactIdentity
from xpolicy.core.errors import UnsupportedCapability


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_sha256(path: Path) -> str:
    """SHA256 of a file, or a versioned manifest of a materialized directory.

    Directory identity ignores empty directories and rejects symlinks. Each
    sorted POSIX relative filename is UTF-8 length-prefixed, followed by its
    raw 32-byte file digest. See docs/core-development.md for the algorithm.
    """
    if path.is_symlink():
        raise ValueError(f"materialize artifact symlinks before hashing: {path}")
    if not path.is_dir():
        return file_sha256(path)
    digest = hashlib.sha256(b"XPL-DIR-SHA256-v1\n")
    for item in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        if item.is_symlink():
            raise ValueError(f"materialize artifact symlinks before hashing: {item}")
        if item.is_dir():
            continue
        if not item.is_file():
            raise ValueError(f"artifact contains a non-regular file: {item}")
        name = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(bytes.fromhex(file_sha256(item)))
    return digest.hexdigest()


def verify_artifact(identity: ArtifactIdentity, declaring_file: Path) -> Path:
    parsed = urlparse(identity.uri)
    if parsed.scheme:
        raise UnsupportedCapability(f"artifact must be materialized locally before verification: {identity.uri}")
    path = Path(identity.uri).expanduser()
    if not path.is_absolute():
        path = declaring_file.parent / path
    actual = artifact_sha256(path)
    path = path.resolve()
    if actual != identity.sha256:
        raise ValueError(f"artifact SHA256 mismatch for {path}: expected {identity.sha256}, got {actual}")
    return path


def source_revision() -> dict:
    import XPolicyLab

    root = Path(XPolicyLab.__path__[0])
    # A wheel installed below a checkout is not that checkout's tracked
    # source. Do not attribute its bytes to the surrounding git worktree.
    if not (root / ".git").exists():
        return {"git_sha": None, "worktree_dirty": None}
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True, timeout=5).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True, timeout=5).stdout
        return {"git_sha": sha, "worktree_dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"git_sha": None, "worktree_dirty": None}


def write_json(path: Path, value: dict) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
