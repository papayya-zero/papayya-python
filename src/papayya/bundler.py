"""Project bundler for code deployment.

Walks the project directory, creates a .tar.gz archive excluding
non-essential files, and computes a SHA256 hash.
"""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
from pathlib import Path

# Directories and patterns to always exclude
EXCLUDE_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".env", ".mypy_cache", ".pytest_cache", "dist", "build", "*.egg-info"}
EXCLUDE_EXTENSIONS = {".pyc", ".pyo", ".so", ".dylib"}
EXCLUDE_FILES = {".env", ".env.local", ".env.production"}


def bundle_project(
    project_dir: str,
    entrypoint: str = "agent.py",
    include_dirs: list[str] | None = None,
) -> tuple[bytes, str]:
    """Bundle a project directory into a .tar.gz archive.

    Args:
        project_dir: Path to the project directory.
        entrypoint: Main agent file (must exist).
        include_dirs: Additional directories to include.

    Returns:
        Tuple of (tarball_bytes, sha256_hex_digest).
    """
    project_path = Path(project_dir).resolve()

    if not project_path.is_dir():
        raise FileNotFoundError(f"Project directory not found: {project_path}")

    entrypoint_path = project_path / entrypoint
    if not entrypoint_path.exists():
        raise FileNotFoundError(f"Entrypoint not found: {entrypoint_path}")

    buf = io.BytesIO()

    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for root, dirs, files in os.walk(project_path):
            # Filter out excluded directories
            dirs[:] = [
                d for d in dirs
                if d not in EXCLUDE_DIRS and not any(d.endswith(ext) for ext in {".egg-info"})
            ]

            rel_root = Path(root).relative_to(project_path)

            for fname in files:
                fpath = Path(root) / fname

                # Skip excluded files
                if fname in EXCLUDE_FILES:
                    continue
                if any(fname.endswith(ext) for ext in EXCLUDE_EXTENSIONS):
                    continue

                arcname = str(rel_root / fname)
                tar.add(str(fpath), arcname=arcname)

    tarball_bytes = buf.getvalue()
    sha256 = hashlib.sha256(tarball_bytes).hexdigest()

    return tarball_bytes, sha256
