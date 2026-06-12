"""Shared utility functions."""

from pathlib import Path


def ensure_directories(paths):
    """Create directories required by the workflow."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def require_file(path, message=None):
    """Raise a helpful error when an expected local file is missing."""
    path = Path(path)
    if not path.exists():
        default_message = f"Required file not found: {path}"
        raise FileNotFoundError(message or default_message)
    return path
