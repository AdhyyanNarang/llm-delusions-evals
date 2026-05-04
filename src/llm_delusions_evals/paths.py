"""Shared path defaults and resolution for eval data directories."""

import os
from pathlib import Path

# Resolve the evals repository root
# __file__ is src/llm_delusions_evals/paths.py
_EVALS_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# In a later iteration, subset data should live inside the
# llm-delusions-evals repository directly or be bundled as package data
# within llm-delusions, instead of assuming sibling directory structure.
# For now, we assume llm-delusions is cloned as a sibling directory.
_DATA_REPO_ROOT = _EVALS_REPO_ROOT.parent / "llm-delusions"

DEFAULT_WINDOWS_PATH = str(_DATA_REPO_ROOT / "subsets" / "items_sanitized.parquet")
DEFAULT_TRANSCRIPTS_PATH = str(
    _DATA_REPO_ROOT / "transcripts_data" / "transcripts.parquet"
)


def _abspath_from_repo_root(path_text: str) -> str:
    """Resolve relative paths from the eval repo root.

    Parameters
    ----------
    path_text:
        Candidate filesystem path from task args or environment.

    Returns
    -------
    str
        Absolute filesystem path.
    """
    candidate = Path(path_text).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((_EVALS_REPO_ROOT / candidate).resolve())


def resolve_path(env_var: str, default_path: str, explicit: str = "") -> str:
    """Resolve a filesystem path from an explicit value, env var, or default.

    Parameters
    ----------
    env_var:
        Environment variable name to check when ``explicit`` is empty.
    default_path:
        Default path to fall back on.
    explicit:
        If non-empty, used first.

    Returns
    -------
    str
        Absolute filesystem path.
    """
    if explicit:
        return _abspath_from_repo_root(explicit)
    from_env = os.environ.get(env_var, "")
    if from_env:
        return _abspath_from_repo_root(from_env)
    return _abspath_from_repo_root(default_path)
