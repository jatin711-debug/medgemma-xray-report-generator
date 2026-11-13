"""Identify the running application build for report provenance."""
import os
import re
from functools import lru_cache
from pathlib import Path

_REPOSITORY = Path(__file__).resolve().parents[2]
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _git_commit(repository: Path) -> str | None:
    """Read the checked-out commit from .git without running git."""
    git_dir = repository / ".git"
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head if _COMMIT.match(head) else None
        ref = head[5:]
        ref_file = git_dir / ref
        if ref_file.is_file():
            commit = ref_file.read_text(encoding="utf-8").strip()
            return commit if _COMMIT.match(commit) else None
        for line in (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref and _COMMIT.match(parts[0]):
                return parts[0]
    except OSError:
        return None
    return None


@lru_cache(maxsize=1)
def app_version() -> str:
    """Return APP_VERSION if set, else the short git commit, else 'unknown'."""
    configured = os.getenv("APP_VERSION", "").strip()
    if configured and configured != "unknown":
        return configured[:64]
    commit = _git_commit(_REPOSITORY)
    return commit[:12] if commit else "unknown"
