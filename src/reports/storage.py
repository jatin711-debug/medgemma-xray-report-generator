"""Publish complete local reports and list only regular report files."""
import os
import re
import stat
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

REPORT_PREFIX = "premium_medical_report_"


def report_path(directory: str, analysis_id: str) -> Path:
    """Build a filename from a bounded identifier, rejecting path separators."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", analysis_id):
        raise ValueError("Invalid report analysis identifier")
    return Path(directory) / f"{REPORT_PREFIX}{analysis_id}.pdf"


@contextmanager
def atomic_report(destination: Path) -> Iterator[str]:
    """Publish by rename after a successful build; remove failed temporary files.

    The temporary file shares the destination filesystem so readers see either
    the previous complete report or the new one. This does not guarantee
    durability across a power loss. Rebuilding the same identifier replaces it.
    """
    descriptor, temporary = tempfile.mkstemp(
        prefix=".report-", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        yield temporary
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def list_local_reports(directory: str) -> list[dict]:
    """Return regular reports in filename order, ignoring links and in-progress files.

    Timestamps use file birth time where available, otherwise modification time,
    and always include UTC. Files removed during enumeration are skipped.
    """
    reports = []
    try:
        entries = os.scandir(directory)
    except FileNotFoundError:
        return reports
    with entries:
        for entry in entries:
            if not (entry.name.startswith(REPORT_PREFIX) and entry.name.endswith(".pdf")):
                continue
            try:
                info = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(info.st_mode):
                continue
            reports.append({
                "filename": entry.name,
                "analysis_id": entry.name.removeprefix(REPORT_PREFIX).removesuffix(".pdf"),
                "created_at": datetime.fromtimestamp(
                    getattr(info, "st_birthtime", info.st_mtime), timezone.utc
                ).isoformat(),
                "size": info.st_size,
            })
    return sorted(reports, key=lambda report: report["filename"])
