"""Shared API-key roles for a single trusted research deployment."""
import hmac
import os
from enum import IntEnum
from pathlib import Path

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader


class Role(IntEnum):
    READER = 1
    OPERATOR = 2
    ADMIN = 3


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def configured_keys() -> dict[Role, bytes]:
    """Load distinct role keys from the environment or mounted secret files."""
    keys = {}
    for role in Role:
        name = f"API_KEY_{role.name}"
        raw, path = os.getenv(name, ""), os.getenv(f"{name}_FILE", "")
        if raw and path:
            raise ValueError(f"Configure either {name} or {name}_FILE, not both")
        if path:
            try:
                raw = Path(path).read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError) as exc:
                raise ValueError(f"Unable to read {name}_FILE") from exc
        if raw:
            if len(raw) < 32 or not raw.isascii() or any(char.isspace() for char in raw):
                raise ValueError(f"{name} must be at least 32 non-whitespace ASCII characters")
            keys[role] = raw.encode("ascii")
    if Role.ADMIN not in keys:
        raise ValueError("API_KEY_ADMIN or API_KEY_ADMIN_FILE is required")
    if len(set(keys.values())) != len(keys):
        raise ValueError("API keys must differ between roles")
    return keys


def require_role(minimum: Role):
    """Return a FastAPI dependency requiring at least the given shared role."""
    async def authorize(key: str | None = Security(api_key_header)) -> Role:
        try:
            keys = configured_keys()
        except ValueError as exc:
            raise HTTPException(status_code=503, detail="Authentication is not configured") from exc
        supplied = (key or "").encode("utf-8")
        role = None
        for candidate_role, candidate in keys.items():
            if hmac.compare_digest(supplied, candidate):
                role = candidate_role
        if role is None:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        if role < minimum:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return role
    return authorize


reader_access = Depends(require_role(Role.READER))
operator_access = Depends(require_role(Role.OPERATOR))
admin_access = Depends(require_role(Role.ADMIN))
