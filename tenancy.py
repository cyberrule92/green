"""
Tenancy resolution — Adaptive Green AI

Header-based (no SSO): every request carries `X-Tenant-Id`. When absent,
the request is attributed to the special tenant `default` so existing
single-tenant deployments keep working unchanged.

Tenant IDs are normalised to `[a-z0-9_-]{1,64}` to keep them safe for
filesystem paths, SQL, and JSON keys.
"""

from __future__ import annotations

import os
import re
from typing import Any

from fastapi import Header, HTTPException

DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "default")
ADMIN_TENANT_ID = os.getenv("ADMIN_TENANT_ID", "admin")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "").strip()

_TENANT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def normalise_tenant_id(raw: str | None) -> str:
    if not raw:
        return DEFAULT_TENANT_ID
    candidate = raw.strip().lower()
    if not candidate:
        return DEFAULT_TENANT_ID
    if not _TENANT_ID_RE.match(candidate):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid X-Tenant-Id. Must match [a-z0-9][a-z0-9_-]{0,63} "
                "(lowercase, 1-64 chars, start with alphanumeric)."
            ),
        )
    return candidate


async def resolve_tenant(
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
) -> str:
    """FastAPI dependency: returns the resolved tenant id (always non-empty)."""
    return normalise_tenant_id(x_tenant_id)


async def require_admin(
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
) -> bool:
    """Soft admin check — only enforced if ADMIN_API_KEY env var is configured.

    When ADMIN_API_KEY is empty, admin endpoints are open (dev mode).
    Production deployments must set ADMIN_API_KEY.
    """
    if not ADMIN_API_KEY:
        return True
    if not x_admin_key or x_admin_key.strip() != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Admin key required.")
    return True


def tenant_metadata(tenant_id: str) -> dict[str, Any]:
    """Return lightweight metadata for audit / response headers."""
    return {
        "tenant_id": tenant_id,
        "is_default": tenant_id == DEFAULT_TENANT_ID,
        "admin_protected": bool(ADMIN_API_KEY),
    }
