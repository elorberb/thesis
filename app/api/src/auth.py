from __future__ import annotations

from fastapi import Header, HTTPException, status
from supabase_auth.errors import AuthApiError
from supabase import Client, create_client

from config import settings

_auth_client: Client = create_client(settings.supabase_url, settings.supabase_anon_key)


def get_current_user(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header",
        )
    token = authorization.removeprefix("Bearer ")
    try:
        response = _auth_client.auth.get_user(token)
    except AuthApiError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc
    if not response.user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
    return response.user.id
