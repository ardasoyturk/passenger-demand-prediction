"""Server-side relay for OpenAI-compatible language-model requests.

The upstream URL is supplied by the browser-safe chat configuration, while the
credential is read only by this API process. The relay preserves the OpenAI
request and streaming response formats expected by the AI SDK.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Request as FastAPIRequest
from fastapi.responses import StreamingResponse

load_dotenv()

router = APIRouter(prefix="/openai-compatible", tags=["OpenAI-compatible AI"])
_BASE_URL_HEADER = "x-openai-compatible-base-url"
_FORWARDED_HEADERS = {"accept", "content-type"}


@router.post("/{path:path}", summary="Relay an OpenAI-compatible request")
async def openai_compatible_request(
    path: str,
    request: FastAPIRequest,
) -> StreamingResponse:
    """Forward an AI SDK OpenAI-compatible request with a server-only key."""
    api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "OPENAI_COMPATIBLE_API_KEY is not configured on the API server."
            ),
        )

    try:
        upstream_url = _upstream_url(
            request.headers.get(_BASE_URL_HEADER),
            path,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    body = await request.body()
    headers = {
        "Authorization": f"Bearer {api_key}",
        **{
            name: value
            for name, value in request.headers.items()
            if name.lower() in _FORWARDED_HEADERS or name.lower().startswith("ai-")
        },
    }
    upstream_request = Request(
        upstream_url,
        data=body,
        headers=headers,
        method="POST",
    )

    try:
        upstream = await asyncio.to_thread(urlopen, upstream_request, timeout=120)
    except HTTPError as exc:
        return StreamingResponse(
            _read_response(exc),
            status_code=exc.code,
            media_type=exc.headers.get_content_type(),
        )
    except OSError as exc:
        raise HTTPException(
            status_code=502,
            detail="OpenAI-compatible API is unreachable.",
        ) from exc

    return StreamingResponse(
        _stream_response(upstream),
        status_code=upstream.status,
        media_type=upstream.headers.get_content_type(),
    )


def _upstream_url(base_url: str | None, path: str) -> str:
    if not base_url:
        raise ValueError(f"{_BASE_URL_HEADER} header is required.")

    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("The OpenAI-compatible base URL must be an HTTP(S) URL.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "The OpenAI-compatible base URL must not contain credentials or query parameters."
        )

    clean_path = path.strip("/")
    if not clean_path or ".." in clean_path.split("/"):
        raise ValueError("The OpenAI-compatible request path is invalid.")

    return f"{base_url.rstrip('/')}/{clean_path}"


def _stream_response(response: object) -> Iterator[bytes]:
    try:
        while chunk := response.read(8192):  # type: ignore[attr-defined]
            yield chunk
    finally:
        response.close()  # type: ignore[attr-defined]


def _read_response(response: HTTPError) -> Iterator[bytes]:
    try:
        yield response.read()
    finally:
        response.close()
