"""Server-side relay for Vercel AI Gateway requests.

The Gateway key remains in the API process and is never made available to the
browser bundle.  This route intentionally supports only the language-model
endpoint used by the chat UI.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from collections.abc import Iterator
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from fastapi import APIRouter, HTTPException, Request as FastAPIRequest
from fastapi.responses import StreamingResponse

load_dotenv()

router = APIRouter(prefix="/gateway", tags=["AI gateway"])
_GATEWAY_LANGUAGE_MODEL_URL = "https://ai-gateway.vercel.sh/v4/ai/language-model"
_FORWARDED_HEADERS = {"accept", "content-type"}


@router.post("/language-model", summary="Relay an AI SDK language-model request")
async def language_model(request: FastAPIRequest) -> StreamingResponse:
    """Forward the AI SDK protocol to Gateway with server-only credentials."""
    api_key = os.getenv("AI_GATEWAY_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="AI_GATEWAY_API_KEY is not configured on the API server.",
        )

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
        _GATEWAY_LANGUAGE_MODEL_URL,
        data=body,
        headers=headers,
        method="POST",
    )

    try:
        upstream = urlopen(upstream_request, timeout=120)
    except HTTPError as exc:
        return StreamingResponse(
            _read_response(exc),
            status_code=exc.code,
            media_type=exc.headers.get_content_type(),
        )
    except OSError as exc:
        raise HTTPException(status_code=502, detail="AI Gateway is unreachable.") from exc

    return StreamingResponse(
        _stream_response(upstream),
        status_code=upstream.status,
        media_type=upstream.headers.get_content_type(),
    )


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
