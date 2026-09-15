from __future__ import annotations

import logging
import os
from urllib.parse import urlsplit

import httpx
from starlette.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("ombre_brain.gateway")

_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

_REQUEST_DROP = _HOP_BY_HOP | {
    "host",
    "content-length",
}

_RESPONSE_DROP = _HOP_BY_HOP | {
    "content-length",
}


def _upstream_base() -> str:
    base = (os.environ.get("OMBRE_GATEWAY_UPSTREAM") or "").strip().rstrip("/")
    if not base:
        raise RuntimeError("OMBRE_GATEWAY_UPSTREAM is not configured")

    parsed = urlsplit(base)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise RuntimeError("OMBRE_GATEWAY_UPSTREAM must be an absolute http/https URL")

    return base


def _forward_request_headers(request) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in request.headers.items():
        if key.lower() in _REQUEST_DROP:
            continue
        headers[key] = value
    return headers


def _forward_response_headers(response: httpx.Response) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in response.headers.items():
        if key.lower() in _RESPONSE_DROP:
            continue
        headers[key] = value
    return headers


def register(mcp) -> None:

    @mcp.custom_route(
        "/gateway/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def gateway_proxy(request):
        try:
            upstream = _upstream_base()
        except RuntimeError as exc:
            logger.error("Gateway configuration error: %s", exc)
            return JSONResponse(
                {"error": {"type": "gateway_config_error", "message": str(exc)}},
                status_code=503,
            )

        path = str(request.path_params.get("path") or "").lstrip("/")
        if not path:
            return JSONResponse(
                {"error": {"type": "gateway_path_error", "message": "Missing upstream path"}},
                status_code=404,
            )

        url = f"{upstream}/{path}"
        if request.url.query:
            url += "?" + request.url.query

        body = await request.body()
        headers = _forward_request_headers(request)

        logger.info(
            "[gateway] %s /%s body_bytes=%d",
            request.method,
            path,
            len(body),
        )

        client = httpx.AsyncClient(
            timeout=None,
            follow_redirects=False,
            trust_env=False,
        )

        try:
            upstream_request = client.build_request(
                method=request.method,
                url=url,
                headers=headers,
                content=body,
            )
            upstream_response = await client.send(upstream_request, stream=True)
        except Exception as exc:
            await client.aclose()
            logger.warning(
                "[gateway] upstream request failed: %s",
                type(exc).__name__,
            )
            return JSONResponse(
                {
                    "error": {
                        "type": "gateway_upstream_error",
                        "message": "Upstream request failed",
                    }
                },
                status_code=502,
            )

        response_headers = _forward_response_headers(upstream_response)

        async def relay_body():
            try:
                async for chunk in upstream_response.aiter_raw():
                    yield chunk
            finally:
                await upstream_response.aclose()
                await client.aclose()

        return StreamingResponse(
            relay_body(),
            status_code=upstream_response.status_code,
            headers=response_headers,
            media_type=None,
        )
