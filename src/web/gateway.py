from __future__ import annotations

import hashlib
import json
import logging
import os
from urllib.parse import urlsplit

import httpx
from starlette.responses import JSONResponse, StreamingResponse

from ombrebrain.gateway import (
    canonical_summary_from_body,
    rewrite_body_for_cache,
    rewrite_shadow_summary_from_body,
)

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

_OBSERVE_MAX_DEPTH = 8
_OBSERVE_MAX_ITEMS = 64
_SAFE_ENUM_KEYS = {"role", "type"}


def _truthy(value) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


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


def _text_fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _json_shape(value, *, key: str = "", depth: int = 0):
    """Describe JSON structure without logging scalar user content."""
    if depth >= _OBSERVE_MAX_DEPTH:
        return {"kind": "depth_limit"}

    if isinstance(value, dict):
        fields = {}
        items = list(value.items())
        for raw_key, child in items[:_OBSERVE_MAX_ITEMS]:
            child_key = str(raw_key)
            fields[child_key] = _json_shape(
                child,
                key=child_key,
                depth=depth + 1,
            )
        result = {
            "kind": "object",
            "field_count": len(items),
            "fields": fields,
        }
        if len(items) > _OBSERVE_MAX_ITEMS:
            result["truncated"] = True
        return result

    if isinstance(value, list):
        result = {
            "kind": "array",
            "count": len(value),
            "items": [
                _json_shape(child, key=key, depth=depth + 1)
                for child in value[:_OBSERVE_MAX_ITEMS]
            ],
        }
        if len(value) > _OBSERVE_MAX_ITEMS:
            result["truncated"] = True
        return result

    if isinstance(value, str):
        result = {
            "kind": "string",
            "chars": len(value),
            "sha256": _text_fingerprint(value),
        }
        if (
            key in _SAFE_ENUM_KEYS
            and len(value) <= 64
            and value.isprintable()
        ):
            result["enum"] = value
        return result

    if value is None:
        return {"kind": "null"}
    if isinstance(value, bool):
        return {"kind": "boolean"}
    if isinstance(value, (int, float)):
        return {"kind": "number"}

    return {"kind": type(value).__name__}


def _observe_canonical_request(body: bytes) -> None:
    if not _truthy(
        os.environ.get("OMBRE_GATEWAY_CANONICAL_OBSERVE")
    ):
        return

    summary = canonical_summary_from_body(body)

    if summary is None:
        logger.info(
            "[gateway.canonical] unsupported_or_non_json"
        )
        return

    logger.info(
        "[gateway.canonical] %s",
        json.dumps(
            summary,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


def _observe_rewrite_shadow(body: bytes) -> None:
    if not _truthy(
        os.environ.get("OMBRE_GATEWAY_REWRITE_SHADOW")
    ):
        return

    summary = rewrite_shadow_summary_from_body(body)

    if summary is None:
        logger.info(
            "[gateway.rewrite_shadow] unsupported_or_non_json"
        )
        return

    logger.info(
        "[gateway.rewrite_shadow] %s",
        json.dumps(
            summary,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


def _rewrite_upstream_body(body: bytes) -> bytes:
    if not _truthy(
        os.environ.get("OMBRE_GATEWAY_REWRITE")
    ):
        return body

    try:
        rewritten_body, report = rewrite_body_for_cache(
            body
        )
    except Exception as exc:
        logger.warning(
            "[gateway.rewrite] failed type=%s fail_open=true",
            type(exc).__name__,
        )
        return body

    logger.info(
        "[gateway.rewrite] %s",
        json.dumps(
            report,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )

    return rewritten_body


def _observe_request(body: bytes, content_type: str) -> None:
    if not _truthy(os.environ.get("OMBRE_GATEWAY_OBSERVE")):
        return

    try:
        payload = json.loads(body)
    except Exception:
        logger.info(
            "[gateway.observe] non_json content_type=%s body_bytes=%d sha256=%s",
            str(content_type or "")[:100],
            len(body),
            hashlib.sha256(body).hexdigest()[:12],
        )
        return

    shape = _json_shape(payload)
    logger.info(
        "[gateway.observe] %s",
        json.dumps(shape, ensure_ascii=False, separators=(",", ":")),
    )


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

        if path:
            url = f"{upstream}/{path}"
        else:
            url = upstream
        if request.url.query:
            url += "?" + request.url.query

        body = await request.body()
        headers = _forward_request_headers(request)

        _observe_request(
            body,
            request.headers.get("content-type", ""),
        )

        _observe_canonical_request(body)
        _observe_rewrite_shadow(body)

        forward_body = _rewrite_upstream_body(body)

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
                content=forward_body,
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
