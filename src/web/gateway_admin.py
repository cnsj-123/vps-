from __future__ import annotations

import os

from starlette.requests import Request
from starlette.responses import (
    JSONResponse,
    Response,
)

from ombrebrain.gateway.gateway_runtime import (
    cache_usage_summary,
    clear_upstream_override,
    resolve_upstream_base,
    set_upstream_override,
)

from . import _shared as sh


def _truthy(value) -> bool:
    return str(
        value or ""
    ).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _current_upstream() -> tuple[
    str,
    str,
]:
    return resolve_upstream_base(
        os.environ.get(
            "OMBRE_GATEWAY_UPSTREAM"
        )
    )


def _payload(
    limit: int,
) -> dict:
    upstream, source = (
        _current_upstream()
    )

    usage = cache_usage_summary(
        limit
    )

    return {
        "upstream": upstream,
        "upstream_source": source,
        "cache_stable": _truthy(
            os.environ.get(
                "OMBRE_GATEWAY_CACHE_STABLE"
            )
        ),
        "requested_ttl": "1h",
        **usage,
    }


def register(mcp) -> None:

    @mcp.custom_route(
        "/api/gateway/cache",
        methods=["GET"],
    )
    async def get_gateway_cache(
        request: Request,
    ) -> Response:
        error = sh._require_auth(
            request
        )

        if error:
            return error

        raw_limit = (
            request.query_params.get(
                "limit",
                "20",
            )
        )

        try:
            limit = int(
                raw_limit
            )
        except ValueError:
            return JSONResponse(
                {
                    "error":
                        "limit 必须是整数",
                },
                status_code=400,
            )

        limit = max(
            1,
            min(
                limit,
                100,
            ),
        )

        try:
            payload = _payload(
                limit
            )
        except RuntimeError:
            return JSONResponse(
                {
                    "error":
                        "Gateway 中转站尚未配置",
                },
                status_code=503,
            )

        return JSONResponse(
            payload,
            headers={
                "Cache-Control":
                    "no-store",
            },
        )

    @mcp.custom_route(
        "/api/gateway/upstream",
        methods=["PUT"],
    )
    async def set_gateway_upstream(
        request: Request,
    ) -> Response:
        error = sh._require_auth(
            request
        )

        if error:
            return error

        try:
            body = await sh._read_json_object(
                request
            )
        except (
            ValueError,
            TypeError,
        ):
            return JSONResponse(
                {
                    "error":
                        "无效 JSON",
                },
                status_code=400,
            )

        if set(body) != {
            "url",
        }:
            return JSONResponse(
                {
                    "error":
                        "只接受 url",
                },
                status_code=400,
            )

        try:
            value = (
                set_upstream_override(
                    body.get("url")
                )
            )
        except ValueError as exc:
            return JSONResponse(
                {
                    "error":
                        str(exc),
                },
                status_code=400,
            )

        return JSONResponse(
            {
                "ok": True,
                "upstream": value,
                "upstream_source":
                    "dashboard",
            },
            headers={
                "Cache-Control":
                    "no-store",
            },
        )

    @mcp.custom_route(
        "/api/gateway/upstream",
        methods=["DELETE"],
    )
    async def reset_gateway_upstream(
        request: Request,
    ) -> Response:
        error = sh._require_auth(
            request
        )

        if error:
            return error

        clear_upstream_override()

        try:
            value, source = (
                _current_upstream()
            )
        except RuntimeError:
            return JSONResponse(
                {
                    "ok": True,
                    "upstream": "",
                    "upstream_source":
                        "unconfigured",
                },
                headers={
                    "Cache-Control":
                        "no-store",
                },
            )

        return JSONResponse(
            {
                "ok": True,
                "upstream": value,
                "upstream_source":
                    source,
            },
            headers={
                "Cache-Control":
                    "no-store",
            },
        )
