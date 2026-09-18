from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


_LOCK = threading.RLock()
_MAX_RECORDS = 500
_DEFAULT_STATE_DIR = "/app/buckets/.gateway"


def _state_dir() -> Path:
    raw = (
        os.environ.get("OMBRE_GATEWAY_STATE_DIR")
        or _DEFAULT_STATE_DIR
    ).strip()

    return Path(raw)


def _upstream_path() -> Path:
    return _state_dir() / "upstream.json"


def _usage_path() -> Path:
    return _state_dir() / "cache_usage.jsonl"


def _ensure_state_dir() -> Path:
    path = _state_dir()
    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        path.chmod(0o700)
    except OSError:
        pass

    return path


def _atomic_write_text(
    path: Path,
    text: str,
) -> None:
    _ensure_state_dir()

    temp = path.with_name(
        path.name + ".tmp"
    )

    temp.write_text(
        text,
        encoding="utf-8",
    )

    try:
        temp.chmod(0o600)
    except OSError:
        pass

    os.replace(
        str(temp),
        str(path),
    )


def validate_upstream_url(
    raw: Any,
) -> str:
    if not isinstance(raw, str):
        raise ValueError(
            "中转站 URL 必须是字符串"
        )

    value = raw.strip()

    if not value:
        raise ValueError(
            "中转站 URL 不能为空"
        )

    if len(value) > 2048:
        raise ValueError(
            "中转站 URL 过长"
        )

    parsed = urlsplit(value)

    if parsed.scheme not in (
        "http",
        "https",
    ):
        raise ValueError(
            "只支持 http:// 或 https://"
        )

    if not parsed.netloc:
        raise ValueError(
            "中转站 URL 缺少主机名"
        )

    if not parsed.hostname:
        raise ValueError(
            "中转站 URL 主机名无效"
        )

    if (
        parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            "不要把账号或密钥写进 URL"
        )

    if parsed.query:
        raise ValueError(
            "中转站 URL 不应包含查询参数"
        )

    if parsed.fragment:
        raise ValueError(
            "中转站 URL 不应包含 #fragment"
        )

    return value.rstrip("/")


def _read_override_unlocked() -> str | None:
    path = _upstream_path()

    if not path.exists():
        return None

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )
    except Exception:
        return None

    if not isinstance(
        payload,
        dict,
    ):
        return None

    value = payload.get("url")

    try:
        return validate_upstream_url(
            value
        )
    except ValueError:
        return None


def get_upstream_override() -> str | None:
    with _LOCK:
        return _read_override_unlocked()


def set_upstream_override(
    raw: Any,
) -> str:
    value = validate_upstream_url(
        raw
    )

    with _LOCK:
        _atomic_write_text(
            _upstream_path(),
            json.dumps(
                {
                    "url": value,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n",
        )

    return value


def clear_upstream_override() -> None:
    with _LOCK:
        path = _upstream_path()

        try:
            path.unlink()
        except FileNotFoundError:
            pass


def resolve_upstream_base(
    env_value: Any,
) -> tuple[str, str]:
    """
    Return (url, source).

    Dashboard override has priority over
    OMBRE_GATEWAY_UPSTREAM.
    """

    with _LOCK:
        override = (
            _read_override_unlocked()
        )

    if override:
        return (
            override,
            "dashboard",
        )

    try:
        env_url = (
            validate_upstream_url(
                env_value,
            )
        )
    except ValueError as exc:
        raise RuntimeError(
            "OMBRE_GATEWAY_UPSTREAM "
            "is not configured correctly"
        ) from exc

    return (
        env_url,
        "environment",
    )


def _safe_token(
    summary: dict[str, Any],
    key: str,
) -> int:
    value = summary.get(
        key,
        0,
    )

    if isinstance(value, bool):
        return 0

    if not isinstance(
        value,
        (int, float),
    ):
        return 0

    if value < 0:
        return 0

    return int(value)


def _read_records_unlocked() -> list[dict[str, Any]]:
    path = _usage_path()

    if not path.exists():
        return []

    records: list[
        dict[str, Any]
    ] = []

    try:
        lines = path.read_text(
            encoding="utf-8",
        ).splitlines()
    except Exception:
        return []

    for line in lines:
        if not line.strip():
            continue

        try:
            item = json.loads(
                line
            )
        except Exception:
            continue

        if isinstance(
            item,
            dict,
        ):
            records.append(
                item
            )

    return records[
        -_MAX_RECORDS:
    ]


def _write_records_unlocked(
    records: list[
        dict[str, Any]
    ],
) -> None:
    records = records[
        -_MAX_RECORDS:
    ]

    text = "".join(
        json.dumps(
            item,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
        for item in records
    )

    _atomic_write_text(
        _usage_path(),
        text,
    )


def record_cache_usage(
    summary: dict[str, Any],
    *,
    upstream: str = "",
) -> dict[str, Any] | None:
    """
    Persist privacy-safe per-request cache accounting.

    No prompt, output text, API key or request ID
    is written.
    """

    if not isinstance(
        summary,
        dict,
    ):
        return None

    if not summary.get(
        "observed"
    ):
        return None

    input_tokens = _safe_token(
        summary,
        "input_tokens",
    )

    creation_tokens = _safe_token(
        summary,
        "cache_creation_input_tokens",
    )

    read_tokens = _safe_token(
        summary,
        "cache_read_input_tokens",
    )

    output_tokens = _safe_token(
        summary,
        "output_tokens",
    )

    total_input = (
        input_tokens
        + creation_tokens
        + read_tokens
    )

    if total_input > 0:
        hit_rate = (
            read_tokens
            / total_input
        )

        cache_status = (
            "hit"
            if read_tokens > 0
            else "miss"
        )
    else:
        hit_rate = None
        cache_status = "unknown"

    upstream_host = ""

    try:
        upstream_host = (
            urlsplit(
                upstream
            ).netloc
            if upstream
            else ""
        )
    except Exception:
        upstream_host = ""

    with _LOCK:
        records = (
            _read_records_unlocked()
        )

        previous_round = 0

        if records:
            value = records[-1].get(
                "round",
                0,
            )

            if (
                isinstance(
                    value,
                    int,
                )
                and not isinstance(
                    value,
                    bool,
                )
            ):
                previous_round = value

        record = {
            "round": (
                previous_round + 1
            ),
            "timestamp": (
                datetime.now(
                    timezone.utc
                )
                .isoformat()
                .replace(
                    "+00:00",
                    "Z",
                )
            ),
            "status": cache_status,
            "hit_rate": hit_rate,
            "input_tokens": input_tokens,
            "cache_creation_input_tokens": (
                creation_tokens
            ),
            "cache_read_input_tokens": (
                read_tokens
            ),
            "total_input_tokens": (
                total_input
            ),
            "output_tokens": (
                output_tokens
            ),
            "upstream_host": (
                upstream_host
            ),
        }

        records.append(
            record
        )

        _write_records_unlocked(
            records
        )

    return record


def recent_cache_usage(
    limit: int = 20,
) -> list[dict[str, Any]]:
    try:
        limit = int(limit)
    except Exception:
        limit = 20

    limit = max(
        1,
        min(
            limit,
            100,
        ),
    )

    with _LOCK:
        records = (
            _read_records_unlocked()
        )

    return records[
        -limit:
    ]


def cache_usage_summary(
    limit: int = 20,
) -> dict[str, Any]:
    records = recent_cache_usage(
        limit
    )

    valid_rates = [
        item["hit_rate"]
        for item in records
        if isinstance(
            item.get(
                "hit_rate"
            ),
            (int, float),
        )
        and not isinstance(
            item.get(
                "hit_rate"
            ),
            bool,
        )
    ]

    average_hit_rate = None

    if valid_rates:
        average_hit_rate = (
            sum(valid_rates)
            / len(valid_rates)
        )

    total_read = sum(
        int(
            item.get(
                "cache_read_input_tokens",
                0,
            )
            or 0
        )
        for item in records
    )

    total_input = sum(
        int(
            item.get(
                "total_input_tokens",
                0,
            )
            or 0
        )
        for item in records
    )

    weighted_hit_rate = None

    if total_input > 0:
        weighted_hit_rate = (
            total_read
            / total_input
        )

    return {
        "rounds": records,
        "count": len(records),
        "valid_count": (
            len(valid_rates)
        ),
        "average_hit_rate": (
            average_hit_rate
        ),
        "weighted_hit_rate": (
            weighted_hit_rate
        ),
    }
