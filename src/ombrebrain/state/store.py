from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .models import State


class StateStore:
    """Persistent OB2.x State storage.

    Read and write paths are deliberately separated:

    - initialize()/save_state() may create the .state directory/database/schema.
    - get_state() never creates a directory, database, table, or journal.
    - a read before initialization simply returns an empty State.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.db_path = self.root / ".state" / "state.sqlite3"

    def _connect_write(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS state (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                payload_json TEXT NOT NULL,
                state_revision INTEGER NOT NULL CHECK(state_revision>=1),
                changed_at TEXT NOT NULL,
                changed_by TEXT NOT NULL
            )
            """
        )
        return conn

    def _connect_read(self):
        # mode=ro prevents an accidental write from silently entering the
        # State read path in the future.
        uri = self.db_path.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def initialize(self) -> None:
        """Explicitly create the State storage/schema.

        This belongs to application startup, not to the read path.
        """
        with self._connect_write():
            pass

    def get_state(self):
        # Important invariant: reading an uninitialized StateStore must not
        # create .state/, state.sqlite3, or any SQLite side files.
        if not self.db_path.is_file():
            return State.empty(), 0

        with self._connect_read() as conn:
            row = conn.execute(
                "SELECT payload_json,state_revision "
                "FROM state WHERE singleton=1"
            ).fetchone()

            if row is None:
                return State.empty(), 0

            payload = json.loads(row["payload_json"])

            return (
                State(
                    relationship=dict(payload.get("relationship") or {}),
                    current_focus=payload.get("current_focus"),
                    last_seen=payload.get("last_seen"),
                ),
                int(row["state_revision"]),
            )

    def save_state(
        self,
        state: State,
        changed_by="system",
        expected_revision=None,
    ):
        now = datetime.now(timezone.utc).isoformat()

        # Write path is allowed to initialize storage, so StateService remains
        # usable in isolated tests even if initialize() was not called first.
        with self._connect_write() as conn:
            row = conn.execute(
                "SELECT state_revision FROM state WHERE singleton=1"
            ).fetchone()

            current_revision = (
                int(row["state_revision"]) if row else 0
            )

            if (
                expected_revision is not None
                and current_revision != expected_revision
            ):
                raise ValueError(
                    "state revision conflict: "
                    f"expected {expected_revision}, "
                    f"actual {current_revision}"
                )

            new_revision = current_revision + 1

            payload = json.dumps(
                state.to_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
            )

            conn.execute(
                """
                INSERT INTO state (
                    singleton,
                    payload_json,
                    state_revision,
                    changed_at,
                    changed_by
                )
                VALUES (1,?,?,?,?)
                ON CONFLICT(singleton) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    state_revision=excluded.state_revision,
                    changed_at=excluded.changed_at,
                    changed_by=excluded.changed_by
                """,
                (
                    payload,
                    new_revision,
                    now,
                    changed_by,
                ),
            )

            return new_revision
