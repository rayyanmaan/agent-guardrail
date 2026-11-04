"""SQLite-backed mock CRM. The only "world" the agent can act on.

This module knows nothing about the guardrail or the ledger. It exposes:
  - read_record:          what the agent (and the guardrail's recheck) sees
  - conditional_update:   compare-and-set write, the only way a change lands. Each
                          applied change is logged with its idempotency key
                          (UNIQUE) in the same transaction, so a retry of an
                          already-applied change is recognised, not re-applied.
  - mutate_out_of_band:   simulates a separate process changing the world
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Whitelisted columns. Field names arrive from model output, so they must
# never be interpolated into SQL unless they appear here.
COLUMNS = ("id", "name", "email", "status", "plan", "seats", "notes", "last_contacted",
           "discount_pct", "balance_due")
EDITABLE_FIELDS = frozenset(COLUMNS) - {"id"}
BOUND_OPS = ("<=", ">=")


@dataclass(frozen=True)
class UpdateResult:
    written: bool
    reason: str
    duplicate: bool = False                    # key already applied; nothing written
    original: Optional[Dict[str, Any]] = None  # when duplicate, the change logged under the key


class MockCRM:
    def __init__(self, path: str):
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS customers (
                id TEXT PRIMARY KEY,
                name TEXT, email TEXT, status TEXT, plan TEXT,
                seats INTEGER, notes TEXT, last_contacted TEXT,
                discount_pct INTEGER, balance_due INTEGER
            )"""
        )
        # One row per applied change. The key lives with the change, written in
        # the same transaction, so there is no state where the change landed
        # but its key did not.
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS applied_changes (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                idempotency_key TEXT NOT NULL UNIQUE,
                record_id TEXT NOT NULL,
                changes TEXT NOT NULL
            )"""
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def seed(self, record: Dict[str, Any]) -> None:
        _check_fields(record)
        cols = ", ".join(record)
        marks = ", ".join("?" for _ in record)
        self._conn.execute(
            f"INSERT OR REPLACE INTO customers ({cols}) VALUES ({marks})",
            tuple(record.values()),
        )
        self._conn.commit()

    def read_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM customers WHERE id = ?", (record_id,)
        ).fetchone()
        return dict(row) if row else None

    def read_all(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM customers ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def applied_changes(self) -> List[Dict[str, Any]]:
        """Every change that landed, in order (without the keys)."""
        rows = self._conn.execute("SELECT record_id, changes FROM applied_changes ORDER BY seq")
        return [{"record_id": r["record_id"], "changes": json.loads(r["changes"])} for r in rows]

    def conditional_update(
        self,
        record_id: str,
        expected: Dict[str, Any],
        changes: Dict[str, Any],
        bounds: Sequence[Tuple[str, str, Any]] = (),
        *,
        idempotency_key: str,
    ) -> UpdateResult:
        """Write `changes` only if every field in `expected` still holds that value
        and every (field, op, limit) in `bounds` still holds, op being "<=" or ">=".

        The check and the write are a single UPDATE statement, so nothing can
        land between them. This closes the gap between the guardrail's recheck
        and the actual effect (TOCTOU).

        If `idempotency_key` was already applied, nothing is written and the
        original change is returned (duplicate=True). The key is checked before
        the conditions: a retry of an applied change usually no longer matches
        them (the field it overwrote has moved), and must be reported as already
        applied, not as stale. The UNIQUE constraint is the final gate if the
        lookup is ever bypassed.
        """
        if not idempotency_key:
            raise ValueError("conditional_update requires an idempotency key")
        if not expected and not bounds:
            # An unconditional write would bypass the whole guardrail.
            raise ValueError("conditional_update requires a non-empty expected snapshot or bounds")
        _check_fields(expected)
        _check_fields(changes)
        _check_fields({col: None for col, _, _ in bounds})
        if any(op not in BOUND_OPS for _, op, _ in bounds):
            raise ValueError(f"bound operators must be one of {BOUND_OPS}")
        if "id" in changes:
            raise ValueError("id is not editable")

        original = self._applied(idempotency_key)
        if original is not None:
            return UpdateResult(False, "duplicate: already applied", duplicate=True, original=original)

        set_sql = ", ".join(f"{col} = ?" for col in changes)
        # `IS` rather than `=` so a NULL in the snapshot compares equal to NULL.
        # A NULL under a bound compares as NULL, so the write is refused.
        where_sql = " AND ".join(
            [f"{col} IS ?" for col in expected] + [f"{col} {op} ?" for col, op, _ in bounds]
        )
        try:
            cur = self._conn.execute(
                f"UPDATE customers SET {set_sql} WHERE id = ? AND {where_sql}",
                (*changes.values(), record_id, *expected.values(), *(limit for _, _, limit in bounds)),
            )
            if cur.rowcount == 1:
                self._record_applied(idempotency_key, record_id, changes)
            self._conn.commit()
        except sqlite3.IntegrityError:
            # The key was applied after the lookup above. Undo this update.
            self._conn.rollback()
            return UpdateResult(False, "duplicate: already applied", duplicate=True,
                                original=self._applied(idempotency_key))
        except BaseException:
            # Interrupted between the update and its key: neither lands.
            self._conn.rollback()
            raise
        if cur.rowcount == 1:
            return UpdateResult(written=True, reason="ok")
        if self.read_record(record_id) is None:
            return UpdateResult(written=False, reason="no such record")
        return UpdateResult(written=False, reason="stale: row no longer matches snapshot")

    def _applied(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT record_id, changes FROM applied_changes WHERE idempotency_key = ?",
            (idempotency_key,)).fetchone()
        return {"record_id": row["record_id"], "changes": json.loads(row["changes"])} if row else None

    def _record_applied(self, idempotency_key: str, record_id: str, changes: Dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO applied_changes (idempotency_key, record_id, changes) VALUES (?, ?, ?)",
            (idempotency_key, record_id, json.dumps(changes, sort_keys=True)))

    def mutate_out_of_band(self, record_id: str, **changes: Any) -> None:
        """Simulate another process editing the record, via its own connection."""
        _check_fields(changes)
        other = sqlite3.connect(self.path)
        try:
            set_sql = ", ".join(f"{col} = ?" for col in changes)
            other.execute(
                f"UPDATE customers SET {set_sql} WHERE id = ?",
                (*changes.values(), record_id),
            )
            other.commit()
        finally:
            other.close()


def _check_fields(fields: Dict[str, Any]) -> None:
    unknown = set(fields) - set(COLUMNS)
    if unknown:
        raise ValueError(f"unknown CRM fields: {sorted(unknown)}")
