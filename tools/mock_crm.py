"""SQLite-backed mock CRM. The only "world" the agent can act on.

This module knows nothing about the guardrail or the ledger. It exposes:
  - read_record:          what the agent (and the guardrail's recheck) sees
  - conditional_update:   compare-and-set write, the only way a change lands
  - mutate_out_of_band:   simulates a separate process changing the world
"""

from __future__ import annotations

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

    def conditional_update(
        self,
        record_id: str,
        expected: Dict[str, Any],
        changes: Dict[str, Any],
        bounds: Sequence[Tuple[str, str, Any]] = (),
    ) -> UpdateResult:
        """Write `changes` only if every field in `expected` still holds that value
        and every (field, op, limit) in `bounds` still holds, op being "<=" or ">=".

        The check and the write are a single UPDATE statement, so nothing can
        land between them. This closes the gap between the guardrail's recheck
        and the actual effect (TOCTOU).
        """
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

        set_sql = ", ".join(f"{col} = ?" for col in changes)
        # `IS` rather than `=` so a NULL in the snapshot compares equal to NULL.
        # A NULL under a bound compares as NULL, so the write is refused.
        where_sql = " AND ".join(
            [f"{col} IS ?" for col in expected] + [f"{col} {op} ?" for col, op, _ in bounds]
        )
        cur = self._conn.execute(
            f"UPDATE customers SET {set_sql} WHERE id = ? AND {where_sql}",
            (*changes.values(), record_id, *expected.values(), *(limit for _, _, limit in bounds)),
        )
        self._conn.commit()
        if cur.rowcount == 1:
            return UpdateResult(written=True, reason="ok")
        if self.read_record(record_id) is None:
            return UpdateResult(written=False, reason="no such record")
        return UpdateResult(written=False, reason="stale: row no longer matches snapshot")

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
