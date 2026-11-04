"""SQLite-backed mock mailer: a contact list and an outbox. Nothing is really sent.

This module knows nothing about the guardrail or the ledger. It exposes:
  - read_contact:        what the agent (and the guardrail's recheck) sees
  - conditional_send:    compare-and-set send, the only way a message lands in the outbox.
                         Each outbox row carries its idempotency key (UNIQUE), so a
                         retry of an already-sent message is recognised, not re-sent.
  - mutate_out_of_band:  simulates a separate process changing a contact
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Whitelisted contact columns. Field names are interpolated into SQL only if listed here.
CONTACT_COLUMNS = ("id", "name", "company", "recipient_email", "opt_in_status", "notes")
BOUND_OPS = ("<=", ">=")


@dataclass(frozen=True)
class SendResult:
    written: bool
    reason: str
    duplicate: bool = False                    # key already sent; nothing written
    original: Optional[Dict[str, Any]] = None  # when duplicate, the outbox row under the key


class MockMailer:
    def __init__(self, path: str):
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS contacts (
                id TEXT PRIMARY KEY,
                name TEXT, company TEXT, recipient_email TEXT, opt_in_status TEXT, notes TEXT
            );
            CREATE TABLE IF NOT EXISTS outbox (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                idempotency_key TEXT NOT NULL UNIQUE,
                contact_id TEXT NOT NULL, to_email TEXT NOT NULL,
                subject TEXT NOT NULL, body TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def seed(self, contact: Dict[str, Any]) -> None:
        _check_fields(contact)
        cols = ", ".join(contact)
        marks = ", ".join("?" for _ in contact)
        self._conn.execute(
            f"INSERT OR REPLACE INTO contacts ({cols}) VALUES ({marks})", tuple(contact.values()))
        self._conn.commit()

    def read_contact(self, contact_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute("SELECT * FROM contacts WHERE id = ?", (contact_id,)).fetchone()
        return dict(row) if row else None

    def read_all_contacts(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in self._conn.execute("SELECT * FROM contacts ORDER BY id")]

    def outbox(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT contact_id, to_email, subject, body FROM outbox ORDER BY seq")
        return [dict(r) for r in rows]

    def conditional_send(
        self,
        contact_id: str,
        expected: Dict[str, Any],
        subject: str,
        body: str,
        bounds: Sequence[Tuple[str, str, Any]] = (),
        *,
        idempotency_key: str,
    ) -> SendResult:
        """Queue a message only if the contact still matches `expected` and `bounds`.

        One INSERT ... SELECT: the address is taken from the contact row in the
        same statement that checks it, so the check and the send cannot be split.

        If `idempotency_key` was already sent, nothing is queued and the original
        outbox row is returned (duplicate=True). The key is checked before the
        conditions, so a retry is reported as already sent even if the contact
        has since changed. The key is a column of the outbox row itself, so the
        send and its key land together, and the UNIQUE constraint is the final
        gate if the lookup is ever bypassed.
        """
        if not idempotency_key:
            raise ValueError("conditional_send requires an idempotency key")
        if not expected and not bounds:
            raise ValueError("conditional_send requires a non-empty expected snapshot or bounds")
        _check_fields(expected)
        _check_fields({col: None for col, _, _ in bounds})
        if any(op not in BOUND_OPS for _, op, _ in bounds):
            raise ValueError(f"bound operators must be one of {BOUND_OPS}")
        original = self._sent(idempotency_key)
        if original is not None:
            return SendResult(False, "duplicate: already sent", duplicate=True, original=original)
        where_sql = " AND ".join(
            [f"{col} IS ?" for col in expected] + [f"{col} {op} ?" for col, op, _ in bounds])
        try:
            cur = self._conn.execute(
                "INSERT INTO outbox (idempotency_key, contact_id, to_email, subject, body)"
                f" SELECT ?, id, recipient_email, ?, ? FROM contacts WHERE id = ? AND {where_sql}",
                (idempotency_key, subject, body, contact_id, *expected.values(),
                 *(limit for _, _, limit in bounds)),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            self._conn.rollback()
            return SendResult(False, "duplicate: already sent", duplicate=True,
                              original=self._sent(idempotency_key))
        if cur.rowcount == 1:
            return SendResult(written=True, reason="queued")
        if self.read_contact(contact_id) is None:
            return SendResult(written=False, reason="no such contact")
        return SendResult(written=False, reason="stale: contact no longer matches snapshot")

    def _sent(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT contact_id, to_email, subject, body FROM outbox WHERE idempotency_key = ?",
            (idempotency_key,)).fetchone()
        return dict(row) if row else None

    def mutate_out_of_band(self, contact_id: str, **changes: Any) -> None:
        """Simulate another process editing the contact, via its own connection."""
        _check_fields(changes)
        other = sqlite3.connect(self.path)
        try:
            set_sql = ", ".join(f"{col} = ?" for col in changes)
            other.execute(f"UPDATE contacts SET {set_sql} WHERE id = ?", (*changes.values(), contact_id))
            other.commit()
        finally:
            other.close()


def _check_fields(fields: Dict[str, Any]) -> None:
    unknown = set(fields) - set(CONTACT_COLUMNS)
    if unknown:
        raise ValueError(f"unknown contact fields: {sorted(unknown)}")
