"""Append-only, hash-chained ledger of guardrail decisions.

Two layers of protection:
  1. Append-only: SQLite triggers abort any UPDATE or DELETE on the table, and
     this class exposes no method that modifies or removes an entry.
  2. Tamper-evident: each entry stores prev_hash and
     entry_hash = sha256(prev_hash + canonical JSON of the entry). Editing any
     past entry (e.g. after dropping the triggers with a raw connection) breaks
     the chain, and verify_chain() reports it.

Only the guardrail layer constructs a Ledger. The CRM tool and the agent's tools
receive no reference to it, so they have no handle to write with. (Within one
Python process this separation is by construction, not by the OS; moving the
guardrail into its own process is a later step.)
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

GENESIS_HASH = "0" * 64
EVENT_TYPES = frozenset({"approval", "recheck", "allow", "block"})


class Ledger:
    def __init__(self, path: str):
        self._conn = sqlite3.connect(path)
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS ledger (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                run_id TEXT NOT NULL,
                event TEXT NOT NULL,
                payload TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                entry_hash TEXT NOT NULL
            );
            CREATE TRIGGER IF NOT EXISTS ledger_no_update
                BEFORE UPDATE ON ledger
                BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
            CREATE TRIGGER IF NOT EXISTS ledger_no_delete
                BEFORE DELETE ON ledger
                BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def append(self, run_id: str, event: str, payload: Dict[str, Any]) -> str:
        """Append one entry and return its hash."""
        if event not in EVENT_TYPES:
            raise ValueError(f"unknown ledger event: {event!r}")
        ts = datetime.now(timezone.utc).isoformat()
        payload_json = _canonical(payload)
        prev_hash = self._last_hash()
        entry_hash = _hash_entry(prev_hash, ts, run_id, event, payload_json)
        self._conn.execute(
            "INSERT INTO ledger (ts, run_id, event, payload, prev_hash, entry_hash)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (ts, run_id, event, payload_json, prev_hash, entry_hash),
        )
        self._conn.commit()
        return entry_hash

    def entries(self, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = "SELECT seq, ts, run_id, event, payload, prev_hash, entry_hash FROM ledger"
        args: tuple = ()
        if run_id is not None:
            sql += " WHERE run_id = ?"
            args = (run_id,)
        rows = self._conn.execute(sql + " ORDER BY seq", args).fetchall()
        return [
            {
                "seq": seq, "ts": ts, "run_id": rid, "event": event,
                "payload": json.loads(payload),
                "prev_hash": prev_hash, "entry_hash": entry_hash,
            }
            for seq, ts, rid, event, payload, prev_hash, entry_hash in rows
        ]

    def verify_chain(self) -> bool:
        """Recompute every hash from the stored content; False on any mismatch."""
        expected_prev = GENESIS_HASH
        rows = self._conn.execute(
            "SELECT ts, run_id, event, payload, prev_hash, entry_hash"
            " FROM ledger ORDER BY seq"
        ).fetchall()
        for ts, run_id, event, payload, prev_hash, entry_hash in rows:
            if prev_hash != expected_prev:
                return False
            if _hash_entry(prev_hash, ts, run_id, event, payload) != entry_hash:
                return False
            expected_prev = entry_hash
        return True

    def _last_hash(self) -> str:
        row = self._conn.execute(
            "SELECT entry_hash FROM ledger ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else GENESIS_HASH


def _canonical(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _hash_entry(prev_hash: str, ts: str, run_id: str, event: str, payload_json: str) -> str:
    body = _canonical({"ts": ts, "run_id": run_id, "event": event, "payload": payload_json})
    return hashlib.sha256((prev_hash + body).encode("utf-8")).hexdigest()
