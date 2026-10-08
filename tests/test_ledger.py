import sqlite3

import pytest

from guardrail.ledger import Ledger, _hash_entry


@pytest.fixture
def ledger_path(tmp_path):
    path = str(tmp_path / "ledger.sqlite")
    ledger = Ledger(path)
    ledger.append("run-1", "approval", {"snapshot": {"status": "active", "plan": "starter"}})
    ledger.append("run-1", "recheck", {"verdict": {"changed": False}})
    ledger.append("run-1", "allow", {"write": "ok"})
    ledger.close()
    return path


def test_untouched_chain_verifies(ledger_path):
    ledger = Ledger(ledger_path)
    assert ledger.verify_chain()
    assert [e["event"] for e in ledger.entries("run-1")] == ["approval", "recheck", "allow"]


def test_entries_link_to_previous_hash(ledger_path):
    entries = Ledger(ledger_path).entries()
    for prev, cur in zip(entries, entries[1:]):
        assert cur["prev_hash"] == prev["entry_hash"]


def test_unknown_event_rejected(ledger_path):
    with pytest.raises(ValueError):
        Ledger(ledger_path).append("run-1", "rewrite_history", {})


@pytest.mark.parametrize("sql", [
    "UPDATE ledger SET payload = '{}' WHERE seq = 1",
    "DELETE FROM ledger WHERE seq = 2",
])
def test_update_and_delete_are_refused(ledger_path, sql):
    raw = sqlite3.connect(ledger_path)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        raw.execute(sql)


def _drop_triggers(raw):
    raw.execute("DROP TRIGGER ledger_no_update")
    raw.execute("DROP TRIGGER ledger_no_delete")


def test_edited_entry_is_detected(ledger_path):
    raw = sqlite3.connect(ledger_path)
    _drop_triggers(raw)
    raw.execute("""UPDATE ledger SET payload = '{"snapshot":{"plan":"starter","status":"cancelled"}}' WHERE seq = 1""")
    raw.commit()
    assert not Ledger(ledger_path).verify_chain()


def test_rehashed_edit_still_breaks_next_link(ledger_path):
    # An attacker who edits an entry and recomputes its own hash still breaks
    # the prev_hash stored in the following entry.
    raw = sqlite3.connect(ledger_path)
    _drop_triggers(raw)
    ts, run_id, event, prev = raw.execute(
        "SELECT ts, run_id, event, prev_hash FROM ledger WHERE seq = 1").fetchone()
    forged = '{"snapshot":{}}'
    raw.execute("UPDATE ledger SET payload = ?, entry_hash = ? WHERE seq = 1",
                (forged, _hash_entry(prev, ts, run_id, event, forged)))
    raw.commit()
    assert not Ledger(ledger_path).verify_chain()


def test_deleted_entry_is_detected(ledger_path):
    raw = sqlite3.connect(ledger_path)
    _drop_triggers(raw)
    raw.execute("DELETE FROM ledger WHERE seq = 2")
    raw.commit()
    assert not Ledger(ledger_path).verify_chain()
