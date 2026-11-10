"""The decision viewer's wording, on hand-written ledger entries (no graph run)."""

from scenarios.explain import condition_sentence, explain_entry, explain_ledger

UPGRADE = {"tool": "crm", "record_id": "C-1001", "field": "plan", "new_value": "growth"}
SEND = {"tool": "mailer", "contact_id": "M-2001", "subject": "Your plan upgrade", "body": "Hi Dana"}
KEY = "a" * 64


def entry(event, payload, run_id="r1"):
    return {"seq": 1, "run_id": run_id, "event": event, "payload": payload}


APPROVAL = entry("approval", {
    "proposal": UPGRADE, "policy": "canonical", "decision_read_at": "2026-11-15T10:00:00+00:00",
    "decision_snapshot": {"plan": "starter", "status": "active", "seats": 5},
    "conditions": [{"field": "plan", "op": "==", "value": "starter", "converges_to": "growth"},
                   {"field": "status", "op": "==", "value": "active"},
                   {"field": "seats", "op": "<=", "value": 20}],
    "idempotency_key": KEY,
})
STALE_RECHECK = entry("recheck", {"proposal": UPGRADE, "verdict": {
    "changed": True, "changed_fields": ["status"],
    "diffs": {"status": {"approved": "active", "current": "cancelled"}}}})
STALE_BLOCK = entry("block", {"proposal": UPGRADE, "stage": "recheck", "changed_fields": ["status"],
                              "diffs": {"status": {"approved": "active", "current": "cancelled"}}})


def text(lines):
    return "\n".join(lines)


def test_block_names_the_field_that_moved():
    out = text(explain_ledger([APPROVAL, STALE_RECHECK, STALE_BLOCK], chain_ok=True))
    assert 'status changed from "active" to "cancelled"' in out
    assert "Blocked: status moved (at recheck)." in out
    assert out.endswith("Ledger hash chain verifies: yes")


def test_approval_says_what_was_read_proposed_and_required():
    out = text(explain_entry(APPROVAL))
    assert 'The agent read: plan = "starter", status = "active", seats = 5' in out
    assert 'It proposed: set plan to "growth" on C-1001 (crm)' in out
    assert "Policy: canonical" in out
    assert "status must stay \"active\"" in out
    assert "seats must stay at or below 20" in out
    assert 'or already be "growth", the value this action writes' in out


def test_canonical_case_is_noted_as_accepted():
    recheck = entry("recheck", {"proposal": SEND, "verdict": {
        "changed": False, "changed_fields": [], "diffs": {},
        "accepted": {"recipient_email": {"approved": "Dana@Acme.example", "current": "dana@acme.example",
                                         "as": "canonical"}}}})
    out = text(explain_entry(recheck))
    assert "recipient_email moved but was accepted as canonical" in out
    assert "Nothing the action depends on changed" not in out


def test_condition_sentences():
    assert condition_sentence({"field": "seats", "op": ">=", "value": 10}) == "seats must stay at or above 10"
    canon = condition_sentence({"field": "recipient_email", "op": "==", "value": "dana@acme.example",
                                "canonicalize": "canonical_email"})
    assert canon.startswith('recipient_email must stay "dana@acme.example"') and "canonical_email" in canon


def test_bound_diff_reads_as_past_its_limit_not_as_a_change_from_the_limit():
    recheck = entry("recheck", {"proposal": UPGRADE, "verdict": {
        "changed": True, "changed_fields": ["seats"],
        "diffs": {"seats": {"approved": 20, "current": 21, "op": "<="}}}})
    assert "seats is now 21, past its limit (at or below 20)" in text(explain_entry(recheck))


def test_unchanged_recheck_allow_and_duplicate():
    quiet = entry("recheck", {"proposal": UPGRADE, "verdict": {"changed": False, "changed_fields": [], "diffs": {}}})
    assert "Nothing the action depends on changed." in text(explain_entry(quiet))
    assert explain_entry(entry("allow", {"proposal": UPGRADE, "write": "ok", "idempotency_key": KEY}))[0] \
        == "Allowed: the write went through."
    dup = entry("duplicate_suppressed", {"proposal": UPGRADE, "idempotency_key": KEY, "original": {}})
    assert explain_entry(dup)[0] == ("Duplicate suppressed: this approved action already took effect; "
                                     "returned the original result.")


def test_block_at_the_write_says_what_moved_after_the_recheck():
    block = entry("block", {"proposal": SEND, "stage": "conditional_update",
                            "reason": "stale: contact no longer matches snapshot",
                            "changed_fields": ["opt_in_status"],
                            "diffs": {"opt_in_status": {"approved": "opted_in", "current": "opted_out"}}})
    out = text(explain_entry(block))
    assert "Blocked: opt_in_status moved (at conditional_update)." in out
    assert 'opt_in_status changed from "opted_in" to "opted_out"' in out


def test_two_plans_are_shown_separately_and_a_broken_chain_says_no():
    out = text(explain_ledger([APPROVAL, entry("approval", APPROVAL["payload"], run_id="r2")], chain_ok=False))
    assert "Plan 1 of 2 (run r1)" in out and "Plan 2 of 2 (run r2)" in out
    assert out.endswith("Ledger hash chain verifies: no")


def test_no_raw_json_in_the_output():
    out = text(explain_ledger([APPROVAL, STALE_RECHECK, STALE_BLOCK], chain_ok=True))
    assert "{" not in out and "}" not in out
