"""Ledger entries in plain English, for `python run.py --explain <scenario_key>`.

Pure formatting: takes ledger entries (as returned by Ledger.entries()) and
returns lines of text. It never runs the graph, so it can be tested on
hand-written entries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

BOUND_WORDS = {"<=": "at or below", ">=": "at or above"}


def explain_ledger(entries: Sequence[Dict[str, Any]], chain_ok: bool) -> List[str]:
    """Every entry, grouped by run (a repeat-plan scenario has two), then the chain check."""
    runs: List[str] = []
    for e in entries:
        if e["run_id"] not in runs:
            runs.append(e["run_id"])
    lines: List[str] = []
    for n, run_id in enumerate(runs, 1):
        if len(runs) > 1:
            lines += ["", f"--- Plan {n} of {len(runs)} (run {run_id}) ---"]
        for e in entries:
            if e["run_id"] == run_id:
                lines += [""] + explain_entry(e)
    lines += ["", f"Ledger hash chain verifies: {'yes' if chain_ok else 'no'}"]
    return lines


def explain_entry(entry: Dict[str, Any]) -> List[str]:
    """One ledger entry as a heading plus indented sentences."""
    event, p = entry["event"], entry["payload"]
    if event == "approval":
        return _approval(p)
    if event == "recheck":
        return _recheck(p)
    if event == "block":
        return _block(p)
    if event == "allow":
        return ["Allowed: the write went through.", f"  Key: {_short(p.get('idempotency_key'))}"]
    if event == "duplicate_suppressed":
        return ["Duplicate suppressed: this approved action already took effect; returned the original result.",
                f"  Key: {_short(p.get('idempotency_key'))}"]
    return [f"{event}: (no description for this event)"]


def _approval(p: Dict[str, Any]) -> List[str]:
    snapshot = ", ".join(f"{f} = {_value(v)}" for f, v in p.get("decision_snapshot", {}).items())
    lines = ["Approval",
             f"  The agent read: {snapshot}",
             f"  It proposed: {_proposal(p.get('proposal') or {})}",
             f"  Policy: {p.get('policy')}"]
    if p.get("dependency_set"):
        lines.append(f"  Declared dependencies: {declared_dependencies(p['dependency_set'])}")
    lines.append("  Conditions checked before the write:")
    lines += [f"    - {condition_sentence(c)}" for c in p.get("conditions", [])]
    lines.append(f"  Idempotency key: {_short(p.get('idempotency_key'))}")
    return lines


def declared_dependencies(deps: Sequence[Dict[str, Any]]) -> str:
    return ", ".join(f"{d['field']} (bounded, {d['op']})" if d["kind"] == "Bound"
                     else f"{d['field']} (must not change)" for d in deps)


def condition_sentence(c: Dict[str, Any]) -> str:
    if c["op"] == "==":
        text = f"{c['field']} must stay {_value(c['value'])}"
    else:
        text = f"{c['field']} must stay {BOUND_WORDS[c['op']]} {_value(c['value'])}"
    if "canonicalize" in c:
        text += f" (compared in canonical form, via {c['canonicalize']})"
    if "converges_to" in c:
        text += f" (or already be {_value(c['converges_to'])}, the value this action writes)"
    return text


def _recheck(p: Dict[str, Any]) -> List[str]:
    verdict = p.get("verdict") or {}
    diffs, accepted = verdict.get("diffs") or {}, verdict.get("accepted") or {}
    lines = ["Recheck (the record read again, right before the write)"]
    lines += [f"  {_diff_sentence(f, d)}" for f, d in diffs.items()]
    lines += [f"  {f} moved but was accepted as {a['as']} ({_value(a['approved'])} -> {_value(a['current'])})"
              for f, a in accepted.items()]
    if not diffs and not accepted:
        lines.append("  Nothing the action depends on changed.")
    return lines


def _block(p: Dict[str, Any]) -> List[str]:
    fields = ", ".join(p.get("changed_fields") or []) or "a dependency"
    lines = [f"Blocked: {fields} moved (at {p.get('stage')})."]
    if p.get("stage") != "recheck":
        # The recheck passed, so its entry shows no change; say what moved after it.
        lines += [f"  {_diff_sentence(f, d)}" for f, d in (p.get("diffs") or {}).items()]
    if p.get("reason"):
        lines.append(f"  Reason: {p['reason']}")
    return lines


def _diff_sentence(field: str, d: Dict[str, Any]) -> str:
    if "op" in d:
        # For a bound, "approved" is the limit, not an earlier value.
        return f"{field} is now {_value(d['current'])}, past its limit ({BOUND_WORDS[d['op']]} {_value(d['approved'])})"
    return f"{field} changed from {_value(d['approved'])} to {_value(d['current'])}"


def _proposal(a: Dict[str, Any]) -> str:
    if "field" in a and "new_value" in a:
        return f"set {a['field']} to {_value(a['new_value'])} on {a.get('record_id')} ({a.get('tool')})"
    if "subject" in a:
        return f"send the email {_value(a['subject'])} to contact {a.get('contact_id')} ({a.get('tool')})"
    return ", ".join(f"{k} = {_value(v)}" for k, v in a.items())


def _value(v: Any) -> str:
    if v is None:
        return "nothing"
    if isinstance(v, str):
        return f'"{v}"'
    return str(v)


def _short(key: Any) -> str:
    return f"{key[:12]}..." if isinstance(key, str) and len(key) > 12 else str(key)
