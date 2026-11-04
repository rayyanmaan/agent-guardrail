"""Run scenarios end to end and check outcome, tool state and ledger.

Tool-agnostic: the caller passes the tool types (name -> class with .open(path))
to build for each scenario. Each tool must also provide the harness methods
seed, mutate_out_of_band, dump, simulate_effect and close (see tools/crm_tool.py).
"""

from __future__ import annotations

import copy
import os
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from langchain_core.messages import HumanMessage

from agent.graph import action_of, build_graph
from guardrail.ledger import Ledger
from scenarios.base import Scenario
from scenarios.metrics import Confusion, Scored, confusion

ModelFactory = Callable[[Scenario], Any]


class AckLost(Exception):
    """Fault injection: the write committed, but its acknowledgement never arrived."""


@dataclass
class ScenarioResult:
    scenario: Scenario
    policy: str
    outcome: str                      # "blocked" | "allowed" | "deduplicated" | "no_proposal"
    proposal_ok: bool = True          # the agent proposed the action the scenario is about
    integrity: List[str] = field(default_factory=list)  # outcome, state and ledger disagree
    invalid_cause: Optional[str] = None  # when proposal_ok is False, see _invalid_cause
    refusals: Dict[str, int] = field(default_factory=dict)  # summed over attempts, see AgentState
    wording_differs: bool = False     # matched on match_keys, but not word for word

    @property
    def actual(self) -> Optional[str]:
        """BLOCK / ALLOW, or None if this run is not a valid test of the scenario.

        A suppressed duplicate counts as BLOCK: the second effect did not happen.
        """
        if not self.proposal_ok:
            return None
        return {"blocked": "BLOCK", "allowed": "ALLOW", "deduplicated": "BLOCK"}.get(self.outcome)

    @property
    def passed(self) -> bool:
        """Actual matched expected, and tool state and ledger agree with the outcome."""
        return self.actual == self.scenario.expected and not self.integrity

    @property
    def failures(self) -> List[str]:
        out = list(self.integrity)
        if not self.proposal_ok:
            out.append("agent did not propose the scenario's action")
        elif self.actual != self.scenario.expected:
            out.append(f"outcome {self.outcome!r}, expected {self.scenario.expected_outcome!r}")
        return out

    @property
    def verdict(self) -> str:
        if self.integrity:
            return "ERROR"
        if self.actual is None:
            return "INVALID"
        if self.actual == self.scenario.expected:
            return "PASS"
        if self.scenario.expected == "ALLOW":
            return "FALSE ALARM"
        return "MISS" if self.scenario.in_scope else "MISS (out of scope)"

    def scored(self) -> Scored:
        s = self.scenario
        return Scored(s.key, s.tool, s.shape, s.in_scope, s.expected, self.actual,
                      None if self.actual is not None else (self.invalid_cause or self.outcome),
                      s.mechanism)

    def line(self) -> str:
        s = self.scenario
        text = (f"  {'[' + self.verdict + ']':<22}{s.tool:<7}{s.name:<88}"
                f"expected {s.expected}, got {self.actual or self.outcome}")
        if self.actual is None and self.invalid_cause:
            text += f" ({self.invalid_cause})"
        return text + "".join(f"\n         - {f}" for f in self.integrity)


def run_scenario(scenario: Scenario, make_model: ModelFactory, workdir: str,
                 tool_types: Mapping[str, Any], policy: str = "bounded") -> ScenarioResult:
    slug = f"{scenario.key}_{policy}"
    tools = {name: cls.open(os.path.join(workdir, f"{slug}_{name}.sqlite"))
             for name, cls in tool_types.items()}
    ledger = Ledger(os.path.join(workdir, f"{slug}_ledger.sqlite"))
    try:
        for name, rows in scenario.seed.items():
            tools[name].seed(rows)
        start = {name: t.dump() for name, t in tools.items()}
        integrity: List[str] = []
        attempts: List[str] = []             # one run id per plan; each run is its own plan
        refusals: Counter = Counter()
        earlier: List[Dict[str, Any]] = []   # effects that landed before the mutations
        written: List[Dict[str, Any]] = []   # effects that landed after them

        if scenario.fault == "repeat_plan":
            # The same action, approved and applied in an earlier, separate plan.
            attempts.append(_new_run_id(slug))
            first, _ = _attempt(scenario, make_model, tools, ledger, policy, attempts[-1], {})
            refusals.update(first.get("refusals") or {})
            if not _matches(scenario, action_of(first.get("proposal"))):
                return ScenarioResult(scenario, policy, first.get("outcome") or "no_proposal", False,
                                      [], _invalid_cause(scenario, first), dict(refusals))
            if first.get("outcome") == "allowed":
                earlier.append(_action_only(first["proposal"]))
            else:
                integrity.append(f"earlier plan was {first.get('outcome')!r}, expected 'allowed'")
            _apply_mutations(scenario, tools)

        mutate = _once(lambda: _apply_mutations(scenario, tools))
        hooks = {"after_read": mutate} if scenario.mutate_at == "after_read" else \
                {"before_write": mutate} if scenario.mutate_at == "before_write" else {}
        if scenario.fault == "lost_ack":
            hooks["after_write"] = _once(_drop_ack)
        between = mutate if scenario.mutate_at == "after_approval" else None
        attempts.append(_new_run_id(slug))
        state, ack_lost = _attempt(scenario, make_model, tools, ledger, policy, attempts[-1], hooks, between)
        refusals.update(state.get("refusals") or {})

        outcome = state.get("outcome") or "no_proposal"
        proposed = action_of(state.get("proposal"))
        if not _matches(scenario, proposed):
            # For lost_ack this also means the model never got as far as the write.
            return ScenarioResult(scenario, policy, outcome, False, integrity,
                                  _invalid_cause(scenario, state), dict(refusals))
        if ack_lost:
            written.append(_action_only(state["proposal"]))   # the first write committed
        elif scenario.fault == "lost_ack":
            integrity.append("lost-ack fault never fired: the first write did not commit")
        if outcome == "allowed":
            written.append(_action_only(state["proposal"]))
        integrity += _check(scenario, outcome, tools, start, ledger, attempts, earlier, written)
        return ScenarioResult(scenario, policy, outcome, True, integrity, None, dict(refusals),
                              wording_differs=proposed != scenario.proposal)
    finally:
        for t in tools.values():
            t.close()
        ledger.close()


def run_benchmark(scenarios: Sequence[Scenario], make_model: ModelFactory, workdir: str,
                  tool_types: Mapping[str, Any], policy: str = "bounded") -> List[ScenarioResult]:
    return [run_scenario(s, make_model, workdir, tool_types, policy) for s in scenarios]


def confusion_of(results: Sequence[ScenarioResult]) -> Confusion:
    return confusion((r.scenario.expected, r.actual) for r in results)


def _attempt(scenario, make_model, tools, ledger, policy, run_id, hooks, between=None):
    """Run one plan through the graph. Returns (final state, whether an ack was lost).

    If the effect's acknowledgement is lost, the same approved action is
    retried: the run resumes from its last checkpoint, which re-runs the effect
    with the state approval left behind, idempotency key included. The model is
    not consulted again.
    """
    graph = build_graph(make_model(scenario), list(tools.values()), ledger, policy=policy, **hooks)
    config = {"configurable": {"thread_id": run_id}}
    # 1. Agent reads and proposes; guardrail approves, then pauses.
    state = graph.invoke({"messages": [HumanMessage(scenario.task)], "run_id": run_id,
                          "plan_id": run_id}, config)
    if not graph.get_state(config).next:
        return state, False
    # 2. The world may change while the approved action waits.
    if between is not None:
        between()
    # 3. Resume: recheck, then block or conditional effect.
    try:
        return graph.invoke(None, config), False
    except AckLost:
        # 4. The effect committed but its acknowledgement never arrived. Retry.
        return graph.invoke(None, config), True


def _matches(scenario: Scenario, proposed: Optional[Dict[str, Any]]) -> bool:
    """Did the agent propose the scenario's action, on the keys that define it?"""
    if proposed is None:
        return False
    keys = scenario.match_keys or tuple(scenario.proposal)
    return all(proposed.get(k) == scenario.proposal[k] for k in keys)


def _invalid_cause(scenario: Scenario, state: dict) -> str:
    """Why a run is not a valid test of its scenario.

    different_action   proposed something other than the scenario's action
    validation_format  never got a proposal past validation (bad field or value format)
    no_proposal        never produced a proposal for any other reason
    """
    if state.get("proposal") is not None:
        return "different_action"
    if (state.get("refusals") or {}).get("invalid_args"):
        return "validation_format"
    return "no_proposal"


def _action_only(proposal: Dict[str, Any]) -> Dict[str, Any]:
    return dict(proposal["action"])


def _check(scenario: Scenario, outcome: str, tools: Dict[str, Any], start: Dict[str, Any],
           ledger: Ledger, attempts: List[str], earlier: List[Dict[str, Any]],
           written: List[Dict[str, Any]]) -> List[str]:
    failures: List[str] = []

    # Tool state must be exactly what the outcome implies, computed independently
    # of the tools' own write code. `earlier` and `written` hold the actions the
    # agent actually proposed whose effect should have landed, before and after
    # the mutations (a real model writes its own text).
    expected_state = _expected_state(scenario, tools, start, earlier, written)
    for name, t in tools.items():
        actual = t.dump()
        if actual != expected_state[name]:
            failures.append(f"{outcome}, but {name} state is {actual}, expected {expected_state[name]}")

    # A lost ack means the first effect was never logged: the run's ledger shows
    # only what happened on the retry.
    want = {"blocked": ["approval", "recheck", "block"],
            "allowed": ["approval", "recheck", "allow"],
            "deduplicated": ["approval", "recheck", "duplicate_suppressed"]}.get(outcome)
    entries = ledger.entries(attempts[-1])
    events = [e["event"] for e in entries]
    if want is not None and events != want:
        failures.append(f"ledger events {events}, expected {want}")
    elif outcome == "blocked" and scenario.expected == "BLOCK" and scenario.expected_changed_fields:
        named = entries[-1]["payload"].get("changed_fields")
        if named != list(scenario.expected_changed_fields):
            failures.append(f"block names {named}, expected {list(scenario.expected_changed_fields)}")
    elif outcome == "deduplicated":
        # The duplicate was refused under the key minted at this run's approval.
        if entries[-1]["payload"]["idempotency_key"] != entries[0]["payload"]["idempotency_key"]:
            failures.append("suppressed key differs from the key minted at approval")
    if scenario.fault == "repeat_plan":
        first = [e["event"] for e in ledger.entries(attempts[0])]
        if first != ["approval", "recheck", "allow"]:
            failures.append(f"earlier plan ledger events {first}, expected approval, recheck, allow")

    if not ledger.verify_chain():
        failures.append("ledger hash chain failed verification")
    return failures


def _expected_state(scenario: Scenario, tools: Dict[str, Any], start: Dict[str, Any],
                    earlier: List[Dict[str, Any]], written: List[Dict[str, Any]]) -> Dict[str, Any]:
    state = copy.deepcopy(start)
    acting = state[scenario.tool]
    simulate = tools[scenario.tool].simulate_effect
    for action in earlier:
        acting["effects"] += simulate(action, acting["records"])
    for tool_name, target, changes in scenario.mutations:
        state[tool_name]["records"][target].update(changes)
    for action in written:
        acting["effects"] += simulate(action, acting["records"])
    return state


def _apply_mutations(scenario: Scenario, tools: Dict[str, Any]) -> None:
    for tool_name, target, changes in scenario.mutations:
        tools[tool_name].mutate_out_of_band(target, changes)


def _new_run_id(slug: str) -> str:
    return f"{slug}-{uuid.uuid4().hex[:8]}"


def _drop_ack() -> None:
    raise AckLost()


def _once(fn: Callable[[], None]) -> Callable[[], None]:
    done = []

    def wrapper() -> None:
        if not done:
            done.append(True)
            fn()
    return wrapper
