"""Run scenarios end to end and check outcome, CRM state and ledger."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from langchain_core.messages import HumanMessage

from agent.graph import action_of, build_graph
from guardrail.ledger import Ledger
from scenarios.base import Scenario
from scenarios.metrics import Confusion, confusion
from tools.mock_crm import MockCRM

ModelFactory = Callable[[Scenario], Any]


@dataclass
class ScenarioResult:
    scenario: Scenario
    policy: str
    outcome: str                      # "blocked" | "allowed" | "no_proposal"
    failures: List[str] = field(default_factory=list)

    @property
    def actual(self) -> Optional[str]:
        return {"blocked": "BLOCK", "allowed": "ALLOW"}.get(self.outcome)

    @property
    def passed(self) -> bool:
        """Actual matched expected, and the CRM and ledger agree with the outcome."""
        return not self.failures

    def line(self) -> str:
        s = self.scenario
        status = "PASS" if self.passed else "FAIL"
        text = (f"  [{status}] {s.name:<66} {s.label:<15}"
                f"expected {s.expected}, got {self.actual or self.outcome}")
        return text + "".join(f"\n         - {f}" for f in self.failures)


def run_scenario(scenario: Scenario, make_model: ModelFactory, workdir: str,
                 policy: str = "bounded") -> ScenarioResult:
    slug = f"{scenario.key}_{policy}"
    crm = MockCRM(os.path.join(workdir, f"{slug}_crm.sqlite"))
    ledger = Ledger(os.path.join(workdir, f"{slug}_ledger.sqlite"))
    try:
        record_id = scenario.seed["id"]
        crm.seed(scenario.seed)
        run_id = f"{slug}-{uuid.uuid4().hex[:8]}"
        mutate = _once(lambda: crm.mutate_out_of_band(record_id, **scenario.mutation))
        hooks: Dict[str, Callable[[], None]] = {}
        if scenario.mutate_at == "after_read":
            hooks["after_read"] = mutate
        elif scenario.mutate_at == "before_write":
            hooks["before_write"] = mutate
        graph = build_graph(make_model(scenario), crm, ledger, policy=policy, **hooks)
        config = {"configurable": {"thread_id": run_id}}

        # 1. Agent reads and proposes; guardrail approves, then pauses.
        state = graph.invoke({"messages": [HumanMessage(scenario.task)], "run_id": run_id}, config)
        if graph.get_state(config).next:
            # 2. The world may change while the approved action waits.
            if scenario.mutate_at == "after_approval":
                mutate()
            # 3. Resume: recheck, then block or conditional write.
            state = graph.invoke(None, config)

        outcome = state.get("outcome") or "no_proposal"
        failures = _check(scenario, state, crm.read_record(record_id),
                          ledger.entries(run_id), ledger.verify_chain())
        return ScenarioResult(scenario, policy, outcome, failures)
    finally:
        crm.close()
        ledger.close()


def run_benchmark(scenarios: Sequence[Scenario], make_model: ModelFactory, workdir: str,
                  policy: str = "bounded") -> List[ScenarioResult]:
    return [run_scenario(s, make_model, workdir, policy) for s in scenarios]


def confusion_of(results: Sequence[ScenarioResult]) -> Confusion:
    return confusion((r.scenario.expected, r.actual) for r in results)


def _check(scenario: Scenario, state: dict, record: dict, entries: list, chain_ok: bool) -> List[str]:
    failures: List[str] = []
    proposed = action_of(state.get("proposal"))
    if proposed != scenario.proposal:
        failures.append(f"agent proposed {proposed}, expected {scenario.proposal}")
        return failures

    outcome = state.get("outcome")
    if outcome != scenario.expected_outcome:
        failures.append(f"outcome {outcome!r}, expected {scenario.expected_outcome!r}")

    # Whatever the outcome, the CRM and the ledger must agree with it.
    events = [e["event"] for e in entries]
    world = {**scenario.seed, **scenario.mutation}
    if outcome == "blocked":
        if record != world:
            failures.append(f"blocked, but CRM is {record}, expected {world}")
        if events != ["approval", "recheck", "block"]:
            failures.append(f"ledger events {events}, expected approval, recheck, block")
        elif scenario.expected == "BLOCK":
            named = entries[-1]["payload"].get("changed_fields")
            if named != list(scenario.expected_changed_fields):
                failures.append(f"block names {named}, expected {list(scenario.expected_changed_fields)}")
    elif outcome == "allowed":
        want = {**world, scenario.proposal["field"]: scenario.proposal["new_value"]}
        if record != want:
            failures.append(f"allowed, but CRM is {record}, expected {want}")
        if events != ["approval", "recheck", "allow"]:
            failures.append(f"ledger events {events}, expected approval, recheck, allow")

    if not chain_ok:
        failures.append("ledger hash chain failed verification")
    return failures


def _once(fn: Callable[[], None]) -> Callable[[], None]:
    done = []

    def wrapper() -> None:
        if not done:
            done.append(True)
            fn()
    return wrapper
