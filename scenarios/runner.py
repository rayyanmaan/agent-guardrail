"""Run a scenario end to end and check the outcome, the CRM and the ledger."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, List

from langchain_core.messages import HumanMessage

from agent.graph import build_graph
from guardrail.ledger import Ledger
from scenarios.base import Scenario
from tools.mock_crm import MockCRM

ModelFactory = Callable[[Scenario], Any]


@dataclass
class ScenarioResult:
    scenario: Scenario
    outcome: str
    failures: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        text = f"{self.scenario.name}: {self.outcome.upper()} — {status}"
        return text + "".join(f"\n    - {f}" for f in self.failures)


def run_scenario(scenario: Scenario, make_model: ModelFactory, workdir: str) -> ScenarioResult:
    slug = scenario.name.split(" (")[0].lower().replace(" ", "_")
    crm = MockCRM(os.path.join(workdir, f"{slug}_crm.sqlite"))
    ledger = Ledger(os.path.join(workdir, f"{slug}_ledger.sqlite"))
    try:
        record_id = scenario.seed["id"]
        crm.seed(scenario.seed)
        run_id = f"{slug}-{uuid.uuid4().hex[:8]}"
        graph = build_graph(make_model(scenario), crm, ledger)
        config = {"configurable": {"thread_id": run_id}}

        # 1. Agent reads and proposes; guardrail approves against S0, then pauses.
        graph.invoke({"messages": [HumanMessage(scenario.task)], "run_id": run_id}, config)
        # 2. The world changes while the approved action waits.
        crm.mutate_out_of_band(record_id, **scenario.out_of_band)
        # 3. Resume: recheck, then block or conditional write.
        state = graph.invoke(None, config)

        outcome = state.get("outcome") or "no_outcome"
        failures = _check(scenario, state, crm.read_record(record_id),
                          ledger.entries(run_id), ledger.verify_chain())
        return ScenarioResult(scenario, outcome, failures)
    finally:
        crm.close()
        ledger.close()


def _check(scenario: Scenario, state: dict, record: dict, entries: list, chain_ok: bool) -> List[str]:
    failures: List[str] = []
    expected = scenario.proposal
    if state.get("proposal") != expected:
        failures.append(f"agent proposed {state.get('proposal')}, expected {expected}")
        return failures

    events = [e["event"] for e in entries]
    field_name = expected["field"]
    if state.get("outcome") != scenario.expected_outcome:
        failures.append(f"outcome {state.get('outcome')!r}, expected {scenario.expected_outcome!r}")

    if scenario.expected_outcome == "blocked":
        if record[field_name] != scenario.seed[field_name]:
            failures.append(f"CRM was written: {field_name}={record[field_name]!r}")
        if events[:1] != ["approval"] or "block" not in events:
            failures.append(f"ledger events {events}, expected approval ... block")
        else:
            block = next(e for e in entries if e["event"] == "block")
            named = block["payload"].get("changed_fields", [])
            if list(scenario.expected_changed_fields) != named:
                failures.append(f"block names {named}, expected {list(scenario.expected_changed_fields)}")
    else:
        if record[field_name] != expected["new_value"]:
            failures.append(f"CRM not updated: {field_name}={record[field_name]!r}")
        for k, v in scenario.out_of_band.items():
            if record[k] != v:
                failures.append(f"out-of-band change to {k} was lost")
        if events != ["approval", "recheck", "allow"]:
            failures.append(f"ledger events {events}, expected approval, recheck, allow")

    if not chain_ok:
        failures.append("ledger hash chain failed verification")
    return failures
