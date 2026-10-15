"""Run the walking-skeleton scenarios and print PASS/FAIL per scenario.

    python run.py              # agent driven by Ollama (qwen2.5:7b-instruct)
    python run.py --scripted   # deterministic planner instead of the LLM
"""

from __future__ import annotations

import argparse
import sys
import tempfile

from agent.graph import ScriptedPlanner, make_ollama_model
from scenarios.runner import run_scenario
from scenarios.scenario_a_stale import SCENARIO as SCENARIO_A
from scenarios.scenario_b_legitimate import SCENARIO as SCENARIO_B

SCENARIOS = [SCENARIO_A, SCENARIO_B]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scripted", action="store_true", help="use the deterministic planner, no Ollama")
    parser.add_argument("--model", default="qwen2.5:7b-instruct", help="Ollama model name")
    parser.add_argument("--workdir", help="where to write the SQLite files (default: a temp dir)")
    args = parser.parse_args()

    if args.scripted:
        make_model = lambda s: ScriptedPlanner(**s.proposal)
        driver = "scripted planner"
    else:
        model = make_ollama_model(args.model)
        make_model = lambda s: model
        driver = f"ollama:{args.model}"

    workdir = args.workdir or tempfile.mkdtemp(prefix="guardrail-run-")
    print(f"Agent driver: {driver}\nDatabases:    {workdir}\n")

    all_passed = True
    for scenario in SCENARIOS:
        try:
            result = run_scenario(scenario, make_model, workdir)
        except Exception as exc:  # e.g. Ollama not running
            print(f"{scenario.name}: ERROR — {type(exc).__name__}: {exc}")
            if not args.scripted:
                print("    (is `ollama serve` running and the model pulled? or try --scripted)")
            all_passed = False
            continue
        print(result.line())
        all_passed &= result.passed
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
