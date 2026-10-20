"""Run the labelled CRM benchmark and report catch rate and false-alarm rate.

    python run.py                    # agent driven by Ollama (qwen2.5:7b-instruct)
    python run.py --scripted         # deterministic planner instead of the LLM
    python run.py --policy bounded   # one policy only (default: bounded and naive)
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings

import langchain_core  # noqa: E402,F401  (installs its own warning filters on import)

# Environment noise unrelated to the results (old LibreSSL; a LangChain notice).
warnings.filterwarnings("ignore", message=".*OpenSSL.*")
warnings.filterwarnings("ignore", message=".*allowed_objects.*")

from agent.graph import ScriptedPlanner, make_ollama_model  # noqa: E402
from scenarios.crm_benchmark import SCENARIOS  # noqa: E402
from scenarios.metrics import format_rate, format_table  # noqa: E402
from scenarios.runner import confusion_of, run_benchmark  # noqa: E402

POLICY_NAMES = ("bounded", "naive")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scripted", action="store_true", help="use the deterministic planner, no Ollama")
    parser.add_argument("--model", default="qwen2.5:7b-instruct", help="Ollama model name")
    parser.add_argument("--policy", choices=POLICY_NAMES + ("both",), default="both")
    parser.add_argument("--workdir", help="where to write the SQLite files (default: a temp dir)")
    args = parser.parse_args()

    if args.scripted:
        make_model = lambda s: ScriptedPlanner(**s.proposal)
        driver = "scripted planner"
    else:
        model = make_ollama_model(args.model)
        make_model = lambda s: model
        driver = f"ollama:{args.model}"
    policies = POLICY_NAMES if args.policy == "both" else (args.policy,)

    workdir = args.workdir or tempfile.mkdtemp(prefix="guardrail-run-")
    print(f"Agent driver: {driver}\nDatabases:    {workdir}\nScenarios:    {len(SCENARIOS)}")

    summary = {}
    ok = True
    for policy in policies:
        print(f"\n=== Policy: {policy} ===")
        try:
            results = run_benchmark(SCENARIOS, make_model, workdir, policy)
        except Exception as exc:  # e.g. Ollama not running
            print(f"ERROR — {type(exc).__name__}: {exc}")
            if not args.scripted:
                print("(is `ollama serve` running and the model pulled? or try --scripted)")
            return 1
        for r in results:
            print(r.line())
        c = confusion_of(results)
        summary[policy] = c
        print()
        print(format_table(c))
        print(f"\n  catch rate:       {format_rate(c.blocked_should_block, c.dangerous)}")
        print(f"  false-alarm rate: {format_rate(c.blocked_should_allow, c.legitimate)}")
        # The bounded policy is the one under test: every scenario must pass.
        # For the naive baseline, wrong outcomes are the point; only
        # inconsistencies between outcome, CRM and ledger count against it.
        if policy == "bounded":
            ok &= all(r.passed for r in results)
        else:
            ok &= all(not _integrity_failures(r) for r in results)

    if len(summary) > 1:
        print("\n=== Policy comparison ===")
        print(f"  {'policy':<10}{'catch rate':<16}{'false-alarm rate'}")
        for policy, c in summary.items():
            print(f"  {policy:<10}{format_rate(c.blocked_should_block, c.dangerous):<16}"
                  f"{format_rate(c.blocked_should_allow, c.legitimate)}")
    return 0 if ok else 1


def _integrity_failures(result):
    return [f for f in result.failures if not f.startswith("outcome ")]


if __name__ == "__main__":
    sys.exit(main())
