"""Run the labelled benchmark (CRM + mailer) and report the guardrail's numbers.

    python run.py --scripted         # headline benchmark: deterministic scripted planner
    python run.py                    # validation pass: real agent via Ollama (qwen2.5:7b-instruct)
    python run.py --policy canonical # one policy only (default: all three)

Policies: naive (strict equality on every dependency), bounded (equality +
bounds, strict comparison), canonical (bounded + per-field canonicalization and
target convergence on equality fields).

Three numbers, never combined:
  in-scope catch    stale-facts cases blocked / all stale-facts cases
  overall coverage  all dangerous cases blocked / all dangerous cases, including
                    failure shapes recheck-before-effect is not built to catch
  false-alarm rate  legitimate cases blocked / all legitimate cases
plus, separately, the invalid rate: runs that were not a valid test of their
scenario (excluded from the three rates above), broken down by cause.

The scripted run is the reproducible headline benchmark. The Ollama run is a
validation pass: it checks the guardrail behaves the same with a real agent,
and its invalid rate and observations are model-reliability findings.
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
from scenarios.benchmark import SCENARIOS  # noqa: E402
from scenarios.metrics import Summary, by_tool, format_rate, format_table, summarize  # noqa: E402
from scenarios.runner import run_benchmark  # noqa: E402
from tools.registry import TOOL_SCHEMAS, TOOL_TYPES  # noqa: E402

POLICY_NAMES = ("naive", "bounded", "canonical")
METRICS = (
    ("in-scope catch", lambda s: (s.in_scope.blocked_should_block, s.in_scope.dangerous)),
    ("overall coverage", lambda s: (s.overall.blocked_should_block, s.overall.dangerous)),
    ("false-alarm rate", lambda s: (s.overall.blocked_should_allow, s.overall.legitimate)),
    ("invalid rate", lambda s: (s.invalid_count, s.runs)),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scripted", action="store_true", help="use the deterministic planner, no Ollama")
    parser.add_argument("--model", default="qwen2.5:7b-instruct", help="Ollama model name")
    parser.add_argument("--policy", choices=POLICY_NAMES + ("all",), default="all")
    parser.add_argument("--workdir", help="where to write the SQLite files (default: a temp dir)")
    args = parser.parse_args()

    if args.scripted:
        make_model = lambda s: ScriptedPlanner(s.script)
        mode = "HEADLINE BENCHMARK — scripted planner (deterministic)"
        scenarios, skipped = SCENARIOS, []
    else:
        model = make_ollama_model(TOOL_SCHEMAS, args.model)
        make_model = lambda s: model
        mode = f"VALIDATION PASS — real agent, ollama:{args.model}"
        # A benign model will not misbehave on cue; those premises stay scripted.
        scenarios = [s for s in SCENARIOS if not s.scripted_only]
        skipped = [s for s in SCENARIOS if s.scripted_only]
    policies = POLICY_NAMES if args.policy == "all" else (args.policy,)

    workdir = args.workdir or tempfile.mkdtemp(prefix="guardrail-run-")
    print(f"{mode}\nDatabases:    {workdir}\nScenarios:    {len(scenarios)}")
    for s in skipped:
        print(f"Skipped:      {s.key} (scripted only: its premise is agent misbehaviour)")

    summaries = {}
    harness_ok = True
    for policy in policies:
        print(f"\n=== Policy: {policy} ===")
        try:
            results = run_benchmark(scenarios, make_model, workdir, TOOL_TYPES, policy)
        except Exception as exc:  # e.g. Ollama not running
            print(f"ERROR — {type(exc).__name__}: {exc}")
            if not args.scripted:
                print("(is `ollama serve` running and the model pulled? or try --scripted)")
            return 1
        for r in results:
            print(r.line())
        rows = [r.scored() for r in results]
        overall, tools = summarize(rows), by_tool(rows)
        summaries[policy] = (overall, tools)
        print()
        print(format_table(overall.overall))
        print()
        _print_metrics({"overall": overall, **tools})
        _print_named(overall)
        _print_reliability(overall, results)
        # Wrong outcomes are results, not errors. The exit code only reports a
        # broken harness: outcome, tool state and ledger disagreeing.
        harness_ok &= all(not r.integrity for r in results)

    if len(summaries) > 1:
        print("\n=== Policy comparison ===")
        scopes = ["overall", *summaries[policies[0]][1]]
        print(f"  {'':<19}{'scope':<10}" + "".join(f"{p:<16}" for p in policies))
        for label, pick in METRICS:
            for scope in scopes:
                cells = []
                for p in policies:
                    overall, tools = summaries[p]
                    s = overall if scope == "overall" else tools[scope]
                    cells.append(format_rate(*pick(s)))
                print(f"  {label:<19}{scope:<10}" + "".join(f"{c:<16}" for c in cells))

    if "bounded" in summaries and "canonical" in summaries:
        print("\n=== False-alarm rate before / after canonicalization ===")
        columns = [p for p in ("naive", "bounded", "canonical") if p in summaries]
        heads = {"naive": "naive", "bounded": "before (strict)", "canonical": "after (canonical)"}
        print(f"  {'scope':<10}" + "".join(f"{heads[p]:<20}" for p in columns))
        for scope in ["overall", *summaries["bounded"][1]]:
            cells = []
            for p in columns:
                overall, tools = summaries[p]
                s = overall if scope == "overall" else tools[scope]
                cells.append(format_rate(s.overall.blocked_should_allow, s.overall.legitimate))
            print(f"  {scope:<10}" + "".join(f"{c:<20}" for c in cells))
        cleared = ({r.key for r in summaries["bounded"][0].false_alarms}
                   - {r.key for r in summaries["canonical"][0].false_alarms})
        flipped = ({r.key for r in summaries["canonical"][0].uncaught}
                   - {r.key for r in summaries["bounded"][0].uncaught})
        print(f"\n  False alarms cleared:            {', '.join(sorted(cleared)) or 'none'}")
        print(f"  Dangerous cases newly allowed:   {', '.join(sorted(flipped)) or 'none'}")
    return 0 if harness_ok else 1


def _print_metrics(columns: dict) -> None:
    print(f"  {'':<19}" + "".join(f"{name:<16}" for name in columns))
    for label, pick in METRICS:
        print(f"  {label:<19}" + "".join(f"{format_rate(*pick(s)):<16}" for s in columns.values()))


def _print_reliability(s: Summary, results) -> None:
    if s.invalid:
        causes = ", ".join(f"{cause} {n}" for cause, n in s.invalid)
        print(f"\n  Invalid runs by cause (excluded from the rates above): {causes}")
    unread = sum(1 for r in results if r.refusals.get("unread_target"))
    reworded = sum(1 for r in results if r.wording_differs)
    print(f"\n  Observations ({len(results)} runs):")
    print(f"    proposed before reading the target (refused, then retried): {unread}")
    print(f"    message wording differed from the script (accepted: matched on contact): {reworded}")


def _print_named(s: Summary) -> None:
    if s.uncaught:
        print("\n  Uncaught dangerous cases:")
        for r in s.uncaught:
            scope = "in scope" if r.in_scope else "outside the recheck mechanism"
            print(f"    - {r.shape} ({r.tool}, {r.key}) — {scope}")
    if s.false_alarms:
        print("\n  False alarms (legitimate cases blocked):")
        for r in s.false_alarms:
            print(f"    - {r.key} ({r.tool})")


if __name__ == "__main__":
    sys.exit(main())
