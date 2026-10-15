# Sequence-aware agent guardrail (walking skeleton)

An agent can make a chain of tool calls where each call is allowed on its own,
but the chain as a whole acts on facts that changed after the plan was approved.
This guardrail runs just before an irreversible write. It re-reads the specific
facts that write depends on, compares them with the values recorded at approval,
and blocks the write if any of them changed. Every decision goes into an
append-only, hash-chained ledger.

This is the first slice: one mock CRM tool, one LangGraph agent, two scenarios.

## Run it

Requires Python 3.9+ (3.11+ recommended) and, for the LLM-driven run, [Ollama](https://ollama.com).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# LLM-driven run (local, no API keys)
ollama pull qwen2.5:7b-instruct      # fallback: llama3.1:8b, then --model llama3.1:8b
python run.py

# Same scenarios with a deterministic planner in place of the LLM
python run.py --scripted

# Unit tests (no Ollama needed)
pytest
```

Expected output:

```
Scenario A (stale edit): BLOCKED — PASS
Scenario B (unrelated change): ALLOWED — PASS
```

The SQLite files for each run are kept (the path is printed). You can inspect
the ledger with `sqlite3 <dir>/scenario_a_ledger.sqlite 'select event, payload from ledger'`.

## How it works

```
agent ⇄ act  →  approve  →  [interrupt]  →  recheck  →  effect | block
```

1. **agent / act.** The model reads the record and calls `propose_update`. It has
   no write tool. A proposal is only a record of what it intends to do. Malformed
   arguments get an error message back, and the model can retry up to 3 times.
2. **approve.** The guardrail looks up the edit's *dependency set*
   (`tools/crm_dependencies.py`), snapshots those fields, and logs `approval`.
3. **interrupt.** The graph pauses. In the scenarios, another process changes the
   record at this point.
4. **recheck.** The guardrail reads the record again and compares the dependency
   fields with the snapshot using strict equality (types must match too). It
   writes the verdict into graph state and logs `recheck`. The branch that
   follows looks only at that stored verdict, never at the live database, so
   replaying from a checkpoint always takes the same branch.
5. **effect.** `conditional_update` runs a single compare-and-set `UPDATE ... WHERE`
   that requires every dependency field to still match the snapshot. A change
   that lands between the recheck and the write is refused, and the guardrail
   logs `block` with `stage: conditional_update`.

| Module | Role |
|---|---|
| `guardrail/ledger.py` | Append-only, hash-chained ledger. SQLite triggers reject `UPDATE`/`DELETE`, and `verify_chain()` detects edits that get around the triggers. |
| `guardrail/recheck.py` | Tool-agnostic snapshot and strict-equality compare that returns a verdict. |
| `tools/mock_crm.py` | SQLite CRM: `read_record`, `conditional_update`, `mutate_out_of_band`. Knows nothing about the guardrail. |
| `tools/crm_dependencies.py` | Per-tool predicate: which fields each edit depends on, and which failure each one guards against. |
| `agent/graph.py` | LangGraph wiring, the Ollama model, and `ScriptedPlanner`. |
| `scenarios/` | Labelled scenarios (`dangerous` / `legitimate`) and the runner that checks outcome, CRM state and ledger. |

### Scenarios

- **A (dangerous):** an upgrade from starter to growth is approved while the account is `active`.
  Before the write, billing sets `status = cancelled`. **Expected: blocked**, the CRM is
  unchanged, and the ledger's `block` entry names `status`.
- **B (legitimate):** the same upgrade, but meanwhile someone edits `notes` and
  `last_contacted`. **Expected: allowed**, the plan changes, and the other edits are kept.

## Known limits of this slice

- The agent and the ledger run in the same Python process. The agent's tools are
  never given a ledger handle, but the operating system does not enforce that
  separation. Running the guardrail as its own process would.
- The approval snapshot comes from the guardrail's own read at proposal time, not
  from the copy the agent read earlier. In these scenarios the two reads happen
  back to back.
- There is only one tool and two scenarios. Catch-rate and false-alarm-rate
  metrics come later. `Scenario.label` is already in place for them.
