"""The full benchmark: every scenario across both tools."""

from scenarios import crm_benchmark, mailer_benchmark

# Sprint 2's co-designed set, kept separate so its results stay comparable.
CO_DESIGNED = list(crm_benchmark.SCENARIOS)
# Added in Sprint 3 with expected outcomes decided before running the policy.
NOT_CO_DESIGNED = (crm_benchmark.HARDER_SCENARIOS + mailer_benchmark.SCENARIOS
                   + mailer_benchmark.OUT_OF_MECHANISM)

SCENARIOS = CO_DESIGNED + NOT_CO_DESIGNED
BY_KEY = {s.key: s for s in SCENARIOS}
assert len(BY_KEY) == len(SCENARIOS), "scenario keys must be unique"
