"""The full benchmark: every scenario across both tools."""

from scenarios import crm_benchmark, mailer_benchmark

# The original co-designed set, kept separate so its results stay comparable.
CO_DESIGNED = list(crm_benchmark.SCENARIOS)
# Added next, with expected outcomes decided before running the policy.
NOT_CO_DESIGNED = (crm_benchmark.HARDER_SCENARIOS + mailer_benchmark.SCENARIOS
                   + mailer_benchmark.SEQUENCE_CASES)
# Added for idempotency: one more retry case (CRM) and two
# legitimate repeats that idempotency must not suppress.
IDEMPOTENCY_ADDITIONS = crm_benchmark.IDEMPOTENCY_SCENARIOS + [mailer_benchmark.MAILER_RESEND_NEW_PLAN]

# Added from reported false-alarm shapes, expected outcomes decided before running.
REPORTED_ADDITIONS = list(crm_benchmark.REPORTED_SCENARIOS)

ORIGINAL_BENCHMARK = CO_DESIGNED + NOT_CO_DESIGNED   # the 23 before idempotency
IDEMPOTENCY_BENCHMARK = ORIGINAL_BENCHMARK + IDEMPOTENCY_ADDITIONS   # the 26 with idempotency
SCENARIOS = IDEMPOTENCY_BENCHMARK + REPORTED_ADDITIONS
BY_KEY = {s.key: s for s in SCENARIOS}
assert len(BY_KEY) == len(SCENARIOS), "scenario keys must be unique"
