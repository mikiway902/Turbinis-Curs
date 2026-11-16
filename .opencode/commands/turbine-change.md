---
description: Run the governed turbine engineering workflow
agent: turbine-lead
subagent: false
---

Handle this turbine task: $ARGUMENTS

Follow the project gate in `AGENTS.md`: identify equations and authority,
define units/limits/tolerances/regression evidence, delegate implementation to
the appropriate DeepSeek agent, run deterministic verification, request both
read-only reviewers for calculation changes, inspect the complete diff, and
return explicit PASS or FAIL. Preserve `book_approx`; do not update references
or promote EXPERIMENTAL physics without the user's separate approval.
