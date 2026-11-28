# Reference data policy

Files under `baselines/` are immutable regression inputs. Agents are denied
write access to this directory. A human-approved baseline change must be a
separate change from the physical formula and its test, and must explain the
before/after values and physical cause.

`book_approx_diagnostic` baselines detect software drift only. They are not
evidence that momentum, thermodynamics, losses, and integral flow are closed.
`published_cases/` is reserved for independently sourced validation cases with
edition/page/equation and applicability metadata.
