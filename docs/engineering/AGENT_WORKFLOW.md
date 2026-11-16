# OpenCode engineering workflow

## One-time provider connection

In OpenCode run `/connect` and connect:

1. **OpenAI** using **Sign in with ChatGPT** for `turbine-lead` and reviewers;
2. **DeepSeek** using its API key for implementation agents.

No credentials are stored in this repository. Confirm the configured models
with `/models`. The project uses the current catalog IDs
`openai/gpt-5.6-sol` and `deepseek/deepseek-flash`.

## Task execution

Start a governed task with:

```text
/turbine-change <task and acceptance requirement>
```

The lead is read-only. It prepares the formal specification, delegates code,
runs deterministic checks, invokes the read-only physics and numerics
reviewers, and accepts or rejects the result. Flash is the only implementation
model; coupled solvers and numerical work require stricter evidence rather
than a second implementation model.

## Failure loop

Implementation or verification failure returns to the implementing DeepSeek
agent with concrete evidence. A methodology conflict returns to the user; the
agent must not choose the closure that happens to satisfy a target.

## Mandatory separation

No single task may change a physical equation, its acceptance test, and its
reference result. Baseline updates require a separate user-approved task.
