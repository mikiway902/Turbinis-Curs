# Optional numerical backends

This document covers the bounded CPU/GPU validation and benchmark added for the
gas-property kernel. It is **not** a new physical model and not an engine
performance claim.

## Scope

The validated kernel is the existing scalar helper
`turbine_engine.gas.calc_gas_cp_k`, which calls
`turbine_engine.cycle.calc_cp_kerosene` and applies the source finite difference:

```text
cp(T<700)  = ((1.2 + 2.25/alpha)/1e5 * (T - 70)  + 0.236) * 4187
cp(T>=700) = ((2.2 + 1.25/alpha)/1e5 * (T + 450) + 0.218) * 4187

dcp/dT = (cp(T + 1 K) - cp(T - 1 K)) / (2 * 1 K)
cp_gas = cp(T) + (T - 273.15 K) * dcp/dT
k      = cp_gas / (cp_gas - R)
```

`turbine_engine/backends.py` contains the reference delegate, a vectorized NumPy
float64 port, and an optional JAX float64 port. JAX is imported lazily. The
700 K branch, the 1 K finite-difference step, the 273.15 K offset, and the 4187
factor are preserved exactly.

The command validates at representative physical temperatures and at
`T = 699.999`, `700.0`, and `700.001` K for several excess-air ratios, including
dry air (`alpha = inf`). It then benchmarks only this kernel with a fixed batch,
separately measuring the first call (JAX tracing/compilation) from warmups and
steady samples. JAX timing uses `block_until_ready`.

## CPU validation

From the repository root with the project venv:

```powershell
.\.venv\Scripts\python.exe scripts\validate_backends.py
```

The default attempts CPU and GPU. If no GPU backend exists, its exact
`jax.devices("gpu")` error is recorded and CPU checks continue. The report is
written to `docs/backend_validation.json`.

To run CPU only:

```powershell
.\.venv\Scripts\python.exe scripts\validate_backends.py --devices cpu
```

## Optional WSL2 CUDA backend

The setup script is user-local: it uses no `sudo` or `apt`, does not modify
system Python, and refuses to overwrite an existing venv unless
`REUSE_EXISTING=1` is explicitly set. It requires Python >= 3.12 with the
standard `venv` module, the WSL NVIDIA driver plumbing (`nvidia-smi` and
`/usr/lib/wsl/lib/libcuda.so.1`), and network access for pip wheels.

Inside WSL, from the repository root:

```bash
bash scripts/setup_wsl_jax_cuda.sh
```

This creates `~/.venvs/turbinis-gpu` and installs
`jax[cuda12]==0.12.0.1` in user space. CUDA runtime libraries come from the
wheel extra; only the driver library comes from WSL. Then run:

```bash
~/.venvs/turbinis-gpu/bin/python scripts/validate_backends.py --devices cpu,gpu
```

The command lists all explicitly selected devices and records
`platform`, `device_kind`, `dtype`, parity errors, first-call/compile time, and
steady sample times. A device is labelled GPU only after
`jax.devices("gpu")` succeeds. There is no silent CPU fallback labelled GPU.
The validation process sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` only if the
caller did not set it, so this small kernel does not reserve most of an 8 GB
consumer GPU; the report records the actual value.

For a CLI check that requires the GPU to validate:

```bash
~/.venvs/turbinis-gpu/bin/python scripts/validate_backends.py \
    --devices cpu,gpu --require-gpu
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest turbine_engine.tests.test_backends -v
```

JAX CPU/GPU classes skip explicitly when that optional backend is unavailable;
the NumPy and lazy-import tests still run. The CLI test uses CPU only and writes
to a temporary file.

## Benchmark interpretation

The report compares median steady kernel time with NumPy on the same host and
batch. Ratios greater than one mean the JAX kernel took longer. This is a
single-kernel measurement only: it is **not** a full-cycle speedup, an engine
speedup, a design sweep, or evidence that GPU is universally faster. Compilation
and warmup are reported separately and are not hidden in the steady median.
