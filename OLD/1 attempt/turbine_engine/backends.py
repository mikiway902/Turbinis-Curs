"""Optional numerical backends for one bounded gas-property kernel.

SCOPE
=====
This module exposes NumPy and optional JAX ports of the existing scalar kernel
:func:`turbine_engine.gas.calc_gas_cp_k`.  That kernel is the hot gas-property
helper used by the stage path; it is *not* the cycle, the stage, or the engine.
The port preserves the source equation exactly and adds no model, clamp, or
constant.

The reference path is the existing scalar implementation, which in turn calls
:func:`turbine_engine.cycle.calc_cp_kerosene`::

    cp(T, alpha) = ((1.2 + 2.25 / alpha) / 1e5 * (T - 70) + 0.236) * 4187
                   for T < 700 K
    cp(T, alpha) = ((2.2 + 1.25 / alpha) / 1e5 * (T + 450) + 0.218) * 4187
                   for T >= 700 K

    dcp/dT = (cp(T + 1 K) - cp(T - 1 K)) / (2 * 1 K)
    cp_gas = cp(T) + (T - 273.15 K) * dcp/dT
    k      = cp_gas / (cp_gas - R)

The branch threshold, finite-difference step, Celsius offset, and ``4187``
calorie-to-joule factor are imported from :mod:`turbine_engine.gas` where
possible; :data:`CP_BRANCH_TEMPERATURE_K` is the literal branch value in
:func:`turbine_engine.cycle.calc_cp_kerosene`.  The implementation is a direct
transcription, not a refit.  Units are K, J/(kg*K), and dimensionless ``k``.

JAX IS OPTIONAL
===============
Importing this module never imports JAX.  JAX is imported only by
:func:`select_jax_device`, :func:`make_jax_gas_cp_k`, and the helpers that use
a returned kernel.  JAX CPU/GPU selection is always explicit: callers ask for
``"cpu"`` or ``"gpu"``, and an unavailable backend raises
:class:`BackendUnavailableError` instead of silently falling back to CPU.

These are small validation and benchmark helpers.  They do not implement a
batch engine, a full-cycle accelerator, an engine-design sweep, or an
automatic GPU speedup claim.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from turbine_engine.gas import (
    GAS_CELSIUS_OFFSET_K,
    GAS_FINITE_DIFFERENCE_STEP_K,
    calc_gas_cp_k,
)

KERNEL_NAME = "gas_cp_k"
KERNEL_DTYPE = "float64"

# Literal branch value in turbine_engine.cycle.calc_cp_kerosene.  The branch
# is ``T < 700.0`` (exactly 700 K uses the upper expression).
CP_BRANCH_TEMPERATURE_K = 700.0
CP_CALORIE_TO_JOULE = 4187.0


class BackendUnavailableError(RuntimeError):
    """A requested optional backend (for example JAX GPU) is unavailable."""


@dataclass(frozen=True)
class DeviceInfo:
    """Identity of one JAX device, recorded in validation output."""

    backend: str
    platform: str
    kind: str
    id: int


@dataclass(frozen=True)
class ArrayComparison:
    """Elementwise comparison of two finite real arrays."""

    element_count: int
    max_abs_error: float
    max_rel_error: float
    allclose: bool
    rtol: float
    atol: float


@dataclass(frozen=True)
class BenchmarkResult:
    """Timing of one kernel callable.

    ``first_call_seconds`` includes JAX tracing and compilation when the
    callable is a freshly jitted JAX kernel.  Warmup and measured sample times
    are recorded separately.  Throughput uses the median measured call time.
    """

    first_call_seconds: float
    warmup_seconds: tuple[float, ...]
    sample_seconds: tuple[float, ...]
    median_seconds: float
    min_seconds: float
    elements_per_call: int
    throughput_elements_per_second: float


@dataclass(frozen=True)
class JaxKernel:
    """A jitted JAX kernel and the explicit device used for its inputs."""

    function: Callable[[Any, Any, Any], tuple[Any, Any]]
    device: Any
    dtype: str

    def info(self) -> DeviceInfo:
        return describe_device(self.device)


def describe_device(device: Any) -> DeviceInfo:
    """Return stable, non-secret identity fields for one JAX device."""

    platform = str(getattr(device, "platform", "unknown"))
    kind = str(getattr(device, "device_kind", "unknown"))
    try:
        device_id = int(getattr(device, "id", -1))
    except (TypeError, ValueError):
        device_id = -1
    return DeviceInfo(
        backend=platform,
        platform=platform,
        kind=kind,
        id=device_id,
    )


def select_jax_device(backend: str, index: int = 0) -> Any:
    """Select a JAX device explicitly, never silently substituting CPU.

    ``backend`` is passed directly to :func:`jax.devices` (for example
    ``"cpu"`` or ``"gpu"``).  A missing backend or device raises
    :class:`BackendUnavailableError`.
    """

    if not isinstance(backend, str) or not backend:
        raise ValueError(f"backend must be a non-empty string, got {backend!r}")
    if index < 0:
        raise ValueError(f"index must be >= 0, got {index!r}")

    try:
        import jax
    except ModuleNotFoundError as exc:
        raise BackendUnavailableError(
            "JAX is not installed; install the optional backend dependencies"
        ) from exc

    try:
        devices = jax.devices(backend)
    except Exception as exc:  # JAX raises RuntimeError for an absent backend.
        raise BackendUnavailableError(
            f"jax.devices({backend!r}) failed with "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if not devices:
        raise BackendUnavailableError(
            f"jax.devices({backend!r}) returned no devices"
        )
    if index >= len(devices):
        raise BackendUnavailableError(
            f"jax.devices({backend!r}) returned {len(devices)} device(s); "
            f"index {index} is unavailable"
        )
    return devices[index]


def _broadcast_host_float64(
    temperatures: Any, alpha: Any, R: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Broadcast and validate the finite physical domain of the source kernel."""

    t = np.asarray(temperatures, dtype=np.float64)
    a = np.asarray(alpha, dtype=np.float64)
    r = np.asarray(R, dtype=np.float64)
    try:
        t, a, r = np.broadcast_arrays(t, a, r)
    except ValueError as exc:
        raise ValueError(
            "temperatures, alpha and R must be broadcastable arrays"
        ) from exc

    if np.any(~np.isfinite(t)) or np.any(t <= 0.0):
        raise ValueError("temperatures must be finite and > 0 K")
    if np.any(np.isnan(a)) or np.any(a <= 0.0):
        raise ValueError("alpha must be > 0 (math.inf is allowed)")
    if np.any(~np.isfinite(r)) or np.any(r <= 0.0):
        raise ValueError("R must be finite and > 0 J/(kg*K)")
    return t, a, r


def _cp_kerosene_numpy(t: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Vectorized literal transcription of ``calc_cp_kerosene``."""

    inverse_alpha = 1.0 / alpha
    lower = (
        (1.2 + 2.25 * inverse_alpha) / 1e5 * (t - 70.0) + 0.236
    ) * CP_CALORIE_TO_JOULE
    upper = (
        (2.2 + 1.25 * inverse_alpha) / 1e5 * (t + 450.0) + 0.218
    ) * CP_CALORIE_TO_JOULE
    return np.where(t < CP_BRANCH_TEMPERATURE_K, lower, upper)


def reference_gas_cp_k(T: float, alpha: float, R: float) -> tuple[float, float]:
    """Reference scalar kernel, delegating to the existing source implementation."""

    return calc_gas_cp_k(T, alpha, R)


def numpy_gas_cp_k(
    temperatures: Any, alpha: Any, R: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized float64 NumPy port of :func:`reference_gas_cp_k`."""

    t, a, r = _broadcast_host_float64(temperatures, alpha, R)
    c_pm = _cp_kerosene_numpy(t, a)
    dc_pm = (
        _cp_kerosene_numpy(t + GAS_FINITE_DIFFERENCE_STEP_K, a)
        - _cp_kerosene_numpy(t - GAS_FINITE_DIFFERENCE_STEP_K, a)
    ) / (2.0 * GAS_FINITE_DIFFERENCE_STEP_K)
    c_p = c_pm + (t - GAS_CELSIUS_OFFSET_K) * dc_pm
    denominator = c_p - r
    if np.any(~np.isfinite(c_p)) or np.any(denominator <= 0.0):
        raise ValueError(
            "invalid cp result: cp and cp - R must be finite, with cp - R > 0"
        )
    return c_p, c_p / denominator


def _cp_kerosene_jax(t: Any, alpha: Any, jnp: Any) -> Any:
    """JAX transcription of ``calc_cp_kerosene`` with the same 700 K branch."""

    inverse_alpha = 1.0 / alpha
    lower = (
        (1.2 + 2.25 * inverse_alpha) / 1e5 * (t - 70.0) + 0.236
    ) * CP_CALORIE_TO_JOULE
    upper = (
        (2.2 + 1.25 * inverse_alpha) / 1e5 * (t + 450.0) + 0.218
    ) * CP_CALORIE_TO_JOULE
    return jnp.where(t < CP_BRANCH_TEMPERATURE_K, lower, upper)


def make_jax_gas_cp_k(device: Any | None = None) -> JaxKernel:
    """Build a jitted float64 JAX kernel on an explicit device.

    Passing ``device=None`` selects the first device returned by
    :func:`jax.devices`, without changing its platform.  Callers that require
    GPU should pass ``select_jax_device("gpu")`` so an unavailable GPU raises
    instead of becoming a silent CPU run.
    """

    try:
        import jax
        import jax.numpy as jnp
    except ModuleNotFoundError as exc:
        raise BackendUnavailableError(
            "JAX is not installed; install the optional backend dependencies"
        ) from exc

    # The source kernel is double precision.  Set this before creating any JAX
    # arrays in this process.
    jax.config.update("jax_enable_x64", True)

    if device is None:
        devices = jax.devices()
        if not devices:
            raise BackendUnavailableError("JAX reported no devices")
        device = devices[0]
    if not hasattr(device, "platform"):
        raise TypeError(
            "device must be a JAX device from select_jax_device(); got "
            f"{type(device).__name__}"
        )

    def kernel_fn(t: Any, alpha: Any, r: Any) -> tuple[Any, Any]:
        c_pm = _cp_kerosene_jax(t, alpha, jnp)
        dc_pm = (
            _cp_kerosene_jax(t + GAS_FINITE_DIFFERENCE_STEP_K, alpha, jnp)
            - _cp_kerosene_jax(t - GAS_FINITE_DIFFERENCE_STEP_K, alpha, jnp)
        ) / (2.0 * GAS_FINITE_DIFFERENCE_STEP_K)
        c_p = c_pm + (t - GAS_CELSIUS_OFFSET_K) * dc_pm
        return c_p, c_p / (c_p - r)

    # Placement is explicit through prepare_jax_inputs/jax.device_put.  JAX
    # deprecated the jit(device=...) argument in favour of input placement.
    compiled = jax.jit(kernel_fn)
    return JaxKernel(
        function=compiled,
        device=device,
        dtype=str(jnp.dtype(jnp.float64)),
    )


def prepare_jax_inputs(
    kernel: JaxKernel, temperatures: Any, alpha: Any, R: Any
) -> tuple[Any, Any, Any]:
    """Validate host inputs and place float64 arrays on the kernel device."""

    try:
        import jax
        import jax.numpy as jnp
    except ModuleNotFoundError as exc:
        raise BackendUnavailableError(
            "JAX is not installed; install the optional backend dependencies"
        ) from exc

    t, a, r = _broadcast_host_float64(temperatures, alpha, R)
    return tuple(
        jax.device_put(jnp.asarray(value, dtype=jnp.float64), kernel.device)
        for value in (t, a, r)
    )


def block_until_ready(value: Any) -> None:
    """Block on every JAX leaf in ``value`` so timing includes device work."""

    import jax

    for leaf in jax.tree_util.tree_leaves(value):
        method = getattr(leaf, "block_until_ready", None)
        if method is not None:
            method()


def run_jax_gas_cp_k(
    kernel: JaxKernel, temperatures: Any, alpha: Any, R: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Execute one JAX kernel call on host inputs and return NumPy float64.

    The JAX output dtype is checked before conversion; a silent float32 result
    is treated as an error rather than accepted.
    """

    inputs = prepare_jax_inputs(kernel, temperatures, alpha, R)
    raw_cp, raw_k = kernel.function(*inputs)
    block_until_ready((raw_cp, raw_k))
    cp_dtype = str(raw_cp.dtype)
    k_dtype = str(raw_k.dtype)
    if cp_dtype != KERNEL_DTYPE or k_dtype != KERNEL_DTYPE:
        raise RuntimeError(
            "JAX kernel returned dtypes "
            f"{cp_dtype!r}/{k_dtype!r}; expected {KERNEL_DTYPE!r}"
        )
    cp = np.asarray(raw_cp, dtype=np.float64)
    k = np.asarray(raw_k, dtype=np.float64)
    r_host = np.asarray(R, dtype=np.float64)
    denominator = cp - r_host
    if (
        np.any(~np.isfinite(cp))
        or np.any(~np.isfinite(k))
        or np.any(denominator <= 0.0)
    ):
        raise RuntimeError(
            "JAX kernel returned invalid cp/k: cp and k must be finite and "
            "cp - R must be > 0"
        )
    return cp, k


def compare_arrays(
    reference: Any,
    actual: Any,
    *,
    rtol: float,
    atol: float,
) -> ArrayComparison:
    """Compare two same-shape finite arrays and report measured errors."""

    expected = np.asarray(reference, dtype=np.float64)
    observed = np.asarray(actual, dtype=np.float64)
    if expected.shape != observed.shape:
        raise ValueError(
            f"shape mismatch: reference {expected.shape}, actual {observed.shape}"
        )
    if np.any(~np.isfinite(expected)) or np.any(~np.isfinite(observed)):
        raise ValueError("comparison arrays must be finite")

    abs_error = np.abs(observed - expected)
    scale = np.maximum(np.abs(expected), np.finfo(np.float64).tiny)
    rel_error = abs_error / scale
    return ArrayComparison(
        element_count=int(expected.size),
        max_abs_error=float(np.max(abs_error)) if expected.size else 0.0,
        max_rel_error=float(np.max(rel_error)) if expected.size else 0.0,
        allclose=bool(np.allclose(observed, expected, rtol=rtol, atol=atol)),
        rtol=rtol,
        atol=atol,
    )


def benchmark_callable(
    function: Callable[[], Any],
    *,
    warmup_calls: int,
    samples: int,
    elements_per_call: int,
    synchronize: Callable[[Any], None] | None = None,
) -> BenchmarkResult:
    """Time one callable, with compilation/first call separate from samples.

    For a newly jitted JAX kernel the first call contains tracing and
    compilation.  ``synchronize`` should be :func:`block_until_ready` so each
    measured interval covers completed device work.
    """

    if not callable(function):
        raise TypeError("function must be callable")
    if warmup_calls < 0:
        raise ValueError(f"warmup_calls must be >= 0, got {warmup_calls!r}")
    if samples < 1:
        raise ValueError(f"samples must be >= 1, got {samples!r}")
    if elements_per_call <= 0:
        raise ValueError(
            f"elements_per_call must be > 0, got {elements_per_call!r}"
        )

    def timed_call() -> float:
        start = time.perf_counter()
        result = function()
        if synchronize is not None:
            synchronize(result)
        return time.perf_counter() - start

    first_call_seconds = timed_call()
    warmup_seconds = tuple(timed_call() for _ in range(warmup_calls))
    sample_seconds = tuple(timed_call() for _ in range(samples))
    median_seconds = float(statistics.median(sample_seconds))
    min_seconds = float(min(sample_seconds))
    return BenchmarkResult(
        first_call_seconds=float(first_call_seconds),
        warmup_seconds=warmup_seconds,
        sample_seconds=sample_seconds,
        median_seconds=median_seconds,
        min_seconds=min_seconds,
        elements_per_call=int(elements_per_call),
        throughput_elements_per_second=float(elements_per_call / median_seconds),
    )


_OOM_MARKERS = (
    "out of memory",
    "resource_exhausted",
    "cuda_error_out_of_memory",
    "cuda error: out of memory",
    "memory allocation failed",
)


def is_out_of_memory_error(exc: BaseException) -> bool:
    """Return True only for recognizable device/host memory-exhaustion errors."""

    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _OOM_MARKERS)


def run_chunked(
    function: Callable[..., Any],
    arrays: Sequence[Any],
    *,
    chunk_size: int,
) -> Any:
    """Run ``function(*chunk)`` in leading-axis chunks, halving on OOM only.

    The helper retries with half the current chunk size when, and only when,
    :func:`is_out_of_memory_error` recognizes the exception.  Any other error
    is re-raised unchanged.  Output arrays are concatenated without a dtype
    conversion, so an all-float32 result remains float32.
    """

    if not arrays:
        raise ValueError("arrays must contain at least one array")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size!r}")

    prepared = tuple(np.asarray(array) for array in arrays)
    length = prepared[0].shape[0] if prepared[0].ndim > 0 else 0
    if length <= 0:
        raise ValueError("arrays must have a positive leading dimension")
    for array in prepared:
        if array.ndim == 0 or array.shape[0] != length:
            raise ValueError(
                "every array must have the same positive leading dimension"
            )

    results: list[Any] = []
    start = 0
    size = chunk_size
    while start < length:
        end = min(start + size, length)
        chunk = tuple(array[start:end] for array in prepared)
        try:
            result = function(*chunk)
        except Exception as exc:
            if is_out_of_memory_error(exc) and (end - start) > 1:
                size = max(1, (end - start) // 2)
                continue
            raise
        results.append(result)
        start = end

    first = results[0]
    if isinstance(first, tuple):
        count = len(first)
        return tuple(
            np.concatenate(
                [np.asarray(result[index]) for result in results], axis=0
            )
            for index in range(count)
        )
    if isinstance(first, list):
        count = len(first)
        return [
            np.concatenate(
                [np.asarray(result[index]) for result in results], axis=0
            )
            for index in range(count)
        ]
    return np.concatenate([np.asarray(result) for result in results], axis=0)


__all__ = [
    "ArrayComparison",
    "BackendUnavailableError",
    "BenchmarkResult",
    "CP_BRANCH_TEMPERATURE_K",
    "CP_CALORIE_TO_JOULE",
    "DeviceInfo",
    "JaxKernel",
    "KERNEL_DTYPE",
    "KERNEL_NAME",
    "benchmark_callable",
    "block_until_ready",
    "compare_arrays",
    "describe_device",
    "is_out_of_memory_error",
    "make_jax_gas_cp_k",
    "numpy_gas_cp_k",
    "prepare_jax_inputs",
    "reference_gas_cp_k",
    "run_chunked",
    "run_jax_gas_cp_k",
    "select_jax_device",
]
