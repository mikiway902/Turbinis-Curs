"""Deterministic sampling helpers without physical assumptions.

The helpers in this module generate input dictionaries only.  They do not
select architectures, impose engineering limits, call evaluators, or decide
which candidates are acceptable.  All ordering is deterministic and explicit.

* :func:`iter_grid` / :func:`deterministic_grid` build a Cartesian grid with
  axis keys sorted lexicographically and caller-supplied value order preserved.
* :func:`targeted_values` selects values near explicit numeric targets.
* :func:`split_pressure_ratio_log` splits a positive total pressure ratio into
  components in log space using explicitly supplied positive ratios.

No default search box, stage count, pressure-ratio distribution, or physical
constraint is encoded here.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Iterator, Mapping, Sequence

from turbine_engine.raw import copy_json_value


def _finite_number(name: str, value: Any) -> float:
    if type(value) is bool or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _materialize_axes(
    axes: Mapping[str, Sequence[Any]],
) -> tuple[tuple[str, ...], tuple[tuple[Any, ...], ...]]:
    if not isinstance(axes, Mapping):
        raise TypeError("axes must be a mapping of axis name to values")
    keys = tuple(sorted(axes))
    if not keys:
        raise ValueError("axes must contain at least one axis")
    values: list[tuple[Any, ...]] = []
    for key in keys:
        if type(key) is not str or not key:
            raise ValueError("axis names must be non-empty strings")
        raw_values = axes[key]
        if isinstance(raw_values, (str, bytes, bytearray)) or not isinstance(
            raw_values, Sequence
        ):
            raise ValueError(
                f"axis {key!r} values must be an ordered sequence of JSON values"
            )
        materialized = tuple(copy_json_value(value, path=f"$.axes.{key}") for value in raw_values)
        if not materialized:
            raise ValueError(f"axis {key!r} must contain at least one value")
        values.append(materialized)
    return keys, tuple(values)


def iter_grid(axes: Mapping[str, Sequence[Any]]) -> Iterator[dict[str, Any]]:
    """Yield deterministic Cartesian-grid input dictionaries.

    Axis keys are sorted; within each axis the supplied order is preserved.
    This changes only ordering, not the sampled set.
    """

    keys, value_sets = _materialize_axes(axes)
    for combination in itertools.product(*value_sets):
        yield dict(zip(keys, combination))


def deterministic_grid(
    axes: Mapping[str, Sequence[Any]],
    *,
    limit: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return a bounded, deterministic Cartesian grid.

    ``limit`` is an optional safety cap.  It truncates only after deterministic
    ordering, so the same prefix is reproduced every time.
    """

    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("limit must be a non-negative integer or None")
    if limit is None:
        return tuple(iter_grid(axes))
    return tuple(itertools.islice(iter_grid(axes), limit))


def targeted_values(
    values: Sequence[Any],
    targets: Sequence[Any],
    *,
    tolerance: Any,
) -> tuple[Any, ...]:
    """Select values within an absolute ``tolerance`` of any explicit target.

    Selection preserves the order of ``values``.  Numeric values only; bools
    and non-numbers are rejected.  This is a generic targeted-sampling helper
    (for example for selected total pressure ratios), not a physical rule.
    """

    if isinstance(values, (str, bytes, bytearray)) or not isinstance(
        values, Sequence
    ):
        raise TypeError("values must be an ordered sequence of numbers")
    if isinstance(targets, (str, bytes, bytearray)) or not isinstance(
        targets, Sequence
    ):
        raise TypeError("targets must be an ordered sequence of numbers")
    tolerance_value = _finite_number("tolerance", tolerance)
    if tolerance_value < 0:
        raise ValueError("tolerance must be >= 0")

    target_values = tuple(_finite_number("target", target) for target in targets)
    if not target_values:
        raise ValueError("targets must contain at least one value")

    selected: list[Any] = []
    seen: set[float] = set()
    for value in values:
        numeric = _finite_number("value", value)
        if numeric in seen:
            continue
        if any(abs(numeric - target) <= tolerance_value for target in target_values):
            selected.append(copy_json_value(value, path="$.value"))
            seen.add(numeric)
    return tuple(selected)


def split_pressure_ratio_log(
    total_pressure_ratio: Any, ratios: Sequence[Any]
) -> tuple[float, ...]:
    """Split a positive total into components in logarithmic space.

    The explicit ``ratios`` are relative log allocations:

    ``component_i = total ** (ratio_i / sum(ratios))``

    so ``prod(component_i) == total`` up to floating-point rounding.  No
    assumption is made that the total is greater than one or that a particular
    number of components is physically meaningful.
    """

    total = _finite_number("total_pressure_ratio", total_pressure_ratio)
    if total <= 0.0:
        raise ValueError("total_pressure_ratio must be > 0")
    if isinstance(ratios, (str, bytes, bytearray)) or not isinstance(
        ratios, Sequence
    ):
        raise TypeError("ratios must be an ordered non-empty sequence of positive numbers")
    ratio_values = tuple(_finite_number("ratio", ratio) for ratio in ratios)
    if not ratio_values:
        raise ValueError("ratios must be non-empty")
    if any(ratio <= 0.0 for ratio in ratio_values):
        raise ValueError("all ratios must be > 0")

    denominator = math.fsum(ratio_values)
    log_total = math.log(total)
    return tuple(
        math.exp(log_total * (ratio / denominator)) for ratio in ratio_values
    )


__all__ = [
    "iter_grid",
    "deterministic_grid",
    "targeted_values",
    "split_pressure_ratio_log",
]
