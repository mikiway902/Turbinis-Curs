"""Source-preserving const-hub three-stage orchestration.

PROVENANCE
==========
This module is a formula-preserving extraction of the three functions defined
before the top-level sweep in code cell ``const-hub-scan`` (cell 31) of
``Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb``:

* ``ступень_const_hub(T0, p0, H0, G_c, G_л, n, α1, κ, D_hub, D0)``
* ``турбина_const_hub(n, D_hub, scale_pi)``
* ``распределить_π_const_hub(n, D_hub, π0)``

The cell's seven-point ``D_hub`` sweep after ``rows = []`` is intentionally
*not* part of this module.  No notebook is executed and no scan, search, warm
start, or optimisation is performed here.

The notebook definitions remain the independent legacy comparison path.  Tests
compare the two only as software-drift diagnostics, never as physical
validation or as an acceptable-design claim.

SOURCE COMPATIBILITY
====================
The source functions resolve their inputs as globals of the notebook namespace.
The extracted orchestration keeps that live binding: every dependency is looked
up in the supplied namespace at call time, so assigning ``π_веса``, ``α1_ст``,
``κ_ст``, ``G_c1`` or ``ступень`` between calls is observed exactly as the
notebook functions observe it.  A replacement of the stage function or of the
``H0_по_p`` helper after construction is therefore not cached.

The extracted unit can also resolve its sibling functions live after
:meth:`ConstHubOrchestration.install` has bound all three into the namespace,
so replacing ``турбина_const_hub`` or ``ступень_const_hub`` is observed exactly
as the notebook globals would observe it.  Before installation the object calls
its own methods, which keeps direct use of the dependency-injected API
self-contained.  The three functions are installed together as a unit; a sibling
that is intentionally replaced afterwards keeps the notebook's live-lookup
behaviour and owns its own contract.

Arithmetic, expression order, iteration limits (60 and 30), tolerances
(``1e-7``, ``1e-5``, ``1e-12``), the half-relaxation step, ``rho = 0.3``,
``D0 = D_hub + 0.02``, the interstage pressure correction, and the last-stage
``p_т`` outlet are preserved verbatim.  Invalid evaluated states still raise the
namespace's own ``НетРешения``; silent exhaustion of the stage reaction fixed
point or the pressure-ratio secant, and a secant denominator stall without a
converged residual, now raise
:class:`turbine_engine.diagnostics.NumericalConvergenceError` with the stage,
reason and last finite iterate instead of returning ``None`` or a stale value.
No physical equation or convergence tolerance is changed.

PRESSURE-RATIO POLICY
=====================
The default ``pressure_ratio_policy="legacy_clipped"`` keeps the source
``[1.05, 1.2 * π_ст_макс]`` secant clip and is arithmetically unchanged.
``pressure_ratio_policy="raw_unbounded"`` is an explicit opt-in diagnostic
policy: it performs the same finite secant arithmetic without calling the clip
or reading ``π_ст_макс``, and rejects only the mathematical p-expansion domain
``y <= 1``.  It is not a physical validation and does not relax any residual,
iteration or reaction tolerance.

NO THIRD-PARTY IMPORTS
======================
The module itself uses only the standard library.  Interpolation and clipping
are explicit callables; :func:`linear_interp` and :func:`clip_scalar` are
source-compatible defaults.  The optimizer supplies its fast ``_линейно``
interpolant and ``numpy.clip`` so that the default wiring is arithmetic-
identical to the notebook cell.

UNITS
=====
SI throughout; ``n`` is rpm, angles are degrees, diameters and blade lengths
are metres, mass flows are kg/s, heat drops are J/kg, and pressures are Pa.
The returned stage objects are the legacy dictionaries already produced by the
injected ``ступень`` function.
"""

from __future__ import annotations

import bisect
import math
import numbers
from typing import Any, Callable, Mapping, Sequence

from turbine_engine.diagnostics import NumericalConvergenceError

# Source-fixed numerical controls.  Do not tune without changing the notebook.
STAGE_ITERATION_LIMIT = 60
REACTION_TOLERANCE = 1e-7
PI_SECANT_ITERATION_LIMIT = 30
PI_RESIDUAL_TOLERANCE = 1e-5
PI_SECANT_TOLERANCE = 1e-12
PI_CLIP_LOWER = 1.05
PI_CLIP_UPPER_FACTOR = 1.2
INITIAL_REACTION = 0.3
INLET_DIAMETER_OFFSET = 0.02
INITIAL_PI_STEP_FACTOR = 1.02

#: Pressure-ratio secant policies.  The default reproduces the notebook clip
#: exactly; ``raw_unbounded`` keeps the same secant arithmetic but enforces only
#: the mathematical p-expansion domain ``y > 1``.
PRESSURE_RATIO_LEGACY_CLIPPED = "legacy_clipped"
PRESSURE_RATIO_RAW_UNBOUNDED = "raw_unbounded"
PRESSURE_RATIO_POLICIES = (
    PRESSURE_RATIO_LEGACY_CLIPPED,
    PRESSURE_RATIO_RAW_UNBOUNDED,
)


def validate_pressure_ratio_policy(policy: str) -> str:
    """Return a validated pressure-ratio policy name."""

    if policy not in PRESSURE_RATIO_POLICIES:
        raise ValueError(
            f"pressure_ratio_policy must be one of {PRESSURE_RATIO_POLICIES!r}, "
            f"got {policy!r}"
        )
    return policy


StageFunction = Callable[..., Mapping[str, float]]
H0Function = Callable[[float, float, float], float]
InterpFunction = Callable[[float, Sequence[float], Sequence[float]], float]
ClipFunction = Callable[[float, float, float], float]

#: Namespace entries required by the extracted orchestration.
REQUIRED_NAMESPACE_KEYS: tuple[str, ...] = (
    "ступень",
    "H0_по_p",
    "ρ_таблица",
    "z",
    "T_г_полн",
    "p_г_полн",
    "G_c1",
    "G_л_ост",
    "μ",
    "p_т",
    "π_веса",
    "α1_ст",
    "κ_ст",
    "НетРешения",
    "π_ст_макс",
)


def linear_interp(
    x: float, xp: Sequence[float], fp: Sequence[float]
) -> float:
    """One-point linear interpolation matching the notebook's fast path.

    This reproduces ``optimize_extended._линейно`` operation-for-operation for
    finite one-dimensional input.  NumPy-only features (periodic interpolation,
    complex inputs, empty tables) are deliberately not provided.
    """

    if x <= xp[0]:
        return fp[0]
    if x >= xp[-1]:
        return fp[-1]
    i = bisect.bisect_right(xp, x) - 1
    return (fp[i + 1] - fp[i]) / (xp[i + 1] - xp[i]) * (x - xp[i]) + fp[i]


def clip_scalar(value: float, lower: float, upper: float) -> float:
    """Scalar clamp with ``numpy.clip`` semantics for finite values.

    ``NaN`` is returned unchanged because both comparisons are false, matching
    ``numpy.clip`` for a scalar.  The source still applies ``float(...)`` to the
    result in the secant solver.
    """

    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def _pos_float(name: str, value: Any) -> float:
    """Return a finite positive float, rejecting booleans and strings."""

    if isinstance(value, bool) or isinstance(value, str):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number, got {value!r}") from exc
    if not (math.isfinite(result) and result > 0.0):
        raise ValueError(f"{name} must be finite and > 0, got {result!r}")
    return result


def _pos_int(name: str, value: Any) -> int:
    """Return a positive integer, rejecting booleans and non-integral values."""

    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


class ConstHubOrchestration:
    """Const-hub stage, turbine, and pressure-distribution functions.

    Parameters
    ----------
    namespace:
        Live mapping containing the notebook dependencies listed in
        :data:`REQUIRED_NAMESPACE_KEYS`.  Entries are resolved on every use.
    interp:
        Interpolation callable with the ``numpy.interp`` positional signature.
        Defaults to :func:`linear_interp`.
    clip:
        Scalar clip callable ``(value, lower, upper)``.  Defaults to
        :func:`clip_scalar`.
    pressure_ratio_policy:
        ``"legacy_clipped"`` (default) reproduces the notebook's
        ``[1.05, 1.2 * π_ст_макс]`` secant clip exactly.
        ``"raw_unbounded"`` keeps the same secant arithmetic but applies no
        engineering clip and does not read ``π_ст_макс``; only the mathematical
        p-expansion domain ``y > 1`` is enforced.

    The object stores no computed state.  Calls are deterministic for the
    current namespace values, just like the notebook globals.
    """

    def __init__(
        self,
        namespace: Mapping[str, Any],
        *,
        interp: InterpFunction | None = None,
        clip: ClipFunction | None = None,
        pressure_ratio_policy: str = PRESSURE_RATIO_LEGACY_CLIPPED,
    ) -> None:
        self._namespace = namespace
        self._interp = interp if interp is not None else linear_interp
        self._clip = clip if clip is not None else clip_scalar
        self._pressure_ratio_policy = validate_pressure_ratio_policy(
            pressure_ratio_policy
        )
        self._installed = False

    @property
    def namespace(self) -> Mapping[str, Any]:
        """The live dependency mapping supplied at construction."""

        return self._namespace

    @property
    def pressure_ratio_policy(self) -> str:
        """The selected pressure-ratio secant policy."""

        return self._pressure_ratio_policy

    def missing_namespace_keys(self) -> tuple[str, ...]:
        """Return required notebook dependencies absent from the namespace.

        ``raw_unbounded`` does not read ``π_ст_макс``, so that key is not
        required by the raw policy.
        """

        keys = REQUIRED_NAMESPACE_KEYS
        if self._pressure_ratio_policy == PRESSURE_RATIO_RAW_UNBOUNDED:
            keys = tuple(key for key in keys if key != "π_ст_макс")
        return tuple(key for key in keys if key not in self._namespace)

    def install(self) -> None:
        """Bind the three extracted functions into the supplied namespace.

        ``build_ns`` uses this to replace exactly the notebook definitions from
        the cell while keeping the rest of the executed cell (notably the
        ``ρ_таблица`` data and the excluded sweep) untouched.  After a
        successful installation the extracted functions also resolve their
        sibling definitions through the namespace, matching the notebook's
        global-name lookup.
        """

        target = self._namespace
        if not hasattr(target, "__setitem__"):
            raise TypeError("namespace must be a mutable mapping")
        target["ступень_const_hub"] = self.ступень_const_hub  # type: ignore[index]
        target["турбина_const_hub"] = self.турбина_const_hub  # type: ignore[index]
        target["распределить_π_const_hub"] = self.распределить_π_const_hub  # type: ignore[index]
        self._installed = True

    def _sibling(self, name: str, fallback: Callable[..., Any]) -> Callable[..., Any]:
        """Resolve a sibling function live after installation."""

        if self._installed:
            return self._namespace[name]
        return fallback

    def _value(self, name: str) -> Any:
        return self._namespace[name]

    def _policy_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Attach the policy only to raw diagnostics, keeping legacy identical."""

        if self._pressure_ratio_policy == PRESSURE_RATIO_RAW_UNBOUNDED:
            return {**state, "pressure_ratio_policy": self._pressure_ratio_policy}
        return dict(state)

    def _raw_pi_candidate(
        self, proposed: float, last_valid: Mapping[str, Any]
    ) -> float:
        """Apply only the mathematical p-expansion domain to a raw proposal.

        Above one the secant proposal is returned unchanged.  A proposal at or
        below one is moved halfway towards the last valid iterate and the
        positive endpoint, never to an application-defined engineering floor.
        A non-finite proposal, or a safeguard that cannot stay above one, is a
        typed numerical failure rather than a clipped value.
        """

        if not math.isfinite(proposed):
            raise NumericalConvergenceError(
                "raw pressure-ratio secant proposal is not finite",
                stage="orchestration.распределить_π_const_hub",
                reason="non_finite_proposed_pressure_ratio",
                last_valid_state=self._policy_state(
                    {**last_valid, "proposed_y": proposed}
                ),
            )
        if proposed > 1.0:
            return float(proposed)
        last_y = float(last_valid["y"])
        safeguard = 0.5 * (last_y + 1.0)
        if not (math.isfinite(safeguard) and safeguard > 1.0):
            raise NumericalConvergenceError(
                "raw pressure-ratio secant cannot keep the p-expansion domain "
                "y > 1",
                stage="orchestration.распределить_π_const_hub",
                reason="pi_expansion_domain_lost",
                last_valid_state=self._policy_state(
                    {**last_valid, "proposed_y": proposed}
                ),
            )
        return float(safeguard)

    def ступень_const_hub(
        self,
        T0: float,
        p0: float,
        H0: float,
        G_c: float,
        G_л: float,
        n: float,
        α1: float,
        κ: float,
        D_hub: float,
        D0: float,
    ) -> dict[str, float]:
        """Close one stage with a constant hub and a reaction-degree iteration.

        Operation order follows the source exactly: the stage is evaluated at
        the current ``(rho, D_c, D_л)``, the annulus areas become diameters, the
        density table gives the next reaction degree, and the update is
        half-relaxed.  The convergence test and the final stage evaluation are
        unchanged.  A missing ``ступень`` solution propagates as the namespace's
        own ``НетРешения`` exception.
        """

        D_c = D_л = D0
        rho = INITIAL_REACTION
        D_c_n, D_л_n, rho_n = D_c, D_л, rho
        D_c_eval, D_л_eval, rho_eval = D_c, D_л, rho
        D_c_next, D_л_next, rho_next = D_c, D_л, rho
        converged = False
        for _ in range(STAGE_ITERATION_LIMIT):
            D_c_eval, D_л_eval, rho_eval = D_c, D_л, rho
            r = self._value("ступень")(
                T0, p0, H0, rho, D_c, D_л, n, G_c, G_л, α1, κ
            )
            A_c = G_c * r["v1"] / r["c1a"]
            A_l = G_л * r["v2"] / r["c2a"]
            l_c = (math.sqrt(D_hub**2 + 4 * A_c / math.pi) - D_hub) / 2
            l_l = (math.sqrt(D_hub**2 + 4 * A_l / math.pi) - D_hub) / 2
            D_c_n, D_л_n = D_hub + l_c, D_hub + l_l
            rho_n = float(
                self._interp(l_l / D_л_n, *self._value("ρ_таблица"))
            )
            if (
                abs(D_c_n - D_c_eval) < REACTION_TOLERANCE
                and abs(D_л_n - D_л_eval) < REACTION_TOLERANCE
                and abs(rho_n - rho_eval) < REACTION_TOLERANCE
            ):
                D_c, D_л, rho = D_c_n, D_л_n, rho_n
                converged = True
                break
            D_c, D_л, rho = (
                0.5 * (D_c_eval + D_c_n),
                0.5 * (D_л_eval + D_л_n),
                0.5 * (rho_eval + rho_n),
            )
            D_c_next, D_л_next, rho_next = D_c, D_л, rho
        if not converged:
            raise NumericalConvergenceError(
                "const-hub reaction fixed point exhausted its iteration limit",
                stage="orchestration.ступень_const_hub",
                reason="reaction_fixed_point_exhausted",
                last_valid_state={
                    "D_c": D_c_eval,
                    "D_l": D_л_eval,
                    "rho": rho_eval,
                    "D_c_delta": D_c_n - D_c_eval,
                    "D_l_delta": D_л_n - D_л_eval,
                    "rho_delta": rho_n - rho_eval,
                    "D_c_target": D_c_n,
                    "D_l_target": D_л_n,
                    "rho_target": rho_n,
                    "D_c_next": D_c_next,
                    "D_l_next": D_л_next,
                    "rho_next": rho_next,
                    "iteration_limit": STAGE_ITERATION_LIMIT,
                    "tolerance": REACTION_TOLERANCE,
                },
            )
        return self._value("ступень")(
            T0, p0, H0, rho, D_c, D_л, n, G_c, G_л, α1, κ
        )

    def турбина_const_hub(
        self, n: float, D_hub: float, scale_pi: float
    ) -> list[dict[str, float]]:
        """Build the three-stage sequence at a constant hub diameter.

        The first stage uses the inlet total state and ``G_c1`` for both flows;
        later stages use the previous stage outlet and ``G_л_ост``.  The
        interstage total pressure is reconstructed from ``p2`` and the rotor
        exit velocity exactly as in the source.  Stage ``i < z`` receives its
        pressure ratio from ``scale_pi`` and ``π_веса``; the last stage uses the
        fixed outlet pressure ``p_т``.
        """

        res: list[dict[str, float]] = []
        for i in range(1, self._value("z") + 1):
            if i == 1:
                T0 = self._value("T_г_полн")
                p0 = self._value("p_г_полн")
                G_c = self._value("G_c1")
                G_л = self._value("G_c1")
            else:
                пр = res[-1]
                T0 = пр["T_ст_п"]
                p0 = пр["p2"] * (
                    1
                    + (self._value("μ") * пр["c2"]) ** 2
                    / (2 * пр["cp"] * пр["T_ст"])
                ) ** (пр["k"] / (пр["k"] - 1))
                G_л_ост = self._value("G_л_ост")
                G_c = G_л = G_л_ост
            if i < self._value("z"):
                H0 = self._value("H0_по_p")(
                    T0,
                    p0,
                    p0
                    / scale_pi
                    ** (
                        self._value("π_веса")[i - 1]
                        / self._value("π_веса")[-1]
                    ),
                )
            else:
                H0 = self._value("H0_по_p")(T0, p0, self._value("p_т"))
            stage_const_hub = self._sibling(
                "ступень_const_hub", self.ступень_const_hub
            )
            res.append(
                stage_const_hub(
                    T0,
                    p0,
                    H0,
                    G_c,
                    G_л,
                    n,
                    self._value("α1_ст")[i - 1],
                    self._value("κ_ст")[i - 1],
                    D_hub,
                    D_hub + INLET_DIAMETER_OFFSET,
                )
            )
        return res

    def распределить_π_const_hub(
        self, n: float, D_hub: float, π0: float
    ) -> tuple[float, list[dict[str, float]] | None]:
        """Solve the common stage pressure ratio by a secant method.

        This reproduces the source secant loop, including the bisection fallback
        when ``турбина_const_hub`` raises the namespace's ``НетРешения``, the
        residual and denominator stops, and (for the default
        ``legacy_clipped`` policy) the ``[1.05, 1.2 * π_ст_макс]`` clip.
        ``raw_unbounded`` keeps the same finite-secant arithmetic but does not
        call the clip or read ``π_ст_макс``; it enforces only ``y > 1`` and
        safeguards a proposal at or below one halfway towards the last valid
        iterate and one.  The source's silent ``(None, None)`` return and its
        ``None`` stage list are replaced by explicit diagnostics: an invalid
        initial evaluation raises the namespace's ``НетРешения``, while a
        stalled or exhausted secant, a lost mathematical domain, or a final
        iteration without a valid evaluation, raises
        :class:`turbine_engine.diagnostics.NumericalConvergenceError`.
        A small denominator is never treated as convergence unless the residual
        itself is already below the existing tolerance.
        """

        def f(y: float) -> tuple[float | None, list[dict[str, float]] | None]:
            if (
                self._pressure_ratio_policy == PRESSURE_RATIO_RAW_UNBOUNDED
                and not (math.isfinite(y) and y > 1.0)
            ):
                raise NumericalConvergenceError(
                    "raw pressure ratio is outside the p-expansion domain y > 1",
                    stage="orchestration.распределить_π_const_hub",
                    reason="pi_expansion_domain_lost",
                    last_valid_state=self._policy_state({"y": float(y)}),
                )
            turbine_const_hub = self._sibling(
                "турбина_const_hub", self.турбина_const_hub
            )
            try:
                ст = turbine_const_hub(n, D_hub, y)
            except self._value("НетРешения"):
                return None, None
            return ст[-1]["π_ст"] - y, ст

        y_a, y_b = π0, π0 * INITIAL_PI_STEP_FACTOR
        f_a, ст = f(y_a)
        if f_a is None:
            raise self._value("НетРешения")(
                "нет решения при начальной π ступени"
            )
        if not math.isfinite(f_a):
            raise NumericalConvergenceError(
                "initial secant residual is not finite",
                stage="orchestration.распределить_π_const_hub",
                reason="non_finite_initial_residual",
                last_valid_state=self._policy_state(
                    {"input_y": float(y_a), "initial_residual": f_a}
                ),
            )
        converged = False
        stalled = False
        f_b: float | None = None
        ст_b: list[dict[str, float]] | None = None
        last_valid: dict[str, Any] = {
            "y": float(y_a),
            "residual": float(f_a),
        }
        for _ in range(PI_SECANT_ITERATION_LIMIT):
            f_b, ст_b = f(y_b)
            if f_b is None:
                y_b = 0.5 * (y_a + y_b)
                continue
            if not math.isfinite(f_b):
                raise NumericalConvergenceError(
                    "secant residual is not finite at the evaluated stage "
                    "state",
                    stage="orchestration.распределить_π_const_hub",
                    reason="non_finite_residual",
                    last_valid_state=self._policy_state(
                        {**last_valid, "y_b": y_b}
                    ),
                )
            last_valid = {"y": float(y_b), "residual": float(f_b)}
            if abs(f_b) < PI_RESIDUAL_TOLERANCE:
                converged = True
                break
            den = f_b - f_a
            if not math.isfinite(den):
                raise NumericalConvergenceError(
                    "secant denominator is not finite",
                    stage="orchestration.распределить_π_const_hub",
                    reason="non_finite_secant_denominator",
                    last_valid_state=self._policy_state(last_valid),
                )
            if abs(den) < PI_SECANT_TOLERANCE:
                stalled = True
                break
            y_a, y_b, f_a = y_b, y_b - f_b * (y_b - y_a) / den, f_b
            if self._pressure_ratio_policy == PRESSURE_RATIO_LEGACY_CLIPPED:
                y_b = float(
                    self._clip(
                        y_b,
                        PI_CLIP_LOWER,
                        self._value("π_ст_макс") * PI_CLIP_UPPER_FACTOR,
                    )
                )
            else:
                y_b = self._raw_pi_candidate(y_b, last_valid)
        if not converged:
            if f_b is None or ст_b is None:
                raise NumericalConvergenceError(
                    "pressure-ratio secant ended without a valid stage "
                    "evaluation",
                    stage="orchestration.распределить_π_const_hub",
                    reason="no_valid_final_evaluation",
                    last_valid_state=self._policy_state({
                        **last_valid,
                        "iteration_limit": PI_SECANT_ITERATION_LIMIT,
                        "residual_tolerance": PI_RESIDUAL_TOLERANCE,
                    }),
                )
            raise NumericalConvergenceError(
                (
                    "pressure-ratio secant denominator stalled "
                    "without convergence"
                    if stalled
                    else "pressure-ratio secant exhausted its iteration limit"
                ),
                stage="orchestration.распределить_π_const_hub",
                reason=(
                    "secant_denominator_stalled"
                    if stalled
                    else "pi_secant_iteration_exhausted"
                ),
                last_valid_state=self._policy_state({
                    **last_valid,
                    "iteration_limit": PI_SECANT_ITERATION_LIMIT,
                    "residual_tolerance": PI_RESIDUAL_TOLERANCE,
                    "secant_tolerance": PI_SECANT_TOLERANCE,
                }),
            )
        return y_b, ст_b

    def _static_pressure_metric(
        self, p_static: float, n: float, D_hub: float, z: int,
        p_inlet_total: float, metric_kind: str,
        mechanical_efficiency: float | None, stage: str,
    ) -> tuple[float, list[dict[str, float]]]:
        """Evaluate the actual meanline metric at one outlet static pressure."""

        self._namespace["p_т"] = p_static
        _, stages = self.распределить_π_const_hub(
            n, D_hub, (p_inlet_total / p_static) ** (1.0 / z)
        )
        if not stages:
            raise NumericalConvergenceError(
                "static-pressure closure returned no stage evaluation",
                stage=stage,
                reason="no_stage_evaluation",
                last_valid_state={
                    "trial_static_pressure_Pa": p_static,
                    "target_kind": metric_kind,
                },
            )
        if metric_kind == "total_outlet_pressure":
            return float(stages[-1]["p_ст_п"]), stages
        power = mechanical_efficiency * sum(
            float(s["G_c"]) * float(s["L_т"]) for s in stages
        )
        return power, stages

    def _solve_static_pressure_closure(
        self, *, n: float, D_hub: float, target: float,
        relative_tol: float, max_iter: int,
        static_pressure_bracket: Sequence[float],
        metric_kind: str, stage: str,
        mechanical_efficiency: float | None = None,
    ) -> dict[str, Any]:
        """Bracketed bisection over the existing meanline outlet pressure."""

        if self._pressure_ratio_policy != PRESSURE_RATIO_RAW_UNBOUNDED:
            raise ValueError(
                f"{stage} requires {PRESSURE_RATIO_RAW_UNBOUNDED!r} pressure "
                f"policy, got {self._pressure_ratio_policy!r}"
            )
        if not hasattr(self._namespace, "__setitem__"):
            raise TypeError("namespace must support item assignment")
        n = _pos_float("n", n)
        D_hub = _pos_float("D_hub", D_hub)
        target = _pos_float("target", target)
        relative_tol = _pos_float("relative_tol", relative_tol)
        max_iter = _pos_int("max_iter", max_iter)
        if mechanical_efficiency is not None:
            mechanical_efficiency = _pos_float(
                "mechanical_efficiency", mechanical_efficiency)
            if mechanical_efficiency > 1.0:
                raise ValueError("mechanical_efficiency must be <= 1")
        p_in = _pos_float("namespace p_г_полн", self._value("p_г_полн"))
        z = _pos_int("namespace z", self._value("z"))
        try:
            bracket = tuple(static_pressure_bracket)
        except TypeError as exc:
            raise ValueError("static_pressure_bracket must be a pair") from exc
        if len(bracket) != 2:
            raise ValueError(
                f"static_pressure_bracket must be a pair, got {len(bracket)}"
            )
        lo = _pos_float("static_pressure_bracket[0]", bracket[0])
        hi = _pos_float("static_pressure_bracket[1]", bracket[1])
        if not (lo < hi < p_in):
            raise ValueError(
                "static_pressure_bracket must satisfy 0 < lo < hi < "
                f"p_г_полн={p_in!r}, got ({lo!r}, {hi!r})"
            )

        original_p_t = self._namespace["p_т"]
        evaluations = 0
        last: dict[str, Any] = {"static_pressure_Pa": hi,
                                "metric": None, "residual": None}

        def evaluate(p_static: float) -> tuple[float, list[dict[str, float]]]:
            nonlocal evaluations, last
            metric, stages = self._static_pressure_metric(
                p_static, n, D_hub, z, p_in, metric_kind,
                mechanical_efficiency, stage)
            evaluations += 1
            residual = metric - target
            last = {"static_pressure_Pa": p_static, "metric": metric,
                    "residual": residual, "signed_residual": residual}
            if not math.isfinite(metric):
                raise NumericalConvergenceError(
                    "static-pressure closure metric is not finite",
                    stage=stage,
                    reason=("non_finite_initial_metric" if evaluations <= 2
                            else "non_finite_metric"),
                    last_valid_state={
                        "target_kind": metric_kind, "target": target,
                        "static_pressure_bracket": [lo, hi],
                        "evaluations": evaluations, **last})
            return metric, stages

        def finish(
            p_static: float, metric: float,
            stages: list[dict[str, float]], iterations: int,
        ) -> dict[str, Any]:
            residual = metric - target
            controls: dict[str, Any] = {
                "n_rpm": n, "D_hub_m": D_hub,
                "relative_tol": relative_tol, "max_iter": max_iter,
                "static_pressure_bracket_Pa": [lo, hi]}
            if mechanical_efficiency is not None:
                controls["mechanical_efficiency"] = mechanical_efficiency
            record: dict[str, Any] = {
                "target_kind": metric_kind, "target_value": target,
                "actual_metric": metric,
                "relative_residual": abs(residual) / abs(target),
                "signed_residual": residual,
                "solved_static_pressure_Pa": p_static,
                "actual_outlet_total_pressure_Pa": float(
                    stages[-1]["p_ст_п"]),
                "actual_stages": stages, "iterations": iterations,
                "iteration_count": iterations, "evaluations": evaluations,
                "pressure_ratio_policy": self._pressure_ratio_policy,
                "controls": controls,
                "source_point": {
                    "inlet_total_pressure_Pa": p_in, "stage_count": z,
                    "last_stage_total_pressure_key": "p_ст_п",
                    "last_stage_total_temperature_key": "T_ст_п"},
                "physical_validation": "NOT_PERFORMED",
                "stage_count_selection": "NONE"}
            if metric_kind == "mechanical_power":
                record["mechanical_power_W"] = metric
            return record

        def fail(reason: str, message: str) -> NumericalConvergenceError:
            return NumericalConvergenceError(
                message, stage=stage, reason=reason,
                last_valid_state={
                    "target_kind": metric_kind, "target": target,
                    "static_pressure_bracket": [lo, hi],
                    "iterations": iterations, "evaluations": evaluations,
                    "last_evaluated": dict(last)})

        try:
            m_lo, s_lo = evaluate(lo)
            m_hi, s_hi = evaluate(hi)
            r_lo, r_hi = m_lo - target, m_hi - target
            tolerance_product = relative_tol * abs(target)
            if abs(r_lo) <= tolerance_product:
                return finish(lo, m_lo, s_lo, 0)
            if abs(r_hi) <= tolerance_product:
                return finish(hi, m_hi, s_hi, 0)
            if (r_lo < 0.0) == (r_hi < 0.0):
                raise NumericalConvergenceError(
                    "static-pressure bracket endpoints do not bracket the "
                    "prescribed metric", stage=stage,
                    reason="boundary_not_bracketed",
                    last_valid_state={
                        "target_kind": metric_kind, "target": target,
                        "static_pressure_bracket": [lo, hi],
                        "evaluated_static_pressure_Pa": [lo, hi],
                        "evaluated_metrics": [m_lo, m_hi],
                        "signed_residuals": [r_lo, r_hi]})
            a, f_a, b = lo, r_lo, hi
            iterations = 0
            for iterations in range(1, max_iter + 1):
                mid = a + 0.5 * (b - a)
                if not (a < mid < b):
                    raise fail("stalled_adjacent_floats",
                               "stalled on adjacent floats without closure")
                m_mid, s_mid = evaluate(mid)
                f_mid = m_mid - target
                if abs(f_mid) <= tolerance_product:
                    return finish(mid, m_mid, s_mid, iterations)
                if (f_a < 0.0) == (f_mid < 0.0):
                    a, f_a = mid, f_mid
                else:
                    b = mid
            raise fail("iteration_limit",
                       "bisection iteration limit without closure")
        finally:
            self._namespace["p_т"] = original_p_t

    def solve_total_outlet_pressure(
        self, n: float, D_hub: float, target_total_pressure: float, *,
        static_pressure_bracket: Sequence[float],
        relative_tol: float = 1e-6, max_iter: int = 40,
    ) -> dict[str, Any]:
        """Solve outlet static pressure for an actual last-stage total pressure."""

        return self._solve_static_pressure_closure(
            n=n, D_hub=D_hub, target=target_total_pressure,
            relative_tol=relative_tol, max_iter=max_iter,
            static_pressure_bracket=static_pressure_bracket,
            metric_kind="total_outlet_pressure",
            stage="orchestration.solve_total_outlet_pressure")

    def solve_prescribed_mechanical_power(
        self, n: float, D_hub: float, target_power: float,
        mechanical_efficiency: float, *,
        static_pressure_bracket: Sequence[float],
        relative_tol: float = 1e-6, max_iter: int = 40,
    ) -> dict[str, Any]:
        """Solve outlet static pressure for actual aggregate turbine power."""

        return self._solve_static_pressure_closure(
            n=n, D_hub=D_hub, target=target_power,
            relative_tol=relative_tol, max_iter=max_iter,
            static_pressure_bracket=static_pressure_bracket,
            metric_kind="mechanical_power",
            stage="orchestration.solve_prescribed_mechanical_power",
            mechanical_efficiency=mechanical_efficiency)


def build_const_hub_orchestration(
    namespace: Mapping[str, Any],
    *,
    interp: InterpFunction | None = None,
    clip: ClipFunction | None = None,
    validate: bool = True,
    pressure_ratio_policy: str = PRESSURE_RATIO_LEGACY_CLIPPED,
) -> ConstHubOrchestration:
    """Create the extracted const-hub orchestration for a live namespace.

    Parameters
    ----------
    namespace:
        Notebook namespace.  Required entries are listed in
        :data:`REQUIRED_NAMESPACE_KEYS`; they are read live at call time.
    interp, clip:
        Optional source-compatible interpolation and clip callables.
    validate:
        When true (default), raise ``KeyError`` if a required dependency is
        absent at construction time.  It does not freeze later changes.
    pressure_ratio_policy:
        ``"legacy_clipped"`` (default) reproduces the source secant clip.
        ``"raw_unbounded"`` removes the engineering clip and enforces only the
        p-expansion domain ``y > 1``.

    Returns
    -------
    ConstHubOrchestration
        Object exposing the three source functions.  Use :meth:`install` to put
        them into the namespace under their notebook names.
    """

    orchestration = ConstHubOrchestration(
        namespace,
        interp=interp,
        clip=clip,
        pressure_ratio_policy=pressure_ratio_policy,
    )
    if validate:
        missing = orchestration.missing_namespace_keys()
        if missing:
            raise KeyError(
                "const-hub namespace is missing required dependencies: "
                + ", ".join(repr(key) for key in missing)
            )
    return orchestration


__all__ = [
    "ConstHubOrchestration",
    "REQUIRED_NAMESPACE_KEYS",
    "STAGE_ITERATION_LIMIT",
    "REACTION_TOLERANCE",
    "PI_SECANT_ITERATION_LIMIT",
    "PI_RESIDUAL_TOLERANCE",
    "PI_SECANT_TOLERANCE",
    "PI_CLIP_LOWER",
    "PI_CLIP_UPPER_FACTOR",
    "INITIAL_REACTION",
    "INLET_DIAMETER_OFFSET",
    "INITIAL_PI_STEP_FACTOR",
    "PRESSURE_RATIO_LEGACY_CLIPPED",
    "PRESSURE_RATIO_RAW_UNBOUNDED",
    "PRESSURE_RATIO_POLICIES",
    "build_const_hub_orchestration",
    "clip_scalar",
    "linear_interp",
    "validate_pressure_ratio_policy",
]
