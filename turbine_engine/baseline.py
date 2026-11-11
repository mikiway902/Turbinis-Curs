"""Conventional multi-stage turbine orchestration from notebook cell 16.

PROVENANCE
==========
This module is a source-preserving extraction of the numerical part of code
cell 16 of ``Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb`` as it is executed by
``Optimizer/optimizer_core.load_model`` / ``_evaluate(stage_core=True)``.  It
executes no notebook and imports only the standard library.

The extracted cell contains four source functions:

* ``турбина(n, D_т, π_ст_)`` -- build the ``z``-stage sequence at a common
  last-stage diameter and a common stage pressure ratio;
* ``распределить_π(n, D_т, π0)`` -- clipped secant solve for that pressure
  ratio from the last-stage ``p*0 / p2`` closure;
* ``решить(π0)`` -- half-relaxed iteration linking the last-stage annulus area
  ``A_т``, rotational speed ``n_т``, diameter ``D_т`` and stage sequence;
* ``N_на_валу(G, π0)`` -- shaft-power residual for an air-flow multiplier
  ``G / G_в``.

The notebook also contains a top-level outer secant solve for ``G``, the
per-stage numerical outputs, and two pandas presentation tables.  The outer
solve is reproduced by :meth:`BaselineTurbine.calculate`; the tables are
presentation only and are intentionally absent here.  When
``stage_core=False`` the notebook cell 16 is still executed in full, including
its tables, as the independent comparison path.

POWER PLANE
===========
The outer ``G`` solve targets ``N_зад`` at the gearbox input, i.e. **before**
the reduction gear.  The extracted and notebook paths therefore use
``N_в = L_тΣ G_c1 η_м - G_в L_к`` without the old trailing ``η_р`` factor.
Reducer loss is a downstream output, not part of this target.  The previous
``... * η_р`` expression is retained only in the labelled legacy differential
tests and in historical/archive scripts.

LAST-STAGE PRESSURE BOUNDARY
============================
Turbine book part 1, printed p. 92, items 138-140 prescribe the last stage
outlet ``p2z = p_t`` and the available drop
``H0z = cp*T0z*[1 - (p2z/p0z*)^((k-1)/k)]``.  For positive ``cp``, ``T0z`` and
``k > 1`` this drop is positive only when ``p0z* > p_t``; the checked text has
no percentage margin.  This extraction therefore keeps only the
``p0 <= p_t`` no-positive-drop rejection.  The earlier ``p0 <= 1.02*p_t``
guard was a legacy code margin with no located source justification.  Removing
it does not assert that a resulting stage is physically feasible: the stage
calculations and their physical checks remain the authority for that.

STABLE-FLOW SEMANTICS
=====================
The extraction copies the currently executed, already patched source.  Every
trial flow starts from the same original state ``(G0, Gc10, Gl0, A0)`` and then
scales the four state quantities by ``G / G0``; the source's raw secant
side-effect accumulation is not revived.  The initial bracket
``(0.9 G0, 1.1 G0)``, the 20-iteration limit, the ``1e-5`` relative residual,
the ``1e-9`` denominator guard, the ``[0.5 G0, 2.0 G0]`` clamp, and the
half-relaxed ``A_т`` update are unchanged.  This is the same code that
``optimizer_core._NEW_G`` would introduce, and the anchor notebook currently
already contains it.

NUMERICAL FAILURES
==================
The source returns the last secant iterate after its iteration limit and can
return ``(y_b, None)`` when the final trial has no valid stage evaluation.  The
extracted solver keeps the internal ``(None, None)`` residual marker so the
bisection fallback and the namespace ``НетРешения`` handling are unchanged, but
it no longer returns that stale state as a result.  A final invalid evaluation,
secant denominator stall, or exhausted pressure/area/outer-flow iteration raises
:class:`turbine_engine.diagnostics.NumericalConvergenceError` with the solver
stage, stable reason, status and last finite iterate.  The successful path keeps
the same equations, evaluation order, limits and tolerances.

NO THIRD-PARTY IMPORTS
======================
All external numerical dependencies are looked up in the supplied namespace at
call time: ``ступень_авто``, ``H0_по_p``, ``НетРешения`` and every scalar or
array.  ``np.clip`` and ``np.mean`` are injected callables;
:func:`clip_scalar` and :func:`mean_scalar` are standard-library defaults.  The
production adapter passes ``numpy.clip`` and ``numpy.mean`` so the arithmetic
is source-identical.

The entry point :func:`calculate_baseline_turbine` does not modify the supplied
namespace.  It returns a :class:`BaselineTurbineResult`; the caller assigns
``legacy_values()`` into the namespace, which reproduces the source cell's
final global state without leaking trial-flow mutations between candidates.

Tests compare this module with the actual legacy cell-16 execution only as a
software-drift diagnostic, never as physical validation.  No equations,
bounds, acceptance criteria, or physical interpretation are added.

UNITS
=====
SI throughout.  Temperatures are K, pressures Pa, heat drops J/kg, mass flows
kg/s, areas m², rotational speed rpm, diameters and blade lengths m, and
``σ_л`` is the rotor-blade stress in MPa.  ``k_n`` includes the source's unit
conversion.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from turbine_engine.diagnostics import NumericalConvergenceError

ClipFunction = Callable[[float, float, float], float]
MeanFunction = Callable[[Sequence[float]], float]

# Source-fixed numerical controls.  Do not tune without changing the notebook.
OUTER_ITERATION_LIMIT = 20
OUTER_RESIDUAL_TOLERANCE = 1e-5
OUTER_DENOMINATOR_TOLERANCE = 1e-9
OUTER_INITIAL_LOW_FACTOR = 0.9
OUTER_INITIAL_HIGH_FACTOR = 1.1
OUTER_LOWER_FACTOR = 0.5
OUTER_UPPER_FACTOR = 2.0
AREA_ITERATION_LIMIT = 40
AREA_RELATIVE_TOLERANCE = 1e-4
AREA_RELAXATION = 0.5
PI_SECANT_ITERATION_LIMIT = 30
PI_RESIDUAL_TOLERANCE = 1e-5
PI_CLIP_LOWER = 1.05
PI_CLIP_UPPER_FACTOR = 1.2
INITIAL_PI_STEP_FACTOR = 1.02

#: Namespace entries required by the extracted cell.
REQUIRED_NAMESPACE_KEYS: tuple[str, ...] = (
    "G_в",
    "G_c1",
    "G_л_ост",
    "A_т0",
    "π_т",
    "z",
    "D_ст",
    "T_г_полн",
    "p_г_полн",
    "μ",
    "p_т",
    "π_веса",
    "α1_ст",
    "κ_ст",
    "закон_ст",
    "ступень_авто",
    "H0_по_p",
    "НетРешения",
    "π_ст_макс",
    "c_та",
    "c0_т",
    "σ_л",
    "k_n",
    "q_n",
    "l_т_D_т",
    "y0",
    "η_м",
    "L_к",
    "N_зад",
)


def clip_scalar(value: float, lower: float, upper: float) -> float:
    """Scalar ``numpy.clip`` substitute for finite values.

    ``NaN`` is returned unchanged because both comparisons are false, matching
    ``numpy.clip`` for a scalar.
    """

    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def mean_scalar(values: Sequence[float]) -> float:
    """Standard-library mean of a short numeric sequence."""

    values = tuple(values)
    return sum(values) / len(values)


def _stage_count_error(
    reason: str, context: Mapping[str, float], **derived: Any
) -> None:
    raise NumericalConvergenceError(
        reason.replace("_", " "),
        stage="baseline.estimate_stage_count",
        reason=reason,
        last_valid_state={**context, **derived},
    )


def estimate_stage_count(c0, u_mean, y0, recovered_exit_velocity):
    """Return the source item-54 stage-count estimate without rounding.

    Exact inline arithmetic with ``recovered_exit_velocity = μ * c2_mean``::

        z = (c0**2 / v**2 - 1) / (u_mean**2 / (v**2 * y0**2) - 1)

    The result is not clamped, rounded or filtered.  A zero/non-finite
    denominator, non-finite numerator or non-finite result raises the typed
    numerical diagnostic; a near-zero denominator is never convergence.
    """

    context = {"c0": c0, "u_mean": u_mean, "y0": y0,
               "recovered_exit_velocity": recovered_exit_velocity}
    for name, value in context.items():
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}")
        context[name] = number

    try:
        numerator = context["c0"] ** 2 / context["recovered_exit_velocity"] ** 2 - 1.0
    except (ZeroDivisionError, OverflowError):
        _stage_count_error("stage_count_numerator_non_finite", context)
    if not math.isfinite(numerator):
        _stage_count_error("stage_count_numerator_non_finite", context)
    try:
        denominator = (context["u_mean"] ** 2
                       / (context["recovered_exit_velocity"] ** 2 * context["y0"] ** 2)
                       - 1.0)
    except ZeroDivisionError:
        _stage_count_error("stage_count_denominator_zero", context, numerator=numerator)
    except OverflowError:
        _stage_count_error("stage_count_denominator_non_finite", context,
                           numerator=numerator)
    if denominator == 0.0 or not math.isfinite(denominator):
        _stage_count_error("stage_count_denominator_zero" if denominator == 0.0
                           else "stage_count_denominator_non_finite", context,
                           numerator=numerator, denominator=denominator)
    try:
        estimate = numerator / denominator
    except OverflowError:
        _stage_count_error("stage_count_estimate_non_finite", context,
                           numerator=numerator, denominator=denominator)
    if not math.isfinite(estimate):
        _stage_count_error("stage_count_estimate_non_finite", context,
                           numerator=numerator, denominator=denominator,
                           estimate=estimate)
    return estimate


@dataclass(frozen=True)
class BaselineTurbineResult:
    """Numerical result of the cell-16 conventional multi-stage calculation.

    The field names are ASCII equivalents of the notebook identifiers:

    ===================  ==================
    result field         notebook identifier
    ===================  ==================
    ``G_v_cycle``        ``G_в_цикл``
    ``pi0``              ``π0``
    ``G_v``              ``G_в``
    ``G_c1``             ``G_c1``
    ``G_l_ost``          ``G_л_ост``
    ``A_t0``             ``A_т0``
    ``G_a``              ``G_a``
    ``G_b``              ``G_b``
    ``N_a``              ``N_a``
    ``N_b``              ``N_b``
    ``A_t``              ``A_т``
    ``A_fact``           ``A_факт``
    ``n_t``              ``n_т``
    ``D_t``              ``D_т``
    ``u_t``              ``u_т``
    ``z_calc``           ``z_расч``
    ``stages``           ``ступени``
    ``iterations``       ``итерации`` (inside the ``решить`` result)
    ``sigma_l_fact``     ``σ_л_факт``
    ===================  ==================
    """

    G_v_cycle: float
    pi0: float
    G_v: float
    G_c1: float
    G_l_ost: float
    A_t0: float
    G_a: float
    G_b: float
    N_a: float
    N_b: float
    A_t: float
    A_fact: float
    n_t: float
    D_t: float
    u_t: float
    z_calc: float
    stages: list[dict[str, Any]]
    iterations: list[tuple[Any, ...]]
    sigma_l_fact: float

    def legacy_values(self) -> dict[str, Any]:
        """Return the final cell-16 globals under their exact identifiers."""

        return {
            "G_в_цикл": self.G_v_cycle,
            "π0": self.pi0,
            "G_в": self.G_v,
            "G_c1": self.G_c1,
            "G_л_ост": self.G_l_ost,
            "A_т0": self.A_t0,
            "G_a": self.G_a,
            "G_b": self.G_b,
            "N_a": self.N_a,
            "N_b": self.N_b,
            "A_т": self.A_t,
            "A_факт": self.A_fact,
            "n_т": self.n_t,
            "D_т": self.D_t,
            "u_т": self.u_t,
            "z_расч": self.z_calc,
            "ступени": self.stages,
            "σ_л_факт": self.sigma_l_fact,
        }


class BaselineTurbine:
    """Source-compatible numerical functions from notebook cell 16.

    Parameters
    ----------
    namespace:
        Live mapping containing the notebook dependencies listed in
        :data:`REQUIRED_NAMESPACE_KEYS`.  External entries are resolved on each
        use, like the notebook's globals.  The mapping is not modified.
    clip:
        Scalar clip callable ``(value, lower, upper)``.  Defaults to
        :func:`clip_scalar`.
    mean:
        Mean callable over a sequence of stage exit velocities.  Defaults to
        :func:`mean_scalar`.

    The object's ``G_v``, ``G_c1``, ``G_l_ost`` and ``A_t0`` attributes mirror
    the four mutable notebook globals.  :meth:`N_на_валу` updates them exactly
    as the source does; :meth:`calculate` resets them from the namespace before
    every run, so repeated calls are deterministic and never accumulate flow
    scaling.
    """

    def __init__(
        self,
        namespace: Mapping[str, Any],
        *,
        clip: ClipFunction | None = None,
        mean: MeanFunction | None = None,
    ) -> None:
        self._namespace = namespace
        self._clip = clip if clip is not None else clip_scalar
        self._mean = mean if mean is not None else mean_scalar
        self.G_v = namespace["G_в"]
        self.G_c1 = namespace["G_c1"]
        self.G_l_ost = namespace["G_л_ост"]
        self.A_t0 = namespace["A_т0"]

    @property
    def namespace(self) -> Mapping[str, Any]:
        """The live dependency mapping supplied at construction."""

        return self._namespace

    def missing_namespace_keys(self) -> tuple[str, ...]:
        """Return required notebook dependencies absent from the namespace."""

        return tuple(
            key for key in REQUIRED_NAMESPACE_KEYS if key not in self._namespace
        )

    def _value(self, name: str) -> Any:
        return self._namespace[name]

    def турбина(
        self, n: float, D_т: float, π_ст_: float
    ) -> list[dict[str, Any]]:
        """Build the stage sequence at ``n``, last-stage diameter and ``π``.

        Source items 70-141 are copied in order.  The first stage uses the
        combustion-chamber total state and ``G_c1`` for both nozzle and rotor;
        later stages use the previous stage outlet, reconstruct the interstage
        total pressure, and use ``G_л_ост``.  All stages except the last receive
        their pressure ratio from ``π_веса``; the last stage closes on ``p_т``.
        The ``НетРешения`` guard rejects only ``p0 <= p_т`` (no positive
        available drop); any ``p0 > p_т`` is passed to the stage calculation.
        """

        res: list[dict[str, Any]] = []
        z = self._value("z")
        for i in range(1, z + 1):
            D_c = D_л = self._value("D_ст")[i - 1] * D_т
            if i == 1:
                T0 = self._value("T_г_полн")
                p0 = self._value("p_г_полн")
                G_c = self.G_c1
                G_л = self.G_c1
            else:
                пр = res[-1]
                T0 = пр["T_ст_п"]
                p0 = пр["p2"] * (
                    1
                    + (self._value("μ") * пр["c2"]) ** 2
                    / (2 * пр["cp"] * пр["T_ст"])
                ) ** (пр["k"] / (пр["k"] - 1))
                G_c = G_л = self.G_l_ost
            if i < z:
                H0 = self._value("H0_по_p")(
                    T0,
                    p0,
                    p0
                    / π_ст_
                    ** (
                        self._value("π_веса")[i - 1]
                        / self._value("π_веса")[-1]
                    ),
                )
            else:
                p_t = self._value("p_т")
                if p0 <= p_t:
                    raise self._value("НетРешения")(
                        "давление перед последней ступенью не выше p_т: "
                        "нет положительного перепада"
                    )
                H0 = self._value("H0_по_p")(T0, p0, p_t)
            res.append(
                self._value("ступень_авто")(
                    T0,
                    p0,
                    H0,
                    D_c,
                    D_л,
                    n,
                    G_c,
                    G_л,
                    self._value("α1_ст")[i - 1],
                    self._value("κ_ст")[i - 1],
                    self._value("закон_ст")[i - 1],
                )
            )
        return res

    def распределить_π(
        self, n: float, D_т: float, π0: float
    ) -> tuple[float, list[dict[str, Any]] | None]:
        """Solve the common stage pressure ratio by a clipped secant method.

        A failed evaluation still returns ``(None, None)`` from the internal
        residual so the bisection fallback can continue, and the initial failed
        evaluation still raises the namespace's ``НетРешения``.  Unlike the
        source, a final iteration without a valid stage evaluation or a secant
        that exhausts its limit no longer returns a stale ``None``/iterate: it
        raises :class:`turbine_engine.diagnostics.NumericalConvergenceError`
        with the stage, reason and last finite iterate.
        """

        def f(y: float) -> tuple[float | None, list[dict[str, Any]] | None]:
            try:
                ст = self.турбина(n, D_т, y)
            except self._value("НетРешения"):
                return None, None
            return ст[-1]["π_ст"] - y, ст

        y_a, y_b = π0, π0 * INITIAL_PI_STEP_FACTOR
        f_a, ст = f(y_a)
        if f_a is None:
            raise self._value("НетРешения")(
                "нет решения при начальной π ступени"
            )
        converged = False
        f_b: float | None = None
        ст_b: list[dict[str, Any]] | None = None
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
                    "secant residual is not finite at the evaluated stage state",
                    stage="baseline.распределить_π",
                    reason="non_finite_residual",
                    last_valid_state={**last_valid, "y_b": y_b},
                )
            last_valid = {"y": float(y_b), "residual": float(f_b)}
            if abs(f_b) < PI_RESIDUAL_TOLERANCE:
                converged = True
                break
            den = f_b - f_a
            if not math.isfinite(den) or den == 0.0:
                raise NumericalConvergenceError(
                    "secant denominator is zero or not finite",
                    stage="baseline.распределить_π",
                    reason="secant_denominator_stalled",
                    last_valid_state=last_valid,
                )
            y_a, y_b, f_a = (
                y_b,
                y_b - f_b * (y_b - y_a) / den,
                f_b,
            )
            y_b = float(
                self._clip(
                    y_b,
                    PI_CLIP_LOWER,
                    self._value("π_ст_макс") * PI_CLIP_UPPER_FACTOR,
                )
            )
        if not converged:
            if f_b is None or ст_b is None:
                raise NumericalConvergenceError(
                    "pressure-ratio secant ended without a valid stage "
                    "evaluation",
                    stage="baseline.распределить_π",
                    reason="no_valid_final_evaluation",
                    last_valid_state={
                        **last_valid,
                        "iteration_limit": PI_SECANT_ITERATION_LIMIT,
                        "residual_tolerance": PI_RESIDUAL_TOLERANCE,
                    },
                )
            raise NumericalConvergenceError(
                "pressure-ratio secant exhausted its iteration limit",
                stage="baseline.распределить_π",
                reason="pi_secant_iteration_exhausted",
                last_valid_state={
                    **last_valid,
                    "iteration_limit": PI_SECANT_ITERATION_LIMIT,
                    "residual_tolerance": PI_RESIDUAL_TOLERANCE,
                },
            )
        return y_b, ст_b

    def решить(self, π0: float) -> dict[str, Any]:
        """Iterate the last-stage area, speed and stage sequence.

        Source items 9-11/46-54 are copied in order: initial ``A_т = A_т0``
        and ``c2_ср = c_та``, at most 40 iterations, ``A_т`` half-relaxation,
        and the strict ``abs(A_факт - A_т) / A_т < 1e-4`` break test.  The
        return dictionary and iteration-log tuples keep the source shapes and
        field order, including the final local ``π0``.
        """

        A_т = self.A_t0
        c2_ср = self._value("c_та")
        итерации: list[tuple[Any, ...]] = []
        converged = False
        A_факт = 0.0
        n_т = 0.0
        D_т = 0.0
        u_т = 0.0
        z_расч = 0.0
        A_t_eval = A_т
        A_t_next = A_т
        relative_residual = math.inf
        ст: list[dict[str, Any]] | None = None
        for it in range(AREA_ITERATION_LIMIT):
            A_t_eval = A_т
            n_т = self._value("q_n") * math.sqrt(
                self._value("σ_л") / (self._value("k_n") * A_т)
            )
            D_т = math.sqrt(A_т / (math.pi * self._value("l_т_D_т")))
            u_т = math.pi * D_т * n_т / 60
            u_1 = math.pi * self._value("D_ст")[0] * D_т * n_т / 60
            u_ср = 0.5 * (u_т + u_1)
            μc = self._value("μ") * c2_ср
            z_расч = estimate_stage_count(
                self._value("c0_т"),
                u_ср,
                self._value("y0"),
                μc,
            )
            π0, ст = self.распределить_π(n_т, D_т, π0)
            if not ст:
                raise NumericalConvergenceError(
                    "area iteration received no stage solution",
                    stage="baseline.решить",
                    reason="stage_solution_unavailable",
                    last_valid_state={
                        "iteration": it,
                        "A_t": A_t_eval,
                        "n_t": n_т,
                        "D_t": D_т,
                    },
                )
            A_факт = math.pi * ст[-1]["D_л"] * ст[-1]["l_л"]
            c2_ср = float(self._mean([s["c2"] for s in ст]))
            итерации.append(
                (it, A_т * 1e4, n_т, D_т * 1e3, u_т, z_расч, π0, A_факт * 1e4)
            )
            relative_residual = abs(A_факт - A_т) / A_т
            if relative_residual < AREA_RELATIVE_TOLERANCE:
                converged = True
                break
            A_т = AREA_RELAXATION * (A_т + A_факт)
            A_t_next = A_т
        if not converged:
            raise NumericalConvergenceError(
                "last-stage area iteration exhausted its iteration limit",
                stage="baseline.решить",
                reason="area_iteration_exhausted",
                last_valid_state={
                    "iteration": len(итерации),
                    "A_t": A_t_eval,
                    "A_fact": A_факт,
                    "n_t": n_т,
                    "D_t": D_т,
                    "u_t": u_т,
                    "z_calc": z_расч,
                    "pi0": π0,
                    "relative_residual": relative_residual,
                    "A_t_next": A_t_next,
                    "iteration_limit": AREA_ITERATION_LIMIT,
                    "relative_tolerance": AREA_RELATIVE_TOLERANCE,
                },
            )
        if ст is None:
            raise NumericalConvergenceError(
                "area iteration produced no stage solution",
                stage="baseline.решить",
                reason="stage_solution_unavailable",
                last_valid_state={
                    "iteration": len(итерации),
                    "A_t": A_т,
                    "A_fact": A_факт,
                    "n_t": n_т,
                    "D_t": D_т,
                },
            )
        return dict(
            A_т=A_т,
            A_факт=A_факт,
            n_т=n_т,
            D_т=D_т,
            u_т=u_т,
            z_расч=z_расч,
            π0=π0,
            ступени=ст,
            итерации=итерации,
        )

    def N_на_валу(
        self, G: float, π0: float
    ) -> tuple[float, dict[str, Any]]:
        """Return the pre-reducer shaft power and solution at trial flow ``G``.

        The target plane is the gearbox input: the source's old trailing
        ``η_р`` factor is intentionally **not** applied here.  Reducer losses
        belong to a separate downstream output and must not be folded into the
        requested ``N_зад``.  Consequently ``η_р`` is no longer a required
        namespace dependency of this closure.

        The source's in-place flow scaling is preserved: ``G_в`` becomes ``G``
        and the other three state quantities are multiplied by ``G / G_в``
        evaluated before the assignment.  The caller is expected to reset the
        state to ``(G0, Gc10, Gl0, A0)`` between trials, exactly as the patched
        ``_N_of`` wrapper does.
        """

        k = G / self.G_v
        self.G_v, self.G_c1, self.G_l_ost, self.A_t0 = (
            G,
            self.G_c1 * k,
            self.G_l_ost * k,
            self.A_t0 * k,
        )
        r = self.решить(π0)
        L_тΣ_ = sum(
            s["G_c"] / self.G_c1 * s["L_т"] for s in r["ступени"]
        )
        return (
            L_тΣ_ * self.G_c1 * self._value("η_м")
            - self.G_v * self._value("L_к")
        ), r

    def calculate(self) -> BaselineTurbineResult:
        """Run the complete cell-16 outer ``G`` secant solve.

        This is the currently executed, patched top-level block.  The original
        state is captured once, every trial starts from it, and the returned
        result contains the final source globals plus the final stage sequence
        and iteration log.  The supplied namespace is not modified.
        """

        G_в_цикл = self._value("G_в")
        self.G_v = G_в_цикл
        self.G_c1 = self._value("G_c1")
        self.G_l_ost = self._value("G_л_ост")
        self.A_t0 = self._value("A_т0")
        π0 = self._value("π_т") ** (1 / self._value("z"))
        G0, Gc10, Gl0, A0 = self.G_v, self.G_c1, self.G_l_ost, self.A_t0

        def _N_of(G: float) -> tuple[float, dict[str, Any]]:
            self.G_v, self.G_c1, self.G_l_ost, self.A_t0 = (
                G0,
                Gc10,
                Gl0,
                A0,
            )
            return self.N_на_валу(G, π0)

        G_a = OUTER_INITIAL_LOW_FACTOR * G0
        G_b = OUTER_INITIAL_HIGH_FACTOR * G0
        N_a, _r = _N_of(G_a)
        N_b, _r = _N_of(G_b)
        converged = False
        outer_iterations = 0
        for _ in range(OUTER_ITERATION_LIMIT):
            outer_iterations += 1
            if abs(N_b - self._value("N_зад")) < (
                OUTER_RESIDUAL_TOLERANCE * self._value("N_зад")
            ):
                converged = True
                break
            den = N_b - N_a
            G_new = (
                0.5 * (G_a + G_b)
                if abs(den) < OUTER_DENOMINATOR_TOLERANCE
                else G_b
                + (self._value("N_зад") - N_b) * (G_b - G_a) / den
            )
            G_new = min(
                max(G_new, OUTER_LOWER_FACTOR * G0), OUTER_UPPER_FACTOR * G0
            )
            G_a, G_b, N_a = G_b, G_new, N_b
            N_b, _r = _N_of(G_b)
        if not converged and abs(N_b - self._value("N_зад")) < (
            OUTER_RESIDUAL_TOLERANCE * self._value("N_зад")
        ):
            # The final secant update may land on the target after the last
            # in-loop test; the source would return that state as converged.
            converged = True
        if not converged:
            target = self._value("N_зад")
            residual = abs(N_b - target)
            raise NumericalConvergenceError(
                "outer flow secant exhausted its iteration limit",
                stage="baseline.calculate",
                reason="outer_flow_iteration_exhausted",
                last_valid_state={
                    "G_a": G_a,
                    "G_b": G_b,
                    "N_a": N_a,
                    "N_b": N_b,
                    "N_target": target,
                    "residual": residual,
                    "relative_residual": (
                        residual / abs(target) if target else residual
                    ),
                    "iterations": outer_iterations,
                    "iteration_limit": OUTER_ITERATION_LIMIT,
                    "relative_tolerance": OUTER_RESIDUAL_TOLERANCE,
                },
            )

        (
            A_т,
            A_факт,
            n_т,
            D_т,
            u_т,
            z_расч,
            ступени,
        ) = (
            _r[k]
            for k in (
                "A_т",
                "A_факт",
                "n_т",
                "D_т",
                "u_т",
                "z_расч",
                "ступени",
            )
        )
        σ_л_факт = self._value("k_n") * n_т**2 * A_факт

        return BaselineTurbineResult(
            G_v_cycle=G_в_цикл,
            pi0=π0,
            G_v=self.G_v,
            G_c1=self.G_c1,
            G_l_ost=self.G_l_ost,
            A_t0=self.A_t0,
            G_a=G_a,
            G_b=G_b,
            N_a=N_a,
            N_b=N_b,
            A_t=A_т,
            A_fact=A_факт,
            n_t=n_т,
            D_t=D_т,
            u_t=u_т,
            z_calc=z_расч,
            stages=ступени,
            iterations=_r["итерации"],
            sigma_l_fact=σ_л_факт,
        )


def calculate_baseline_turbine(
    namespace: Mapping[str, Any],
    *,
    clip: ClipFunction | None = None,
    mean: MeanFunction | None = None,
    validate: bool = True,
) -> BaselineTurbineResult:
    """Evaluate the conventional multi-stage cell-16 calculation.

    Parameters
    ----------
    namespace:
        Notebook namespace with the dependencies listed in
        :data:`REQUIRED_NAMESPACE_KEYS`.  It is read only and is not modified.
    clip, mean:
        Optional source-compatible clip and mean callables.  Production wiring
        passes ``numpy.clip`` and ``numpy.mean``.
    validate:
        When true (default), raise ``KeyError`` if a required dependency is
        absent before any calculation starts.

    Returns
    -------
    BaselineTurbineResult
        Final source globals, stage sequence, iteration log, and the
        recomputed ``σ_л_факт``.
    """

    if validate:
        missing = tuple(
            key for key in REQUIRED_NAMESPACE_KEYS if key not in namespace
        )
        if missing:
            raise KeyError(
                "baseline turbine namespace is missing required dependencies: "
                + ", ".join(repr(key) for key in missing)
            )
    return BaselineTurbine(namespace, clip=clip, mean=mean).calculate()


__all__ = [
    "AREA_ITERATION_LIMIT",
    "AREA_RELATIVE_TOLERANCE",
    "AREA_RELAXATION",
    "INITIAL_PI_STEP_FACTOR",
    "OUTER_DENOMINATOR_TOLERANCE",
    "OUTER_INITIAL_HIGH_FACTOR",
    "OUTER_INITIAL_LOW_FACTOR",
    "OUTER_ITERATION_LIMIT",
    "OUTER_LOWER_FACTOR",
    "OUTER_RESIDUAL_TOLERANCE",
    "OUTER_UPPER_FACTOR",
    "PI_CLIP_LOWER",
    "PI_CLIP_UPPER_FACTOR",
    "PI_RESIDUAL_TOLERANCE",
    "PI_SECANT_ITERATION_LIMIT",
    "BaselineTurbine",
    "BaselineTurbineResult",
    "ClipFunction",
    "MeanFunction",
    "REQUIRED_NAMESPACE_KEYS",
    "calculate_baseline_turbine",
    "clip_scalar",
    "estimate_stage_count",
    "mean_scalar",
]
