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

    The object stores no computed state.  Calls are deterministic for the
    current namespace values, just like the notebook globals.
    """

    def __init__(
        self,
        namespace: Mapping[str, Any],
        *,
        interp: InterpFunction | None = None,
        clip: ClipFunction | None = None,
    ) -> None:
        self._namespace = namespace
        self._interp = interp if interp is not None else linear_interp
        self._clip = clip if clip is not None else clip_scalar
        self._installed = False

    @property
    def namespace(self) -> Mapping[str, Any]:
        """The live dependency mapping supplied at construction."""

        return self._namespace

    def missing_namespace_keys(self) -> tuple[str, ...]:
        """Return required notebook dependencies absent from the namespace."""

        return tuple(
            key for key in REQUIRED_NAMESPACE_KEYS if key not in self._namespace
        )

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
        """Solve the common stage pressure ratio by a clipped secant method.

        This reproduces the source secant loop, including the bisection fallback
        when ``турбина_const_hub`` raises the namespace's ``НетРешения``, the
        residual and denominator stops, and the ``[1.05, 1.2 * π_ст_макс]``
        clip.  The source's silent ``(None, None)`` return and its ``None``
        stage list are replaced by explicit diagnostics: an invalid initial
        evaluation raises the namespace's ``НетРешения``, while a stalled or
        exhausted secant, or a final iteration without a valid evaluation,
        raises :class:`turbine_engine.diagnostics.NumericalConvergenceError`.
        A small denominator is never treated as convergence unless the residual
        itself is already below the existing tolerance.
        """

        def f(y: float) -> tuple[float | None, list[dict[str, float]] | None]:
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
                    last_valid_state={**last_valid, "y_b": y_b},
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
                    last_valid_state=last_valid,
                )
            if abs(den) < PI_SECANT_TOLERANCE:
                stalled = True
                break
            y_a, y_b, f_a = y_b, y_b - f_b * (y_b - y_a) / den, f_b
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
                    stage="orchestration.распределить_π_const_hub",
                    reason="no_valid_final_evaluation",
                    last_valid_state={
                        **last_valid,
                        "iteration_limit": PI_SECANT_ITERATION_LIMIT,
                        "residual_tolerance": PI_RESIDUAL_TOLERANCE,
                    },
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
                last_valid_state={
                    **last_valid,
                    "iteration_limit": PI_SECANT_ITERATION_LIMIT,
                    "residual_tolerance": PI_RESIDUAL_TOLERANCE,
                    "secant_tolerance": PI_SECANT_TOLERANCE,
                },
            )
        return y_b, ст_b


def build_const_hub_orchestration(
    namespace: Mapping[str, Any],
    *,
    interp: InterpFunction | None = None,
    clip: ClipFunction | None = None,
    validate: bool = True,
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

    Returns
    -------
    ConstHubOrchestration
        Object exposing the three source functions.  Use :meth:`install` to put
        them into the namespace under their notebook names.
    """

    orchestration = ConstHubOrchestration(namespace, interp=interp, clip=clip)
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
    "build_const_hub_orchestration",
    "clip_scalar",
    "linear_interp",
]
