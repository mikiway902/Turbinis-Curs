"""Stateless single-point cycle core extracted from Turbinis-Cycle.

PROVENANCE
==========
This module is a formula-preserving extraction of the scalar cycle code in
``Turbinis-Cycle/Turbinis-Cycle.ipynb`` (code cells 2-5), combined with the
derived fields that ``Optimizer/optimizer_core.cycle_point`` previously
computed after calling ``calc_tvd``.  It executes no notebook, performs no
sweep/optimization, and keeps no mutable global state.

The notebook remains the independent legacy comparison path.  Constants below
are the *current* notebook assumptions and are documented with their source
comments; they are not new engineering claims.  Tests compare this core with
observed legacy outputs only as software-drift diagnostics.

POWER-PLANE CONTRACT (user-confirmed 2026-11-10)
================================================
``CycleState.N`` is the requested **mechanical shaft power at the gearbox
input, before the reduction gear**.  Consequently the specific shaft power
``N_ud`` is ``L_t * g_sum * eta_mech_turbine - L_k`` and the reducer efficiency
is *not* applied to it.  ``G = N / N_ud`` therefore sizes the cycle for the
pre-reducer plane.  The downstream gearbox-output specific power is exposed
separately as ``N_reducer_out_ud = N_ud * eta_reducer``.  ``N_e_ud`` (the
equivalent shaft-plus-nozzle-thrust metric) is built on that downstream plane,
while ``C_e`` is deliberately based on the requested pre-reducer shaft power.
The legacy notebook path (which multiplied the old ``N_ud`` by ``eta_reducer``)
is retained only as a labelled differential reference in the tests.

UNITS
=====
SI, as in the source notebook.  ``q_lhv_kj`` is kJ/kg (the notebook calls the
same quantity ``Q_н_р``); all other energies are J/kg.  Angles would be in
degrees if any were used here (none are).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

# Solver controls copied from the notebook defaults (absolute tolerances in K
# for the compressor/turbine temperature iteration and in pressure-ratio units
# for the turbine expansion iteration).
SOLVER_TOL = 1e-6
SOLVER_MAX_ITER = 100


class CycleSolverError(RuntimeError):
    """The extracted scalar iteration did not converge or lost validity."""


@dataclass(frozen=True)
class CycleConfig:
    """Fixed cycle assumptions, defaulting to the current notebook values.

    Source cells are noted per group.  ``from_namespace`` adapts these fields
    from a namespace produced by ``Turbinis-Calc/cycle_link.load_cycle`` so a
    loaded model's constants are used rather than silently replaced by the
    defaults below.
    """

    # --- Turbinis-Cycle.ipynb cell 3: atmosphere / station 0 ---------------
    T_a_full: float = 288.0          # T_а_полн, K (type 0: static = total)
    p_a: float = 101300.0            # p_а, Pa (static ambient)
    p_a_full: float = 101300.0       # p_а_полн, Pa (type 0: static = total)
    pi_v: float = 1.0                # π_v = p*в/p_а (type 0: M_v = 0)
    R: float = 287.4                 # J/(kg*K)
    k_air: float = 1.4               # air isentropic exponent
    v: float = 0.0                   # flight speed, m/s (type 0 stand)

    # --- Turbinis-Cycle.ipynb cell 2: loss/efficiency assumptions ----------
    sigma_in: float = 0.98           # σ_вх, inlet total-pressure recovery
    sigma_combustion: float = 0.96   # σ_г, combustor total-pressure recovery
    eta_compressor_poly: float = 0.85   # η_к_п_полн, centrifugal polytropic
    eta_turbine_poly: float = 0.90      # η_т_п_полн, axial polytropic
    eta_combustion: float = 0.98        # η_г, combustion completeness
    eta_mech_turbine: float = 0.99      # η_м_тур, referred to turbine power
    eta_reducer: float = 0.98           # η_р, reduction-gear efficiency

    # --- Turbinis-Cycle.ipynb cell 5: economics / nozzle --------------------
    g_leak: float = 0.0075           # g_ут (mean of 0.005...0.010)
    q_lhv_kj: float = 43150.0        # Q_н_р, kJ/kg (mean of 43600...42700)
    l0: float = 14.61                # l_0, kg air per kg kerosene
    M_t: float = 0.3379              # M_т, turbine-exit Mach number
    pi_nozzle: float = 1.05          # π_с = p_т/p_а
    phi_nozzle: float = 0.95         # φ_с, nozzle velocity coefficient
    T_ref: float = 288.0             # T_0, reference temperature for g_m
    eta_propeller: float = 0.82      # η_в (flight branch of N_e_уд)
    beta_prop: float = 0.015         # β_в, N/W (stand branch of N_e_уд)

    # --- Turbinis-Cycle.ipynb cell 5 / former cycle_point cooling rule -----
    # These branch values are literals in the current source, not independent
    # namespace variables; keeping them here makes the branch explicit.
    T_cool_threshold: float = 1250.0  # cooled = T_g > 1250.0
    g_cooling: float = 0.05           # g_охл when cooled
    g_cooling_off: float = 0.015      # g_охл when not cooled
    g_return_fraction: float = 0.6    # g_воз = 0.6 * g_охл

    def __post_init__(self) -> None:
        positive = (
            "T_a_full", "p_a", "p_a_full", "pi_v", "R", "q_lhv_kj", "l0",
            "pi_nozzle", "T_ref", "beta_prop",
        )
        for name in positive:
            value = _as_positive_float(name, getattr(self, name))
            object.__setattr__(self, name, value)

        k_air = _as_float("k_air", self.k_air)
        if k_air <= 1.0:
            raise ValueError(f"k_air must be > 1, got {self.k_air!r}")
        object.__setattr__(self, "k_air", k_air)

        for name in (
            "sigma_in", "sigma_combustion", "eta_compressor_poly",
            "eta_turbine_poly", "eta_combustion", "eta_mech_turbine",
            "eta_reducer", "eta_propeller", "phi_nozzle",
        ):
            value = _as_float(name, getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value!r}")
            object.__setattr__(self, name, value)

        for name in ("v", "M_t", "g_cooling", "g_cooling_off", "g_return_fraction"):
            value = _as_float(name, getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value!r}")
            object.__setattr__(self, name, value)

        g_leak = _as_float("g_leak", self.g_leak)
        if g_leak < 0.0 or g_leak >= 1.0:
            raise ValueError(f"g_leak must be in [0, 1), got {self.g_leak!r}")
        object.__setattr__(self, "g_leak", g_leak)

        T_cool_threshold = _as_positive_float(
            "T_cool_threshold", self.T_cool_threshold
        )
        object.__setattr__(self, "T_cool_threshold", T_cool_threshold)

    @classmethod
    def from_namespace(cls, namespace: Mapping[str, Any]) -> "CycleConfig":
        """Build a config from a loaded cycle namespace.

        The notebook's Russian identifiers are preferred; the field names are
        also accepted for programmatic use.  Every stable constant is required:
        a missing one raises ``KeyError`` instead of silently falling back to a
        default.  State-dependent cooling variables are intentionally not read
        here because ``cycle_point`` derives that branch from ``T_g``.
        """

        aliases = {
            "T_a_full": ("T_а_полн",),
            "p_a": ("p_а",),
            "p_a_full": ("p_а_полн",),
            "pi_v": ("π_v",),
            "R": ("R",),
            "k_air": ("k_в",),
            "sigma_in": ("σ_вх",),
            "sigma_combustion": ("σ_г",),
            "eta_compressor_poly": ("η_к_п_полн",),
            "eta_turbine_poly": ("η_т_п_полн",),
            "eta_combustion": ("η_г",),
            "eta_mech_turbine": ("η_м_тур",),
            "eta_reducer": ("η_р",),
            "g_leak": ("g_ут",),
            "q_lhv_kj": ("Q_н_р",),
            "l0": ("l_0",),
            "M_t": ("M_т",),
            "pi_nozzle": ("π_с",),
            "phi_nozzle": ("φ_с",),
            "T_ref": ("T_0",),
            "v": ("v",),
            "eta_propeller": ("η_в",),
            "beta_prop": ("β_в",),
        }
        values: dict[str, float] = {}
        for field_name, notebook_names in aliases.items():
            for key in (*notebook_names, field_name):
                if key in namespace:
                    values[field_name] = namespace[key]
                    break
            else:
                raise KeyError(
                    "cycle namespace is missing required constant "
                    f"{notebook_names[0]!r} for CycleConfig.{field_name}"
                )
        return cls(**values)


@dataclass(frozen=True)
class CycleState:
    """One requested cycle operating point."""

    pi: float          # compressor total pressure ratio
    T_g: float         # turbine inlet total temperature, K
    N: float           # requested gearbox-input shaft power, W (pre-reducer)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pi", _as_positive_float("pi", self.pi))
        object.__setattr__(self, "T_g", _as_positive_float("T_g", self.T_g))
        object.__setattr__(self, "N", _as_positive_float("N", self.N))


@dataclass(frozen=True)
class CycleResult:
    """Single-point cycle outputs, including the former ``calc_tvd`` fields."""

    pi: float
    T_g: float
    N: float
    G: float
    p_g: float
    p_t: float
    c_t: float
    g_m: float
    g_t: float
    alpha: float          # d["α"] = 1 / (g_m * l_0)
    L_k: float
    L_t: float
    pi_t: float           # d["π_т"]
    pi_c_full: float      # d["π_с_полн"]
    eta_t: float          # d["η_т"], used as eta_t_cycle
    T_t: float
    H_c_full: float       # d["H_с_полн"]
    c_c: float            # d["c_с"]
    N_ud: float           # d["N_в_уд"]: pre-reducer specific shaft power, J/kg
    N_reducer_out_ud: float  # N_ud * eta_reducer: downstream gearbox output
    R_ud: float           # d["R_уд"]
    N_e_ud: float         # equivalent specific power including nozzle thrust,
                          # built on N_reducer_out_ud (propeller/gearbox-output
                          # plane), not on the pre-reducer N_ud
    C_e: float            # fuel per pre-reducer shaft kW, kg/(kW*h)
    eta_e: float          # effective efficiency of the same pre-reducer plane
    g_cooling: float
    g_return: float
    cooled: bool


def _as_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def _as_positive_float(name: str, value: Any) -> float:
    result = _as_float(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0, got {result!r}")
    return result


DEFAULT_CONFIG = CycleConfig()


# ---------------------------------------------------------------------------
# Gas properties (notebook cell 3)
# ---------------------------------------------------------------------------
def calc_cp_kerosene(T_K: float, alpha: float = math.inf) -> float:
    """Mean cp of kerosene combustion products, J/(kg*K).

    ``alpha=inf`` gives dry air.  Formula and 700 K branch are copied verbatim
    from the notebook (calories -> joules via the 4187 factor).
    """

    T = _as_float("T_K", T_K)
    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"alpha must be a real number, got {alpha!r}") from exc
    if math.isnan(alpha_value) or alpha_value <= 0.0:
        raise ValueError(f"alpha must be > 0, got {alpha!r}")
    if T < 700.0:
        cpm = (
            (1.2 + 2.25 / alpha_value) / 1e5 * (T - 70.0) + 0.236
        ) * 4187.0
    else:
        cpm = (
            (2.2 + 1.25 / alpha_value) / 1e5 * (T + 450.0) + 0.218
        ) * 4187.0
    return cpm


def calc_k_kerosene(T_K: float, alpha: float, gas_constant: float) -> float:
    """Isentropic exponent k = cp / (cp - R), with R supplied by config."""

    cp = calc_cp_kerosene(T_K, alpha)
    denominator = cp - _as_positive_float("gas_constant", gas_constant)
    if denominator <= 0.0:
        raise CycleSolverError(
            f"invalid cp={cp!r} for R={gas_constant!r}: cp - R <= 0"
        )
    return cp / denominator


def calc_alpha(T_k: float, T_g: float) -> float:
    """Excess-air ratio polynomial from notebook cell 3 (Прил. 1)."""

    x = _as_float("T_k", T_k) / 1000.0
    y = _as_float("T_g", T_g) / 1000.0
    denominator = (
        -0.004149
        + 0.367730 * y
        - 0.011584 * y**2
        + 0.027054 * y**3
        - 0.398434 * x
        + 0.031502 * x * y
        - 0.022030 * x * y**2
        + 0.062033 * x**2
        - 0.003988 * x**2 * y
        - 0.049657 * x**3
    )
    if denominator == 0.0:
        raise CycleSolverError("calc_alpha denominator is zero")
    return 1.0 / denominator


# ---------------------------------------------------------------------------
# Scalar cycle pieces (notebook cells 3 and 5)
# ---------------------------------------------------------------------------
def _compressor_exit_temperature(
    T_in: float,
    pi: float,
    config: CycleConfig,
    tol: float,
    nmax: int,
) -> tuple[float, float]:
    """Variable-k polytropic compressor exit temperature and mean k.

    This is the single implementation of the existing compressor equation.
    Callers validate their own inputs.  The mean-gas temperature is
    ``0.5 * (T_in + T_out)`` and the initial guess uses ``config.k_air`` exactly
    as the notebook's cycle path does.
    """

    T_out = T_in * pi ** (
        (config.k_air - 1.0) / (config.k_air * config.eta_compressor_poly)
    )
    for _ in range(nmax):
        k = calc_k_kerosene(0.5 * (T_in + T_out), math.inf, config.R)
        T_out_new = T_in * pi ** (
            (k - 1.0) / (k * config.eta_compressor_poly)
        )
        if abs(T_out_new - T_out) < tol:
            return T_out_new, k
        T_out = T_out_new
    raise CycleSolverError(
        f"compressor iteration did not converge for pi={pi!r} "
        f"within {nmax} iterations (tol={tol!r})"
    )


def calc_compressor(
    pi: float,
    config: CycleConfig = DEFAULT_CONFIG,
    *,
    tol: float = SOLVER_TOL,
    nmax: int = SOLVER_MAX_ITER,
    T_in: float | None = None,
) -> tuple[float, float]:
    """Compressor exit total temperature and mean k (variable-k iteration).

    ``T_in`` is an optional keyword-only inlet total temperature.  Omitting it
    keeps the notebook/cycle anchor path bit-for-bit unchanged by using
    ``config.T_a_full``; supplying it applies the same equation from another
    inlet total temperature.
    """

    pi = _as_positive_float("pi", pi)
    nmax = _positive_int("nmax", nmax)
    tol = _as_float("tol", tol)
    if tol < 0.0:
        raise ValueError(f"tol must be >= 0, got {tol!r}")
    inlet_temperature = (
        config.T_a_full if T_in is None else _as_positive_float("T_in", T_in)
    )
    return _compressor_exit_temperature(
        inlet_temperature, pi, config, tol, nmax
    )


def calc_eta_expansion(pi: float, k_gas: float, eta_turbine_poly: float) -> float:
    """Turbine total-to-total efficiency from the polytropic efficiency."""

    pi = _as_positive_float("pi", pi)
    k_gas = _as_positive_float("k_gas", k_gas)
    eta_turbine_poly = _as_float("eta_turbine_poly", eta_turbine_poly)
    if not 0.0 < eta_turbine_poly <= 1.0:
        raise ValueError(
            f"eta_turbine_poly must be in (0, 1], got {eta_turbine_poly!r}"
        )
    denominator = 1.0 - pi ** ((1.0 - k_gas) / k_gas)
    if denominator == 0.0:
        raise CycleSolverError(
            f"eta expansion denominator is zero for pi={pi!r}, k={k_gas!r}"
        )
    return (
        1.0 - pi ** ((1.0 - k_gas) * eta_turbine_poly / k_gas)
    ) / denominator


def calc_g_m(
    T_k: float, T_g: float, config: CycleConfig = DEFAULT_CONFIG
) -> float:
    """Relative fuel flow g_m from notebook cell 5 (p. 20 formula)."""

    T_k = _as_positive_float("T_k", T_k)
    T_g = _as_positive_float("T_g", T_g)
    T_p = T_k
    c_air_g = calc_cp_kerosene(T_g)
    c_air_p = calc_cp_kerosene(T_p)
    c_air_0 = calc_cp_kerosene(config.T_ref)
    c_products_g = calc_cp_kerosene(T_g, alpha=1.0)
    c_products_0 = calc_cp_kerosene(config.T_ref, alpha=1.0)

    numerator = c_air_g * T_g - c_air_p * T_p
    denominator = (
        config.q_lhv_kj * 1e3 * config.eta_combustion
        - (1.0 + config.l0) * (c_products_g * T_g - c_products_0 * config.T_ref)
        + config.l0 * (c_air_g * T_g - c_air_0 * config.T_ref)
    )
    if denominator == 0.0:
        raise CycleSolverError("calc_g_m denominator is zero")
    return numerator / denominator


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# Single point
# ---------------------------------------------------------------------------
def calculate_cycle_point(
    state: CycleState, config: CycleConfig = DEFAULT_CONFIG
) -> CycleResult:
    """Evaluate one cycle point without notebook state or optimizer search."""

    pi = state.pi
    T_g = state.T_g
    N = state.N

    cooled = T_g > config.T_cool_threshold
    g_cooling = config.g_cooling if cooled else config.g_cooling_off
    g_return = config.g_return_fraction * g_cooling

    T_k, k_comp = calc_compressor(pi, config)
    L_k = config.R * k_comp / (k_comp - 1.0) * (T_k - config.T_a_full)
    alpha_combustion = calc_alpha(T_k, T_g)
    g_m = calc_g_m(T_k, T_g, config)
    g_t = (1.0 + g_m) * (1.0 - config.g_leak - g_cooling)

    pi_t = (
        config.sigma_in
        * config.sigma_combustion
        * pi
        * config.pi_v
        / config.pi_nozzle
    )
    for _ in range(SOLVER_MAX_ITER):
        T_avg = 0.5 * T_g * (1.0 + pi_t ** (-0.25))
        c_p_t = calc_cp_kerosene(T_avg, alpha_combustion)
        k_t = calc_k_kerosene(T_avg, alpha_combustion, config.R)
        pi_c_full = config.pi_nozzle * (
            1.0 + (k_t - 1.0) / 2.0 * config.M_t**2
        ) ** (k_t / (k_t - 1.0))
        pi_t_new = (
            config.sigma_in
            * config.sigma_combustion
            * pi
            * config.pi_v
            / pi_c_full
        )
        if abs(pi_t_new - pi_t) < SOLVER_TOL:
            pi_t = pi_t_new
            break
        pi_t = pi_t_new
    else:
        raise CycleSolverError(
            f"turbine expansion iteration did not converge for pi={pi!r} "
            f"within {SOLVER_MAX_ITER} iterations"
        )

    # c_p_t and k_t intentionally come from the last iterate, as in the
    # notebook: on convergence pi_t is updated once more before break.
    eta_t = calc_eta_expansion(pi_t, k_t, config.eta_turbine_poly)
    L_t = (
        c_p_t
        * T_g
        * (1.0 - pi_t ** ((1.0 - k_t) / k_t))
        * eta_t
    )
    T_t = T_g - L_t / c_p_t

    T_avg_nozzle = 0.5 * T_t * (1.0 + pi_c_full ** (-0.25))
    c_p_nozzle = calc_cp_kerosene(T_avg_nozzle, alpha_combustion)
    k_nozzle = calc_k_kerosene(T_avg_nozzle, alpha_combustion, config.R)
    H_c_full = c_p_nozzle * T_t * (
        1.0 - pi_c_full ** ((1.0 - k_nozzle) / k_nozzle)
    )
    c_c = config.phi_nozzle * math.sqrt(2.0 * H_c_full)

    g_sum = g_t + g_return
    # Requested target plane: mechanical shaft power at the gearbox input.
    # The reducer loss is applied only to the downstream useful shaft output.
    N_ud = L_t * g_sum * config.eta_mech_turbine - L_k
    N_reducer_out_ud = N_ud * config.eta_reducer
    R_ud = g_sum * (c_c - config.v)
    N_e_ud = N_reducer_out_ud + (
        R_ud / config.beta_prop
        if config.v == 0.0
        else R_ud * config.v / config.eta_propeller
    )

    p_g = config.sigma_combustion * config.sigma_in * config.p_a_full * pi
    p_t = config.pi_nozzle * config.p_a
    G = N / N_ud

    # Former cycle_point derived fields.  C_e is intentionally referred to the
    # same pre-reducer shaft-power plane as N (the user-confirmed target).
    alpha_gas = 1.0 / (g_m * config.l0)
    C_e = g_m * (1.0 - config.g_leak - g_cooling) * 3600.0 / (N_ud / 1e3)
    eta_e = 3600.0 / (C_e * config.q_lhv_kj)
    k_exit = calc_k_kerosene(T_t, alpha_gas, config.R)
    T_static = T_t / (1.0 + (k_exit - 1.0) / 2.0 * config.M_t**2)
    c_t = config.M_t * math.sqrt(k_exit * config.R * T_static)

    return CycleResult(
        pi=pi,
        T_g=T_g,
        N=N,
        G=G,
        p_g=p_g,
        p_t=p_t,
        c_t=c_t,
        g_m=g_m,
        g_t=g_t,
        alpha=alpha_gas,
        L_k=L_k,
        L_t=L_t,
        pi_t=pi_t,
        pi_c_full=pi_c_full,
        eta_t=eta_t,
        T_t=T_t,
        H_c_full=H_c_full,
        c_c=c_c,
        N_ud=N_ud,
        N_reducer_out_ud=N_reducer_out_ud,
        R_ud=R_ud,
        N_e_ud=N_e_ud,
        C_e=C_e,
        eta_e=eta_e,
        g_cooling=g_cooling,
        g_return=g_return,
        cooled=cooled,
    )


__all__ = [
    "CycleConfig",
    "CycleResult",
    "CycleSolverError",
    "CycleState",
    "DEFAULT_CONFIG",
    "SOLVER_MAX_ITER",
    "SOLVER_TOL",
    "calc_alpha",
    "calc_compressor",
    "calc_cp_kerosene",
    "calc_eta_expansion",
    "calc_g_m",
    "calc_k_kerosene",
    "calculate_cycle_point",
]
