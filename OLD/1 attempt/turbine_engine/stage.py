"""Stateless meanline calculation of one turbine stage.

PROVENANCE
==========
This module is a formula-preserving extraction of ``ступень`` from code cell 12
of ``Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb``.  It executes no notebook,
performs no stage iteration or optimisation, and keeps no mutable global state.

The notebook remains the independent legacy comparison path.  The constants in
:class:`StageConfig` default to the *current* notebook assumptions, but they are
documented as assumptions and can be replaced through
:meth:`StageConfig.from_namespace`.  Tests compare this core with observed legacy
outputs only as software-drift diagnostics, never as physical validation.

Gas properties are deliberately not recomputed here.  The notebook first calls
``cp_k_ступени(T0, H0)``; the extracted equivalent lives in
:mod:`turbine_engine.gas`, and the already-calculated ``cp`` and ``k`` are
passed explicitly in :class:`StageState`.  This keeps gas-property and stage
equations separately testable.

UNITS
=====
SI throughout except ``alpha1_deg`` in degrees, ``n`` in rpm, and the rotor
stress ``sigma_rotor`` in MPa (``K_n`` carries the unit conversion used by the
source).  Energies are J/kg, pressures Pa, temperatures K, lengths m, mass flows
kg/s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


class StageCalculationError(RuntimeError):
    """The stage equations are undefined for the supplied finite state.

    This mirrors the notebook's ``НетРешения`` stops (for example
    ``c1a >= c1`` or ``w2 <= c2a``) plus mathematically invalid square-root and
    inverse-sine arguments.  It is not an engineering acceptance filter.
    """


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


def _as_nonnegative_float(name: str, value: Any) -> float:
    result = _as_float(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be >= 0, got {result!r}")
    return result


def _checked_sqrt(name: str, value: float) -> float:
    try:
        return math.sqrt(value)
    except ValueError as exc:
        raise StageCalculationError(
            f"{name}: square root of negative value {value!r}"
        ) from exc


def _checked_asin(name: str, value: float) -> float:
    try:
        return math.asin(value)
    except ValueError as exc:
        raise StageCalculationError(
            f"{name}: inverse sine argument outside [-1, 1]: {value!r}"
        ) from exc


@dataclass(frozen=True)
class StageConfig:
    """Fixed stage assumptions.

    The defaults are the current notebook literals and are deliberately tagged
    as assumptions, not silently presented as book values.  Use
    :meth:`from_namespace` with a loaded model namespace to consume the actual
    constants (``R``, ``φ``, ``ψ``, ``δ_отн``, ``δ_мин``, ``ε_парц``, ``k_n``)
    instead of these defaults.
    """

    R: float = 287.4            # J/(kg*K), current cycle gas constant
    phi: float = 0.97           # φ, stator velocity coefficient (notebook assumption)
    psi: float = 0.96           # ψ, rotor velocity coefficient (notebook assumption)
    delta_rel: float = 0.005    # δ_отн = δ_л/l (notebook assumption)
    delta_min: float = 0.2e-3   # δ_мин, m (notebook assumption)
    epsilon_parc: float = 1.0   # ε_парц, partial-admission degree (notebook assumption)
    K_n: float = 6.8e-6         # k_n, blade-form factor for σ_р = K_n n² A

    def __post_init__(self) -> None:
        object.__setattr__(self, "R", _as_positive_float("R", self.R))
        for name in ("phi", "psi"):
            value = _as_float(name, getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value!r}")
            object.__setattr__(self, name, value)

        object.__setattr__(
            self, "delta_rel", _as_nonnegative_float("delta_rel", self.delta_rel)
        )
        object.__setattr__(
            self, "delta_min", _as_nonnegative_float("delta_min", self.delta_min)
        )

        epsilon_parc = _as_float("epsilon_parc", self.epsilon_parc)
        if not 0.0 <= epsilon_parc <= 1.0:
            raise ValueError(
                f"epsilon_parc must be in [0, 1], got {epsilon_parc!r}"
            )
        object.__setattr__(self, "epsilon_parc", epsilon_parc)
        object.__setattr__(self, "K_n", _as_positive_float("K_n", self.K_n))

    @classmethod
    def from_namespace(cls, namespace: Mapping[str, Any]) -> "StageConfig":
        """Build a config from a loaded model namespace.

        Notebook identifiers are preferred; the field names are accepted too.
        Every constant is required so a missing value cannot silently fall back
        to a default assumption.
        """

        aliases = {
            "R": ("R",),
            "phi": ("φ", "phi"),
            "psi": ("ψ", "psi"),
            "delta_rel": ("δ_отн", "delta_rel"),
            "delta_min": ("δ_мин", "delta_min"),
            "epsilon_parc": ("ε_парц", "epsilon_parc"),
            "K_n": ("k_n", "K_n"),
        }
        values: dict[str, Any] = {}
        for field_name, notebook_names in aliases.items():
            for key in (*notebook_names, field_name):
                if key in namespace:
                    values[field_name] = namespace[key]
                    break
            else:
                raise KeyError(
                    "stage namespace is missing required constant "
                    f"{notebook_names[0]!r} for StageConfig.{field_name}"
                )
        return cls(**values)


@dataclass(frozen=True)
class StageState:
    """One fully specified stage operating point.

    ``cp`` and ``k`` must already be evaluated by the caller (in the notebook
    they come from ``cp_k_ступени(T0, H0)``).
    """

    T0: float           # stage-inlet total temperature, K
    p0: float           # stage-inlet total pressure, Pa
    H0: float           # available isentropic heat drop, J/kg
    rho: float          # reaction degree ρ, 0...1
    D_c: float          # stator mean diameter, m
    D_l: float          # rotor mean diameter, m
    n: float            # rotor speed, rpm
    G_c: float          # stator mass flow, kg/s
    G_l: float          # rotor mass flow, kg/s
    alpha1_deg: float   # stator exit angle α1, degrees (design input)
    kappa: float        # κ = c2a/c1a
    cp: float           # mean specific heat, J/(kg*K)
    k: float            # isentropic exponent

    def __post_init__(self) -> None:
        for name in (
            "T0", "p0", "H0", "D_c", "D_l", "n", "G_c", "G_l", "cp",
        ):
            object.__setattr__(
                self, name, _as_positive_float(name, getattr(self, name))
            )

        rho = _as_float("rho", self.rho)
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"rho must be in [0, 1], got {rho!r}")
        object.__setattr__(self, "rho", rho)

        alpha1_deg = _as_float("alpha1_deg", self.alpha1_deg)
        if not 0.0 < alpha1_deg < 180.0:
            raise ValueError(
                f"alpha1_deg must be in (0, 180), got {alpha1_deg!r}"
            )
        object.__setattr__(self, "alpha1_deg", alpha1_deg)

        object.__setattr__(self, "kappa", _as_positive_float("kappa", self.kappa))

        k = _as_float("k", self.k)
        if k <= 1.0:
            raise ValueError(f"k must be > 1, got {k!r}")
        object.__setattr__(self, "k", k)


@dataclass(frozen=True)
class StageResult:
    """Scalar outputs of one stage, in source order and units.

    :meth:`as_legacy_dict` converts to the exact dictionary returned by the
    notebook ``ступень`` function.
    """

    T0: float
    p0: float
    H0: float
    rho: float
    Hc: float
    cp: float
    k: float
    c1: float
    T1: float
    T1_ad: float
    p1: float
    A1a: float
    v1: float
    c1a: float
    alpha1_deg: float
    c1u: float
    w1: float
    beta1_deg: float
    H_l: float
    u1: float
    u2: float
    w2: float
    T2: float
    T2_ad: float
    p2: float
    v2: float
    A_la: float
    c2a: float
    beta2_deg: float
    c2u: float
    alpha2_deg: float
    c2: float
    L_u: float
    eta_u: float
    h_c: float
    xi_c: float
    xi_c_prime: float
    h_l: float
    xi_l: float
    h_v: float
    xi_v: float
    eta_u2: float
    delta_l: float
    h_z: float
    xi_z: float
    L_u3: float
    eta_t1: float
    eta_l1: float
    N_tv: float
    h_tv: float
    xi_tv: float
    eta_t: float
    eta_l: float
    L_t: float
    T_stat: float
    T_total: float
    p_total: float
    H0_next: float
    eta_t_total: float
    T_w1: float
    M_c1: float
    M_w2: float
    sigma_rotor: float
    l_c: float
    l_l: float
    D_c: float
    D_l: float
    G_c: float
    G_l: float
    pi_stage: float

    def as_legacy_dict(self) -> dict[str, float]:
        """Return the exact key set of the notebook ``ступень`` result."""

        return {
            legacy_key: getattr(self, field_name)
            for field_name, legacy_key in _LEGACY_FIELDS
        }

    @classmethod
    def from_legacy_dict(cls, values: Mapping[str, Any]) -> "StageResult":
        """Build a result from the notebook ``ступень`` dictionary.

        This is the inverse of :meth:`as_legacy_dict` and is used by adapters
        that receive already-computed legacy stage dictionaries.  Every legacy
        key is required; no value is guessed or defaulted.
        """

        missing = [key for _, key in _LEGACY_FIELDS if key not in values]
        if missing:
            raise KeyError(
                "legacy stage dictionary is missing required keys: "
                + ", ".join(repr(key) for key in missing)
            )
        return cls(**{field_name: values[key] for field_name, key in _LEGACY_FIELDS})


_LEGACY_FIELDS: tuple[tuple[str, str], ...] = (
    ("T0", "T0"),
    ("p0", "p0"),
    ("H0", "H0"),
    ("rho", "ρ"),
    ("Hc", "Hc"),
    ("cp", "cp"),
    ("k", "k"),
    ("c1", "c1"),
    ("T1", "T1"),
    ("T1_ad", "T1_ад"),
    ("p1", "p1"),
    ("A1a", "A1a"),
    ("v1", "v1"),
    ("c1a", "c1a"),
    ("alpha1_deg", "α1"),
    ("c1u", "c1u"),
    ("w1", "w1"),
    ("beta1_deg", "β1"),
    ("H_l", "Hл"),
    ("u1", "u1"),
    ("u2", "u2"),
    ("w2", "w2"),
    ("T2", "T2"),
    ("T2_ad", "T2_ад"),
    ("p2", "p2"),
    ("v2", "v2"),
    ("A_la", "A_лa"),
    ("c2a", "c2a"),
    ("beta2_deg", "β2"),
    ("c2u", "c2u"),
    ("alpha2_deg", "α2"),
    ("c2", "c2"),
    ("L_u", "L_u"),
    ("eta_u", "η_u"),
    ("h_c", "h_c"),
    ("xi_c", "ξ_c"),
    ("xi_c_prime", "ξ_c_"),
    ("h_l", "h_л"),
    ("xi_l", "ξ_л"),
    ("h_v", "h_в"),
    ("xi_v", "ξ_в"),
    ("eta_u2", "η_u2"),
    ("delta_l", "δ_л"),
    ("h_z", "h_з"),
    ("xi_z", "ξ_з"),
    ("L_u3", "L_u3"),
    ("eta_t1", "η_т1"),
    ("eta_l1", "η_л1"),
    ("N_tv", "N_тв"),
    ("h_tv", "h_тв"),
    ("xi_tv", "ξ_тв"),
    ("eta_t", "η_т"),
    ("eta_l", "η_л"),
    ("L_t", "L_т"),
    ("T_stat", "T_ст"),
    ("T_total", "T_ст_п"),
    ("p_total", "p_ст_п"),
    ("H0_next", "H0_п"),
    ("eta_t_total", "η_т_п"),
    ("T_w1", "T_w1"),
    ("M_c1", "M_c1"),
    ("M_w2", "M_w2"),
    ("sigma_rotor", "σ_р"),
    ("l_c", "l_c"),
    ("l_l", "l_л"),
    ("D_c", "D_c"),
    ("D_l", "D_л"),
    ("G_c", "G_c"),
    ("G_l", "G_л"),
    ("pi_stage", "π_ст"),
)


DEFAULT_STAGE_CONFIG = StageConfig()


def calculate_stage(
    state: StageState, config: StageConfig = DEFAULT_STAGE_CONFIG
) -> StageResult:
    """Evaluate one stage exactly as notebook cell 12, with precomputed cp/k.

    Engineering quantities such as ``alpha1``, blade lengths and rotor stress are
    outputs only; they are never used as acceptance filters here.
    """

    T0 = state.T0
    p0 = state.p0
    H0 = state.H0
    rho = state.rho
    D_c = state.D_c
    D_l = state.D_l
    n = state.n
    G_c = state.G_c
    G_l = state.G_l
    alpha1_deg = state.alpha1_deg
    kappa = state.kappa
    cp = state.cp
    k = state.k

    phi = config.phi
    psi = config.psi
    delta_rel = config.delta_rel
    delta_min = config.delta_min
    epsilon_parc = config.epsilon_parc
    K_n = config.K_n
    R = config.R

    # ---------- stator -----------------------------------------------------
    Hc = H0 * (1.0 - rho)
    c1 = phi * _checked_sqrt("stator exit velocity c1", 2.0 * Hc)
    T1 = T0 - Hc * phi**2 / cp
    T1_ad = T0 - Hc / cp
    if T1 <= 0.0:
        raise StageCalculationError(f"stator exit temperature T1 is not positive: {T1!r}")
    if T1_ad <= 0.0:
        raise StageCalculationError(
            f"stator isentropic exit temperature T1_ad is not positive: {T1_ad!r}"
        )
    p1 = p0 * (T1_ad / T0) ** (k / (k - 1))
    if p1 <= 0.0:
        raise StageCalculationError(f"stator exit pressure p1 is not positive: {p1!r}")
    v1 = R * T1 / p1

    sin_alpha1 = math.sin(math.radians(alpha1_deg))
    denominator_l_c = math.pi * D_c * sin_alpha1 * c1
    if denominator_l_c == 0.0:
        raise StageCalculationError("stator blade length denominator is zero")
    l_c = G_c * v1 / denominator_l_c
    A1a = math.pi * D_c * l_c
    if A1a == 0.0:
        raise StageCalculationError("stator exit area A1a is zero")
    c1a = G_c * v1 / A1a
    if c1a >= c1:
        raise StageCalculationError("c1a >= c1")
    alpha1 = _checked_asin("stator flow angle α1", c1a / c1)
    c1u = c1 * math.cos(alpha1)

    # ---------- rotor inlet ------------------------------------------------
    u1 = math.pi * D_c * n / 60.0
    u2 = math.pi * D_l * n / 60.0
    w1 = _checked_sqrt(
        "rotor inlet relative velocity w1",
        c1**2 + u1**2 - 2.0 * c1 * u1 * math.cos(alpha1),
    )
    beta1 = math.atan2(c1a, c1u - u1)
    H_l = rho * H0 * T1 / T1_ad

    # ---------- rotor exit -------------------------------------------------
    w2 = psi * _checked_sqrt(
        "rotor exit relative velocity w2",
        w1**2 + 2.0 * H_l + u2**2 - u1**2,
    )
    T2 = T1 - (w2**2 - w1**2 - u2**2 + u1**2) / (2.0 * cp)
    if T2 <= 0.0:
        raise StageCalculationError(f"rotor exit temperature T2 is not positive: {T2!r}")
    T2_ad = T1 - H_l / cp
    if T2_ad <= 0.0:
        raise StageCalculationError(
            f"rotor isentropic exit temperature T2_ad is not positive: {T2_ad!r}"
        )
    p2 = p1 * (T2_ad / T1) ** (k / (k - 1))
    if p2 <= 0.0:
        raise StageCalculationError(f"rotor exit pressure p2 is not positive: {p2!r}")
    v2 = R * T2 / p2
    c2a = kappa * c1a
    denominator_l_l = c2a * math.pi * D_l
    if denominator_l_l == 0.0:
        raise StageCalculationError("rotor blade length denominator is zero")
    l_l = G_l * v2 / denominator_l_l
    A_la = math.pi * D_l * l_l
    if w2 <= c2a:
        raise StageCalculationError("w2 <= c2a")
    beta2 = _checked_asin("rotor exit flow angle β2", c2a / w2)
    c2u = w2 * math.cos(beta2) - u2
    alpha2 = math.degrees(math.atan2(c2a, c2u))
    c2 = math.hypot(c2a, c2u)

    # ---------- work and losses -------------------------------------------
    L_u = c1u * u1 + c2u * u2
    eta_u = L_u / H0
    h_c = (phi**-2 - 1.0) * c1**2 / 2.0
    xi_c = h_c / H0
    xi_c_prime = xi_c * T2_ad / T1
    h_l = (psi**-2 - 1.0) * w2**2 / 2.0
    xi_l = h_l / H0
    h_v = c2**2 / 2.0
    xi_v = h_v / H0
    eta_u2 = 1.0 - xi_c_prime - xi_l - xi_v
    delta_l = max(delta_rel * l_l, delta_min)
    D_mean = 0.5 * (D_c + D_l)
    if D_mean == 0.0:
        raise StageCalculationError("mean diameter D_mean is zero")
    if l_l == 0.0:
        raise StageCalculationError("rotor blade length l_l is zero")
    h_z = 1.37 * (1.0 + 1.6 * rho) * (1.0 + l_l / D_mean) * (delta_l / l_l) * L_u
    xi_z = h_z / H0
    L_u3 = L_u - h_z
    eta_t1 = eta_u - xi_z
    eta_l1 = eta_t1 + xi_v
    rho_g = 0.5 * (1.0 / v1 + 1.0 / v2)
    u_mean = 0.5 * (u1 + u2)
    N_tv = (
        1.07 * D_mean**2 + 61.0 * (1.0 - epsilon_parc) * D_mean * l_l
    ) * (u_mean / 100.0) ** 3 * rho_g * 1e3
    h_tv = N_tv / G_c
    xi_tv = h_tv / H0
    eta_t = eta_t1 - xi_tv
    eta_l = eta_l1 - xi_tv
    L_t = H0 * eta_t

    # ---------- outlet state ----------------------------------------------
    T_stat = T2 + (h_z + h_tv) / cp
    if T_stat <= 0.0:
        raise StageCalculationError(
            f"stage static exit temperature T_stat is not positive: {T_stat!r}"
        )
    T_total = T_stat + h_v / cp
    p_total = p2 * (T_total / T_stat) ** (k / (k - 1))
    H0_next = cp * T0 * (1.0 - (p_total / p0) ** ((k - 1.0) / k))
    if H0_next == 0.0:
        raise StageCalculationError("next-stage available heat drop H0_next is zero")
    eta_t_total = L_t / H0_next
    T_w1 = T1 + w1**2 / (2.0 * cp)
    M_c1 = c1 / _checked_sqrt("stator Mach denominator", k * R * T1)
    M_w2 = w2 / _checked_sqrt("rotor Mach denominator", k * R * T2)
    sigma_rotor = K_n * n**2 * A_la
    pi_stage = p0 / p2

    return StageResult(
        T0=T0,
        p0=p0,
        H0=H0,
        rho=rho,
        Hc=Hc,
        cp=cp,
        k=k,
        c1=c1,
        T1=T1,
        T1_ad=T1_ad,
        p1=p1,
        A1a=A1a,
        v1=v1,
        c1a=c1a,
        alpha1_deg=math.degrees(alpha1),
        c1u=c1u,
        w1=w1,
        beta1_deg=math.degrees(beta1),
        H_l=H_l,
        u1=u1,
        u2=u2,
        w2=w2,
        T2=T2,
        T2_ad=T2_ad,
        p2=p2,
        v2=v2,
        A_la=A_la,
        c2a=c2a,
        beta2_deg=math.degrees(beta2),
        c2u=c2u,
        alpha2_deg=alpha2,
        c2=c2,
        L_u=L_u,
        eta_u=eta_u,
        h_c=h_c,
        xi_c=xi_c,
        xi_c_prime=xi_c_prime,
        h_l=h_l,
        xi_l=xi_l,
        h_v=h_v,
        xi_v=xi_v,
        eta_u2=eta_u2,
        delta_l=delta_l,
        h_z=h_z,
        xi_z=xi_z,
        L_u3=L_u3,
        eta_t1=eta_t1,
        eta_l1=eta_l1,
        N_tv=N_tv,
        h_tv=h_tv,
        xi_tv=xi_tv,
        eta_t=eta_t,
        eta_l=eta_l,
        L_t=L_t,
        T_stat=T_stat,
        T_total=T_total,
        p_total=p_total,
        H0_next=H0_next,
        eta_t_total=eta_t_total,
        T_w1=T_w1,
        M_c1=M_c1,
        M_w2=M_w2,
        sigma_rotor=sigma_rotor,
        l_c=l_c,
        l_l=l_l,
        D_c=D_c,
        D_l=D_l,
        G_c=G_c,
        G_l=G_l,
        pi_stage=pi_stage,
    )


__all__ = [
    "DEFAULT_STAGE_CONFIG",
    "StageCalculationError",
    "StageConfig",
    "StageResult",
    "StageState",
    "calculate_stage",
]
