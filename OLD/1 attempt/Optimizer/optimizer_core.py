"""Ядро параметрической оптимизации ТВД.

Использует существующую расчётную модель проекта:
  Turbinis-Cycle/Turbinis-Cycle.ipynb  - цикл;
  Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb - поступенчатый расчёт.

Новая физическая модель не создаётся: код ячеек блокнота исполняется так же,
как в mathcad/make_mathcad.py и podbor_parametrov.py. Здесь добавлены только:
  * пересчёт цикла для произвольных pi*к, T*г, N;
  * расчёт ступени чистым ядром turbine_engine.stage (ячейка блокнота остаётся
    независимым эталоном и по-прежнему исполняется);
  * независимая валидация решения;
  * численная оптимизация (DE / случайный поиск / локальная доводка);
  * исправление неустойчивого внешнего итератора по расходу G (см. _patch_cell16).

Единицы СИ. Углы в градусах, где так принято в исходной модели.
"""
from __future__ import annotations

import copy
import json
import math
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

ПРОЕКТ = Path(__file__).resolve().parent.parent
ПАПКА_CALC = ПРОЕКТ / "Turbinis-Calc"
ПАПКА_CYCLE = ПРОЕКТ / "Turbinis-Cycle"
ПАПКА_РЕЗУЛЬТАТОВ = Path(__file__).resolve().parent / "results"
ПАПКА_РЕЗУЛЬТАТОВ.mkdir(exist_ok=True)

# единый источник книжных ограничений и оцифрованных рис. 18/19
sys.path.insert(0, str(ПАПКА_CALC))
import constraints as cs  # noqa: E402

# Новое расчётное ядро цикла (без исполнения блокнота и без глобального состояния)
if str(ПРОЕКТ) not in sys.path:
    sys.path.insert(0, str(ПРОЕКТ))
from turbine_engine.cycle import (  # noqa: E402
    ALPHA_BASIS_LEGACY,
    CYCLE_GAS_PROPERTY_BASES,
    CYCLE_GAS_PROPERTY_BASIS_LEGACY,
    CYCLE_GAS_PROPERTY_BASIS_TEXTBOOK,
    CycleConfig,
    CycleState,
    calculate_cycle_point,
)
from turbine_engine.baseline import (  # noqa: E402
    calculate_baseline_turbine,
)
from turbine_engine.expansion import (  # noqa: E402
    calculate_turbine_expansion,
)
from turbine_engine.geometry import (  # noqa: E402
    MeridionalGeometryConfig,
    calculate_meridional_geometry,
)
from turbine_engine.gas import (  # noqa: E402
    GAS_PROPERTY_BASES,
    GAS_PROPERTY_BASIS_LEGACY,
    calc_gas_cp_k,
    calc_h0_for_pressure,
    calc_stage_cp_k,
    calc_textbook_gas_cp_k,
)
from turbine_engine.performance import (  # noqa: E402
    calculate_turbine_performance,
)
from turbine_engine.reaction import calculate_root_reaction  # noqa: E402
from turbine_engine.stage import (  # noqa: E402
    StageCalculationError,
    StageConfig,
    StageResult,
    StageState,
    calculate_stage,
)

# ---------------------------------------------------------------------------
# Исправление внешнего итератора расхода G в ячейке 16.
#
# В исходном коде N_на_валу(G, pi0) умножает глобальные G_c1, G_л_ост, A_т0
# на k = G/G_в. При повторных вызовах (метод секущих) это накапливается,
# A_т0 уходит в минус и math.sqrt даёт domain error. При N = 110 кВт секущих
# хватало, при масштабировании N ошибка проявляется.
# Исправление: каждый вызов N_of(G) начинается с одного и того же базового
# состояния, а G ограничивается положительным диапазоном. Формулы не меняются.
# ---------------------------------------------------------------------------
_OLD_G = '''G_в_цикл = G_в
π0 = π_т ** (1 / z)  # первое приближение: равные π
G_a, G_b = G_в_цикл, 1.05 * G_в_цикл
N_a, _r = N_на_валу(G_a, π0)
for _ in range(20):  # метод секущих: N_в почти линейна по G
    N_b, _r = N_на_валу(G_b, _r["π0"])
    if abs(N_b - N_зад) < 1e-5 * N_зад:
        break
    G_a, G_b, N_a = G_b, G_b + (N_зад - N_b) * (G_b - G_a) / (N_b - N_a), N_b
'''
_NEW_G = '''G_в_цикл = G_в
π0 = π_т ** (1 / z)
G0, Gc10, Gl0, A0 = G_в, G_c1, G_л_ост, A_т0
def _N_of(G):
    global G_в, G_c1, G_л_ост, A_т0
    G_в, G_c1, G_л_ост, A_т0 = G0, Gc10, Gl0, A0
    return N_на_валу(G, π0)
G_a, G_b = 0.9 * G0, 1.1 * G0
N_a, _r = _N_of(G_a)
N_b, _r = _N_of(G_b)
for _ in range(20):
    if abs(N_b - N_зад) < 1e-5 * N_зад:
        break
    den = N_b - N_a
    G_new = 0.5 * (G_a + G_b) if abs(den) < 1e-9 else G_b + (N_зад - N_b) * (G_b - G_a) / den
    G_new = min(max(G_new, 0.5 * G0), 2.0 * G0)
    G_a, G_b, N_a = G_b, G_new, N_b
    N_b, _r = _N_of(G_b)
'''


def _код(ячейка):
    return "".join(s for s in ячейка["source"] if not s.lstrip().startswith(("%", "!")))


@dataclass
class Model:
    nb: dict
    code: dict[int, str]
    param_idx: int
    core_idx: list[int]
    full_idx: list[int]
    base: dict
    cycle: dict
    patched: bool


def load_model(calc_dir: Path | str | None = None, quiet: bool = True) -> Model:
    import sys

    calc_dir = Path(calc_dir or ПАПКА_CALC)
    sys.path.insert(0, str(calc_dir))
    from cycle_link import load_cycle

    nb = json.loads((calc_dir / "Main_Turbinis_Calc_Book.ipynb").read_text(encoding="utf-8"))
    code = {i: _код(c) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"}
    param_idx = next(i for i, s in code.items() if "D_ст = [" in s and "l_c1_мин" in s)
    check_idx = next(i for i, s in code.items() if "проверки = []" in s)
    core_markers = ("def газ(", "π_т = p_г_полн", "def ступень(", "def ступень_авто(",
                    "def турбина(", "N_вал = (")
    full_markers = ("def γ_участка(", "def γ_установки(", "def отставание(")
    core_idx = [i for i, s in code.items() if param_idx < i < check_idx and any(m in s for m in core_markers)]
    full_idx = [i for i, s in code.items() if param_idx < i < check_idx and any(m in s for m in full_markers)]

    patched = False
    for i in core_idx:
        if "N_на_валу(G_a, π0)" in code[i] and _OLD_G in code[i]:
            code[i] = code[i].replace(_OLD_G, _NEW_G)
            patched = True
            break

    base = {"display": lambda *a, **k: None}
    for i in (1, 3):
        exec(compile(code[i], f"ячейка {i}", "exec"), base)
    cycle = load_cycle(quiet=quiet)
    return Model(nb, code, param_idx, sorted(core_idx), sorted(full_idx), base, cycle, patched)


# ---------------------------------------------------------------------------
# Цикл для произвольных pi*к, T*г, N
# ---------------------------------------------------------------------------
def cycle_point(m: Model, pi: float = 11.0, T_g: float = 1250.0,
                N: float = 110000.0, *,
                gas_property_basis: str = CYCLE_GAS_PROPERTY_BASIS_LEGACY) -> dict[str, Any]:
    """Пересчитывает цикл через stateless-ядро ``turbine_engine.cycle``.

    Публичная сигнатура и состав результата сохранены (добавлено явное поле
    ``N_reducer_out_ud``/``N_ред_уд`` для выхода редуктора). Константы берутся
    из фактически загруженного namespace ``m.cycle``, поэтому изменения
    блокнота не игнорируются. Состояние ``m.cycle`` не изменяется.

    ``gas_property_basis`` — явная keyword-only опция выбора метода цикла.
    Значение по умолчанию ``legacy_mean_cp`` сохраняет исходную
    notebook-аппроксимацию среднего/производного ``cp`` и старую форму
    результата. ``textbook_true_cp`` — диагностический вариант с опубликованным
    прямым fit; для него дополнительно возвращаются фактические
    ``cycle_gas_property_basis``, ``cycle_alpha_basis`` и
    ``cycle_method_diagnostics``. Неизвестное значение отклоняется ядром
    :func:`calculate_cycle_point` до физического расчёта.

    ``N`` — мощность на входе редуктора (до потерь редуктора). ``N_ud`` —
    удельная мощность на той же плоскости; ``N_reducer_out_ud = N_ud*η_р`` —
    отдельная downstream-величина на выходе редуктора. ``C_e``/``eta_e``
    отнесены к входной (pre-reducer) плоскости.
    """
    config = CycleConfig.from_namespace(m.cycle)
    state = CycleState(pi=float(pi), T_g=float(T_g), N=float(N))
    if gas_property_basis == CYCLE_GAS_PROPERTY_BASIS_LEGACY:
        # Keep the historical call shape for existing adapter/mock seams; the
        # keyword is only passed when the caller explicitly opts in.
        r = calculate_cycle_point(state, config)
    else:
        r = calculate_cycle_point(
            state, config, gas_property_basis=gas_property_basis
        )
    d = {
        "g_m": r.g_m, "g_т": r.g_t, "α": r.alpha,
        "L_к": r.L_k, "L_т": r.L_t, "π_т": r.pi_t,
        "π_с_полн": r.pi_c_full, "η_т": r.eta_t, "T_т": r.T_t,
        "H_с_полн": r.H_c_full, "c_с": r.c_c,
        "N_в_уд": r.N_ud, "N_ред_уд": r.N_reducer_out_ud,
        "R_уд": r.R_ud, "N_e_уд": r.N_e_ud,
    }
    result = {
        "pi": r.pi, "T_g": r.T_g, "N": r.N,
        "G": r.G, "p_g": r.p_g, "p_t": r.p_t, "c_t": r.c_t,
        "g_t": r.g_t, "alpha_g": r.alpha,
        "L_k": r.L_k, "L_t": r.L_t, "T_t": r.T_t,
        "eta_t_cycle": r.eta_t, "N_ud": r.N_ud,
        "N_reducer_out_ud": r.N_reducer_out_ud,
        "C_e": r.C_e, "eta_e": r.eta_e,
        "g_ohl": r.g_cooling, "g_voz": r.g_return, "охлаждение": r.cooled,
        "d": d,
    }
    if r.gas_property_basis != CYCLE_GAS_PROPERTY_BASIS_LEGACY:
        result["cycle_gas_property_basis"] = r.gas_property_basis
        result["cycle_alpha_basis"] = r.alpha_basis
        result["cycle_method_diagnostics"] = r.method_diagnostics
    return result


# ---------------------------------------------------------------------------
# Вектор переменных
# ---------------------------------------------------------------------------
БАЗА = {
    "D": [0.67, 1.0, 1.0], "lD": 0.250586,
    "a1": [9.25, 11.0, 19.0], "w": [0.91, 0.89, 1.0],
    "k": [1.38, 1.12, 1.15], "sigma": 290.0, "phi": 0.97, "psi": 0.96,
}
КНИГА = {
    "D": [0.915, 0.97, 1.0], "lD": 0.26,
    "a1": [11.2, 13.8, 19.5], "w": [0.86, 0.87, 1.0],
    "k": [1.12, 1.04, 1.02], "sigma": 290.0, "phi": 0.965, "psi": 0.95,
}


def _имена(cfg):
    names = ["D1", "D2", "lD", "a11", "a12", "a13", "w1", "w2"]
    if cfg.get("vary_kappa", True):
        names += ["k1", "k2", "k3"]
    if cfg.get("vary_sigma", True):
        names += ["sigma"]
    if cfg.get("mode") in ("B", "C"):
        names += ["pi"]
    if cfg.get("mode") == "C":
        names += ["N"]
    if cfg.get("vary_tg", False):
        names += ["Tg"]
    return names


def границы(cfg):
    b = {
        # lD: широкий поиск 1/9…1/3 (таблица с. 58); книжная рекомендация
        # последней ступени 0.25…1/3 (с. 58 и с. 87) — проверяется отдельно,
        # но не сужает поиск.
        "D1": (0.50, 1.00), "D2": (0.60, 1.00), "lD": (1.0 / 9.0, 1.0 / 3.0),
        # α1: книга с. 35/73 — малые турбины 11…14°; ниже 11° не ищем
        "a11": (11.0, 20.0), "a12": (11.0, 20.0), "a13": (11.0, 24.0),
        "w1": (0.60, 1.20), "w2": (0.60, 1.20),
        # κ > 1 — проектное условие l_л1=l_л2 (с. 62); ниже 1.0 не ищем
        "k1": (1.00, 1.40), "k2": (1.00, 1.30), "k3": (1.00, 1.30),
        # σ: книга с. 57 — 120…200 обычные, ≤300 особо напряжённые; 400 не книга
        "sigma": (120.0, 300.0), "pi": (6.0, 18.0), "N": (80000.0, 500000.0),
        "Tg": (1100.0, 1450.0),
    }
    return np.array([b[n] for n in _имена(cfg)], dtype=float)


def warm_start(cfg):
    src = КНИГА if (cfg.get("mode") in ("B", "C") or cfg.get("warm") == "book") else БАЗА
    d = dict(src)
    x = []
    for n in _имена(cfg):
        if n == "D1": x.append(d["D"][0])
        elif n == "D2": x.append(d["D"][1])
        elif n == "lD": x.append(d["lD"])
        elif n == "a11": x.append(d["a1"][0])
        elif n == "a12": x.append(d["a1"][1])
        elif n == "a13": x.append(d["a1"][2])
        elif n == "w1": x.append(d["w"][0])
        elif n == "w2": x.append(d["w"][1])
        elif n == "k1": x.append(d["k"][0])
        elif n == "k2": x.append(d["k"][1])
        elif n == "k3": x.append(d["k"][2])
        elif n == "sigma": x.append(d["sigma"])
        elif n == "pi": x.append(cfg.get("pi", 11.0))
        elif n == "N": x.append(cfg.get("N", 460000.0))
        elif n == "Tg": x.append(cfg.get("T_g", 1250.0))
    return np.array(x, dtype=float)


def декодировать(x, cfg):
    d = dict(zip(_имена(cfg), map(float, x)))
    D = [d["D1"], d["D2"], 1.0]
    k = [d.get("k1", КНИГА["k"][0]), d.get("k2", КНИГА["k"][1]), d.get("k3", КНИГА["k"][2])]
    return {
        "D": D, "lD": d["lD"], "a1": [d["a11"], d["a12"], d["a13"]],
        "w": [d["w1"], d["w2"], 1.0], "k": k,
        "sigma": d.get("sigma", КНИГА["sigma"]), "phi": cfg.get("phi", 0.965), "psi": cfg.get("psi", 0.95),
        "pi": d.get("pi", cfg.get("pi", 11.0)), "N": d.get("N", cfg.get("N", 110000.0)),
        "Tg": d.get("Tg", cfg.get("T_g", 1250.0)),
    }


# ---------------------------------------------------------------------------
# Один расчёт
# ---------------------------------------------------------------------------
GAS_PROPERTY_BASIS_ATTR = "gas_property_basis_build"
CYCLE_GAS_PROPERTY_BASIS_ATTR = "cycle_gas_property_basis_build"
CYCLE_ALPHA_BASIS_ATTR = "cycle_alpha_basis_build"
CYCLE_METHOD_DIAGNOSTICS_ATTR = "cycle_method_diagnostics_build"


def _validate_gas_property_basis(value: str) -> str:
    """Return a valid property-basis selector or raise ``ValueError``."""

    if value not in GAS_PROPERTY_BASES:
        raise ValueError(
            f"unknown gas_property_basis {value!r}; "
            f"expected one of {GAS_PROPERTY_BASES!r}"
        )
    return value


def _validate_cycle_gas_property_basis(value: str) -> str:
    """Return a valid cycle property-basis selector or raise ``ValueError``."""

    if not isinstance(value, str) or value not in CYCLE_GAS_PROPERTY_BASES:
        raise ValueError(
            f"unknown cycle gas_property_basis {value!r}; "
            f"expected one of {CYCLE_GAS_PROPERTY_BASES!r}"
        )
    return value


def _mark_gas_property_basis(function, value: str):
    """Attach the construction-time basis to an adapter closure.

    The attribute is a plain immutable string on the function object, so
    callers can recover the wiring actually installed in a namespace without
    trusting the mutable namespace marker.
    """

    setattr(function, GAS_PROPERTY_BASIS_ATTR, _validate_gas_property_basis(value))
    return function


def _mark_cycle_metadata(function, result: dict[str, Any]):
    """Attach the construction-time cycle method metadata to a closure.

    ``result`` is the dictionary returned by :func:`cycle_point` for the basis
    actually used to build the surrounding namespace.  The legacy path carries
    no method dictionary, which is recorded explicitly as ``None``.  These
    attributes are construction-bound and independent of the stage-property
    attribute so the two selectors cannot silently couple.
    """

    basis = _validate_cycle_gas_property_basis(
        result.get("cycle_gas_property_basis", CYCLE_GAS_PROPERTY_BASIS_LEGACY)
    )
    setattr(function, CYCLE_GAS_PROPERTY_BASIS_ATTR, basis)
    setattr(
        function,
        CYCLE_ALPHA_BASIS_ATTR,
        result.get("cycle_alpha_basis", ALPHA_BASIS_LEGACY),
    )
    setattr(
        function,
        CYCLE_METHOD_DIAGNOSTICS_ATTR,
        copy.deepcopy(result.get("cycle_method_diagnostics")),
    )
    return function


def _ступень_ядро(ns: dict[str, Any], *,
                  gas_property_basis: str = GAS_PROPERTY_BASIS_LEGACY):
    """Адаптер ``turbine_engine.stage`` под сигнатуру и результат ячейки блокнота.

    Сигнатура, ключи результата и источники ``cp``/``k`` сохранены: значения
    берутся из чистого ``turbine_engine.gas.calc_stage_cp_k`` с фактическими
    ``α_г``, ``R`` и ``φ`` текущего пространства имён, а константы
    (``R``, ``φ``, ``ψ``, ``δ_отн``, ``δ_мин``, ``ε_парц``, ``k_n``) — из него же
    через :meth:`StageConfig.from_namespace`. Поэтому проектные допущения
    (φ/ψ, зазоры, парциальность) читаются после настройки параметров, а не из
    значений по умолчанию ядра.

    ``gas_property_basis`` фиксируется при создании адаптера. По умолчанию
    используется ``legacy_derivative`` — исходная notebook-аппроксимация
    производной ``calc_cp`` (средняя/конечно-разностная), а не опубликованный
    прямой fit. ``textbook_true_cp`` — только явный диагностический вариант
    с опубликованным прямым fit; он не меняет формулу ступени.
    """
    _validate_gas_property_basis(gas_property_basis)
    нет_решения = ns["НетРешения"]

    def ступень(T0, p0, H0, ρ, D_c, D_л, n, G_c, G_л, α1_град, κ):
        cp, k = calc_stage_cp_k(
            T0, H0, alpha=ns["α_г"], R=ns["R"], phi=ns["φ"],
            gas_property_basis=gas_property_basis,
        )
        state = StageState(
            T0=T0, p0=p0, H0=H0, rho=ρ, D_c=D_c, D_l=D_л, n=n,
            G_c=G_c, G_l=G_л, alpha1_deg=α1_град, kappa=κ, cp=cp, k=k,
        )
        try:
            return calculate_stage(
                state, StageConfig.from_namespace(ns)
            ).as_legacy_dict()
        except StageCalculationError as exc:
            raise нет_решения(str(exc)) from exc

    return _mark_gas_property_basis(ступень, gas_property_basis)


def _газ_ядро(ns: dict[str, Any], *,
              gas_property_basis: str = GAS_PROPERTY_BASIS_LEGACY):
    """Adapter ``газ(T)`` backed by the selected pure gas core.

    Like the notebook function, the closure resolves ``α_г`` and ``R`` from the
    active namespace at call time, so later helpers that look up ``газ`` in
    their globals keep the same composition semantics as the source.

    ``legacy_derivative`` (default) keeps :func:`calc_gas_cp_k` and therefore the
    original notebook derivative/mean-``cp`` approximation.  It is not the
    published direct ``cp(T)`` fit.  ``textbook_true_cp`` evaluates
    :func:`calc_textbook_gas_cp_k` directly at the requested temperature; no
    mean-``cp`` or cycle substitution is introduced here.
    """

    _validate_gas_property_basis(gas_property_basis)

    def газ(T):
        if gas_property_basis == GAS_PROPERTY_BASIS_LEGACY:
            return calc_gas_cp_k(T, ns["α_г"], ns["R"])
        return calc_textbook_gas_cp_k(T, ns["α_г"], ns["R"])

    return _mark_gas_property_basis(газ, gas_property_basis)


def _h0_по_p_ядро(ns: dict[str, Any], *,
                  gas_property_basis: str = GAS_PROPERTY_BASIS_LEGACY):
    """Adapter ``H0_по_p(T0, p0, p2)`` backed by the pure gas core.

    Resolves ``α_г``, ``R`` and ``φ`` from the active namespace at call time,
    so cell 16 (and any later cell that looks up ``H0_по_p``) sees the same
    design values as the notebook reference path.  ``gas_property_basis`` is
    fixed when the adapter is built and is passed to the pressure closure.
    """
    _validate_gas_property_basis(gas_property_basis)

    def H0_по_p(T0, p0, p2):
        return calc_h0_for_pressure(
            T0, p0, p2, ns["α_г"], ns["R"], ns["φ"],
            gas_property_basis=gas_property_basis,
        )

    return _mark_gas_property_basis(H0_по_p, gas_property_basis)


def _ρ_корня_ядро(ns: dict[str, Any]):
    """Adapter ``ρ_корня(r, закон)`` backed by :mod:`turbine_engine.reaction`.

    The notebook function receives the legacy stage dictionary.  It is mapped
    back to the typed :class:`turbine_engine.stage.StageResult` without changing
    any field, and the live namespace values ``R`` and ``φ`` are resolved at
    call time, exactly as the source function does.
    """

    def ρ_корня(r, закон):
        return calculate_root_reaction(
            StageResult.from_legacy_dict(r), закон, ns["R"], ns["φ"]
        )

    return ρ_корня


def _турбина_показатели_ядро(ns: dict[str, Any]):
    """Adapter for the numerical aggregation at the start of notebook cell 20.

    The pure function :func:`turbine_engine.performance.calculate_turbine_performance`
    receives the live namespace values as explicit arguments, so no hidden
    namespace lookup or notebook execution remains on the wired path.  The
    legacy identifier keys are assigned by the caller from ``legacy_values()``.
    """

    return calculate_turbine_performance(
        ns["ступени"],
        G_c1=ns["G_c1"],
        H_t=ns["H_т"],
        T_g_total=ns["T_г_полн"],
        p_g_total=ns["p_г_полн"],
        eta_m=ns["η_м"],
        G_v=ns["G_в"],
        L_k=ns["L_к"],
        eta_r=ns["η_р"],
        gas=ns["газ"],
    )


def _расширение_ядро(ns: dict[str, Any]):
    """Adapter for the numerical block at the start of notebook cell 9.

    The pure function :func:`turbine_engine.expansion.calculate_turbine_expansion`
    receives the live namespace values as explicit arguments.  The legacy
    identifier keys are assigned by the caller from ``legacy_values()``; the
    pandas presentation table from the source cell is not built on this path.
    """

    return calculate_turbine_expansion(
        ns["p_г_полн"],
        ns["p_т"],
        ns["T_г_полн"],
        ns["η_л_ожид"],
        gas=ns["газ"],
        R=ns["R"],
        G_l_ost=ns["G_л_ост"],
        c_ta=ns["c_та"],
        sigma_l=ns["σ_л"],
        k_n=ns["k_n"],
        q_n=ns["q_n"],
        y0=ns["y0"],
        l_t_D_t=ns["l_т_D_т"],
    )


def _геометрия_ядро(ns: dict[str, Any]):
    """Adapter for the numerical meridional geometry block of notebook cell 22.

    :func:`turbine_engine.geometry.calculate_meridional_geometry` receives the
    live stage mappings and every cell constant explicitly.  The legacy
    identifier keys (``контур``, ``геом``, ``γ_max``, ``dx_т``, ``γ_турб``) are
    assigned by the caller from ``legacy_values()``; the pandas display objects
    from the source cell are not built on the wired path.
    """

    config = MeridionalGeometryConfig(
        b_a_min=ns["b_a_мин"],
        lb_c=ns["lb_c"],
        lb_l=ns["lb_л"],
        lb_min_c=ns["lb_мин_c"],
        lb_min_l=ns["lb_мин_л"],
        delta_a_ratio=ns["δа_к_ba"],
        gamma_target=ns["γ_цель"],
        gamma_limit=ns["γ_макс"],
    )
    return calculate_meridional_geometry(ns["ступени"], config)


def _baseline_ядро(ns: dict[str, Any]):
    """Adapter for the numerical part of notebook cell 16.

    :func:`turbine_engine.baseline.calculate_baseline_turbine` receives the live
    namespace as its dependency mapping and resolves ``ступень_авто``,
    ``H0_по_p`` and every scalar/array from it at call time.  ``numpy.clip`` and
    ``numpy.mean`` are injected so the arithmetic is source-identical; the
    module returns the final legacy identifiers via ``legacy_values()`` and does
    not mutate ``ns``.  The pandas presentation tables and ``display`` calls
    from the source cell are not part of the wired path.
    """

    return calculate_baseline_turbine(ns, clip=np.clip, mean=np.mean)


def _optional_graph_diagnostic(rho: float) -> tuple[float | None, str, str]:
    """Необязательная диагностика рис. 18 без экстраполяции.

    ``ValueError`` штатно означает, что ρ вне оцифрованного диапазона. Это
    отсутствие диагностики, а не физический отказ кандидата: возвращается
    ``(None, "NOT_CHECKED", reason)``. Прочие исключения не перехватываются
    здесь — их должен показать вызывающий код, сохранив уже посчитанную физику.
    """

    try:
        return float(cs.uc0_opt_from_rho(rho)), "ok", ""
    except ValueError as exc:
        return None, "NOT_CHECKED", str(exc)


def _engineering_provenance(ns: dict[str, Any], res: dict[str, Any],
                            p: dict[str, Any]) -> dict[str, Any]:
    """Явная provenance инженерных подборов ячейки 14.

    ``ступень_авто``/``ступень_с_подбором_ρ`` — часть исходной модели, а не
    математические guard'ы: они поднимают ρ до ``ρ_корня_мин`` (с ограничением
    ``ρ_макс``) и для закона ``Γ`` повышают α1 до ``α1_мин_корня``. Эти
    действия сохраняются; в результат добавляются запрошенные и фактические
    параметры, чтобы ``res["p"]["a1"]`` не принимали за фактический α1.
    """

    stages = res["stages"]
    requested_a1 = [float(v) for v in p["a1"]]
    actual_a1 = [float(s["α1"]) for s in stages]
    actual_rho = [float(s["ρ"]) for s in stages]
    laws = [str(v) for v in ns.get("закон_ст", [])]
    repair_available = (
        "ступень_авто" in ns and "ступень_с_подбором_ρ" in ns
    )
    gamma_repair = repair_available and "Γ" in laws
    return {
        "model_variant": "legacy_cell16+cell14_engineering_repair",
        "a1_requested": requested_a1,
        "a1_actual": actual_a1,
        "rho_requested": None,
        "rho_actual": actual_rho,
        "engineering_adjustment": {
            "active": repair_available,
            "legacy_repair": repair_available,
            "mechanisms": [
                {
                    "name": "rho_root_floor",
                    "function": "ступень_с_подбором_ρ",
                    "status": "enabled" if repair_available else "unavailable",
                    "rho_root_min": ns.get("ρ_корня_мин"),
                    "rho_max": ns.get("ρ_макс"),
                },
                {
                    "name": "gamma_inlet_angle_floor",
                    "function": "ступень_авто",
                    "status": (
                        "enabled" if gamma_repair
                        else "not_applicable" if repair_available
                        else "unavailable"
                    ),
                    "alpha1_root_min": ns.get("α1_мин_корня"),
                },
            ],
            "requested": {"a1": requested_a1, "rho": None},
            "actual": {"a1": actual_a1, "rho": actual_rho},
            "changed": {
                "a1": [
                    not math.isclose(a, r, rel_tol=0.0, abs_tol=1e-9)
                    for r, a in zip(requested_a1, actual_a1)
                ],
            },
            "note": (
                "ρ не является входом: он подбирается внутри ступени; "
                "ступень_авто может изменить α1 для закона Γ. "
                "Это инженерная доводка модели, не подтверждение RAW."
            ),
        },
    }


def evaluate(x, m: Model, cfg, full: bool = False) -> dict[str, Any]:
    """Один расчёт кандидата.

    Ступень считается чистым ядром ``turbine_engine.stage``; ячейка блокнота
    по-прежнему исполняется, а её ``ступень`` нужна только как независимый
    эталон (см. ``_evaluate(..., stage_core=False)``).
    """
    return _evaluate(x, m, cfg, full, stage_core=True)


def _evaluate(x, m: Model, cfg, full: bool = False, *,
              stage_core: bool = True) -> dict[str, Any]:
    p = декодировать(x, cfg)
    if cfg.get("D_override"):
        p["D"] = [float(v) for v in cfg["D_override"]]
    if cfg.get("lD_override") is not None:
        p["lD"] = float(cfg["lD_override"])
    c = cycle_point(m, p["pi"], p["Tg"], p["N"])
    ns = dict(m.base)
    ns["T_г_полн"] = p["Tg"]
    ns["охлаждение_лопаток"] = c["охлаждение"]
    ns["g_охл"] = c["g_ohl"]
    ns["g_воз"] = c["g_voz"]
    exec(compile(m.code[m.param_idx], "параметры", "exec"), ns)
    # проектные допущения, которые не должны быть «книжными» константами
    ns["q_n"] = float(cfg.get("q_n", 0.995))
    seal = str(cfg.get("seal_type", "honeycomb"))
    if seal == "honeycomb":
        ns["δ_отн"], ns["δ_мин"] = 0.005, 0.2e-3
    else:
        ns["δ_отн"], ns["δ_мин"] = 0.0085, 0.5e-3
    ns.update({
        "π_к": c["pi"], "p_г_полн": c["p_g"], "p_т": c["p_t"],
        "G_в": c["G"], "G_c1": c["G"] * c["g_t"],
        "G_л_ост": c["G"] * (c["g_t"] + c["g_voz"]),
        "L_к": c["L_k"], "c_та": c["c_t"], "N_зад": c["N"],
        "g_т": c["g_t"], "α_г": c["alpha_g"],
        "η_т_полн_цикл": c["eta_t_cycle"],
        "N_т_цикл": c["G"] * c["L_t"] * (c["g_t"] + c["g_voz"]) * float(m.cycle["η_м_тур"]),
        "р": c["d"],
    })
    Ц = dict(m.cycle)
    Ц["c_т"] = c["c_t"]
    Ц["T_г_полн"] = p["Tg"]
    ns["Ц"] = Ц
    ns.update({
        "D_ст": p["D"], "l_т_D_т": p["lD"], "α1_ст": p["a1"],
        "π_веса": p["w"], "κ_ст": p["k"], "σ_л": p["sigma"],
        "φ": p["phi"], "ψ": p["psi"],
    })
    gas_cell = next(
        (
            i
            for i in m.core_idx
            if "def газ(" in m.code[i] and "def H0_по_p(" in m.code[i]
        ),
        None,
    )
    if stage_core and gas_cell is None:
        raise RuntimeError("notebook gas cell (cell 7) not found in core_idx")
    stage_cell = next((i for i in m.core_idx if "def ступень(" in m.code[i]), None)
    if stage_core and stage_cell is None:
        raise RuntimeError("notebook stage cell not found in core_idx")
    reaction_cell = next(
        (i for i in m.core_idx if "def ρ_корня(" in m.code[i]), None
    )
    if stage_core and reaction_cell is None:
        raise RuntimeError("notebook root-reaction cell (cell 14) not found in core_idx")
    performance_cell = next(
        (
            i
            for i in m.core_idx
            if 'g_i = [s["G_c"] / G_c1' in m.code[i] and "N_вал = (" in m.code[i]
        ),
        None,
    )
    if stage_core and performance_cell is None:
        raise RuntimeError(
            "notebook turbine performance cell (cell 20) not found in core_idx"
        )
    expansion_cell = next(
        (
            i
            for i in m.core_idx
            if "π_т = p_г_полн / p_т" in m.code[i] and "табл_п1_31" in m.code[i]
        ),
        None,
    )
    if stage_core and expansion_cell is None:
        raise RuntimeError(
            "notebook turbine expansion cell (cell 9) not found in core_idx"
        )
    if stage_core and expansion_cell < gas_cell:
        raise RuntimeError(
            "notebook turbine expansion cell (cell 9) precedes the gas cell (cell 7)"
        )
    baseline_cell = next(
        (
            i
            for i in m.core_idx
            if "def турбина(" in m.code[i]
            and "def распределить_π(" in m.code[i]
            and "def решить(" in m.code[i]
            and "def N_на_валу(" in m.code[i]
            and "def _N_of(G)" in m.code[i]
        ),
        None,
    )
    if stage_core and baseline_cell is None:
        raise RuntimeError(
            "notebook baseline turbine cell (cell 16) not found in core_idx"
        )
    geometry_cell = next(
        (
            i
            for i in m.full_idx
            if "def γ_участка(" in m.code[i]
            and "def ширина(" in m.code[i]
            and "def r_нв(" in m.code[i]
            and "контур, геом, x = [], [], 0.0" in m.code[i]
        ),
        None,
    )
    if full and stage_core and geometry_cell is None:
        raise RuntimeError(
            "notebook meridional geometry cell (cell 22) not found in full_idx"
        )
    for i in m.core_idx:
        # Числовая часть ячейки 16 (многоступенчатая турбина: подбор A_т, n_т,
        # D_т, π ступеней и расхода G) заменена чистым модулем. Презентационные
        # таблицы pandas/display не строятся; при stage_core=False ячейка
        # исполняется целиком как независимый эталон. Ячейки 7/9/12/14 к этому
        # моменту уже исполнены, поэтому baseline читает из ns подменённые
        # газовые/ступенчатые/реактивные адаптеры.
        if stage_core and i == baseline_cell:
            ns.update(_baseline_ядро(ns).legacy_values())
            continue
        # Числовая агрегация ячейки 20 заменена чистым модулем. Таблицы
        # DataFrame/display из этой ячейки оптимизатору не нужны; при
        # stage_core=False ячейка исполняется целиком как независимый эталон.
        if stage_core and i == performance_cell:
            ns.update(_турбина_показатели_ядро(ns).legacy_values())
            continue
        # Числовая часть ячейки 9 (расширение и предварительное профилирование)
        # заменена чистым модулем. Презентационная таблица pandas не строится;
        # при stage_core=False ячейка исполняется целиком как независимый
        # эталон. Газовая ячейка 7 уже исполнена выше и её ``газ`` заменён
        # чистым адаптером.
        if stage_core and i == expansion_cell:
            ns.update(_расширение_ядро(ns).legacy_values())
            continue
        exec(compile(m.code[i], f"ядро {i}", "exec"), ns)
        # после газовой ячейки подменяем газовые функции чистыми ядрами; ячейки
        # 9/16/20 уже будут читать их из пространства имён. При
        # stage_core=False замены нет, поэтому эталонное поведение блокнота
        # сохраняется полностью.
        if stage_core and i == gas_cell:
            ns["газ"] = _газ_ядро(ns)
            ns["H0_по_p"] = _h0_по_p_ядро(ns)
        # после определения ступени заменяем её на чистое ядро; ячейка 12 при
        # этом уже исполнена, поэтому её определение остаётся доступным как
        # независимый эталон при stage_core=False.
        if stage_core and i == stage_cell:
            ns["ступень"] = _ступень_ядро(ns)
        # после ячейки 14 заменяем корневую реактивность чистым ядром; сама
        # ячейка уже исполнена и остаётся эталоном при stage_core=False.
        if stage_core and i == reaction_cell:
            ns["ρ_корня"] = _ρ_корня_ядро(ns)
    if full:
        for i in m.full_idx:
            # Числовая часть ячейки 22 (меридиональный контур и ширины венцов)
            # заменена чистым модулем. Презентационные DataFrame/display не
            # строятся; при stage_core=False ячейка исполняется целиком как
            # независимый эталон, а при full=False не исполняется вовсе.
            if stage_core and i == geometry_cell:
                ns.update(_геометрия_ядро(ns).legacy_values())
                continue
            exec(compile(m.code[i], f"полный {i}", "exec"), ns)

    st = ns["ступени"]
    res = {
        "x": np.asarray(x, dtype=float), "p": p, "cycle": c, "stages": st,
        "lc1": float(st[0]["l_c"]), "lc": [float(s["l_c"]) for s in st],
        "ll": [float(s["l_л"]) for s in st],
        "a2": [float(s["α2"]) for s in st],
        "pi": [float(s["π_ст"]) for s in st],
        "sigma": [float(s["σ_р"]) for s in st],
        "D_t": float(ns["D_т"]), "u_t": float(ns["u_т"]), "n_t": float(ns["n_т"]),
        "N_val": float(ns["N_вал"]), "eta": float(ns["η_т_полн"]),
        "loss": 1.0 - float(ns["η_т_полн"]),
    }
    res["sigma_max"] = max(res["sigma"])
    if full:
        res["gamma_max"] = float(ns["γ_max"])
        res["rho_root"] = [float(df.loc["ρ", "корень"]) for df in _профили(ns)]
        res["lambda_c1"] = [float(df.loc["λ_c1", "средний"]) for df in _профили(ns)]
        res["alpha1_root"] = [float(df.loc["α₁, град", "корень"]) for df in _профили(ns)]
        res["phi_ref"] = [float(v) for v in ns["решётки_df"].loc["φ уточнённый"]]
        res["psi_ref"] = [float(v) for v in ns["решётки_df"].loc["ψ уточнённый"]]
        res["geom"] = ns.get("геом", [])
    res.update(_engineering_provenance(ns, res, p))
    return res


def evaluate_sizing(m: Model, cfg: dict, full: bool = True, max_iter: int = 6, tol: float = 1e-3) -> dict[str, Any]:
    """Новый путь: D_ср, l, D_вт — из A и l/D; D_вт = const подбирается λ_i.

    cfg["lam"] — начальные l/D по ступеням; cfg["D_hub"] — цель (если None,
    берётся среднее D_вт первой итерации). Возвращает res с полем "sizing".
    """
    import sizing as sz

    cfg = dict(cfg)
    cfg["geometry"] = "sizing"
    lam = [float(v) for v in cfg.get("lam", [0.25, 0.25, 0.25])]
    x = warm_start(cfg)
    r = evaluate(x, m, cfg, full=full)
    target = cfg.get("D_hub")
    for _ in range(max_iter):
        st = r["stages"]
        if target is None:
            target = sum(s["D_л"] - s["l_л"] for s in st) / len(st)
        new_lam, D = [], []
        for s in st:
            A2 = s["G_л"] * s["v2"] / s["c2a"]
            l = sz.solve_lam_for_hub(A2, target)
            new_lam.append(l)
            D.append(math.sqrt(A2 / (math.pi * l)))
        D_last = D[-1]
        cfg2 = dict(cfg)
        cfg2["D_override"] = [d / D_last for d in D]
        cfg2["lD_override"] = new_lam[-1]
        r2 = evaluate(x, m, cfg2, full=full)
        hub = [s["D_л"] - s["l_л"] for s in r2["stages"]]
        spread = max(hub) - min(hub)
        r, lam = r2, new_lam
        if spread <= tol * max(hub):
            break
    u_c0_opt: list[float | None] = []
    u_c0_opt_status: list[str] = []
    u_c0_opt_reason: list[str] = []
    for s in r["stages"]:
        value, status, reason = _optional_graph_diagnostic(s["ρ"])
        u_c0_opt.append(value)
        u_c0_opt_status.append(status)
        u_c0_opt_reason.append(reason)
    r["sizing"] = dict(
        lam=lam, target=target,
        D_hub=[s["D_л"] - s["l_л"] for s in r["stages"]],
        D_mean=[s["D_л"] for s in r["stages"]],
        l_l=[s["l_л"] for s in r["stages"]],
        u_c0=[0.5 * (s["u1"] + s["u2"]) / math.sqrt(2.0 * s["H0"]) for s in r["stages"]],
        u_c0_opt=u_c0_opt,
        u_c0_opt_status=u_c0_opt_status,
        u_c0_opt_reason=u_c0_opt_reason,
    )
    return r


def _профили(ns):
    законы = ns["закон_ст"]
    return [ns["профили"][з][i] for i, з in enumerate(законы)]


# ---------------------------------------------------------------------------
# Валидация
# ---------------------------------------------------------------------------
def _interval_amount(v, lo, hi):
    if v < lo:
        return (lo - v) / max(abs(lo), 1e-9)
    if v > hi:
        return (v - hi) / max(abs(hi), 1e-9)
    return 0.0


def validate(res: dict, cfg, full: bool = False) -> list[dict[str, Any]]:
    """Возвращает список проверок. kind: hard / project / target / info."""
    c = []
    st = res["stages"]
    L = cfg.get("limits", {})

    def add(name, value, limit, ok, kind, amount=0.0, note=""):
        c.append(dict(name=name, value=value, limit=limit, ok=bool(ok), kind=kind, amount=float(amount), note=note))

    # геометрия и компоновка
    add("z = 3", len(st), "3", len(st) == 3, "hard")
    D = res["p"]["D"]
    if cfg.get("geometry") == "sizing":
        add("D_ср (sizing)", "/".join(f"{v:.3f}" for v in D), "из A и l/D", True, "info", 0.0)
    else:
        add("D1 <= D2 <= D3=1", "/".join(f"{v:.3f}" for v in D), "не убывают, D3=1",
            D[0] <= D[1] + 1e-9 and abs(D[2] - 1.0) < 1e-9, "project",
            max(0.0, D[0] - D[1]))
    add("l_т/D_т <= 1/3", f"{res['p']['lD']:.4f}", "<= 0.3333", res["p"]["lD"] <= 1 / 3 + 1e-9, "project",
        max(0.0, res["p"]["lD"] - 1 / 3))

    # углы
    a1 = res["p"]["a1"]
    amin = L.get("a1_min", 11.0)
    for i, v in enumerate(a1, 1):
        add(f"α1 ст.{i} >= {amin:.0f}°", f"{v:.2f}°", f">= {amin:.0f}°", v >= amin - 1e-9, "hard",
            max(0.0, (amin - v) / amin))
    # по умолчанию книга, стр. 38: 70…90 / 80…90; нижние границы можно ослабить только явно через cfg["limits"]
    lo_first = L.get("a2_lo_first", 70.0)
    lo_mid = L.get("a2_lo_mid", 70.0)
    lo_last = L.get("a2_lo_last", 80.0)
    for i, v in enumerate(res["a2"], 1):
        if i == len(res["a2"]):
            lo, hi = lo_last, 90.0
        elif i == 1:
            lo, hi = lo_first, 90.0
        else:
            lo, hi = lo_mid, 90.0
        ok = lo - 1e-9 <= v <= hi + 1e-9
        add(f"α2 ст.{i} ∈ [{lo:.0f}; {hi:.0f}]°", f"{v:.2f}°", f"[{lo:.0f}; {hi:.0f}]°", ok, "hard",
            _interval_amount(v, lo, hi))
    add("α2 не более 90° (все ступени)", "/".join(f"{v:.1f}" for v in res["a2"]), "<= 90°",
        all(v <= 90.0 + 1e-9 for v in res["a2"]), "hard",
        max([max(0.0, (v - 90.0) / 90.0) for v in res["a2"]] or [0.0]))

    # π ступеней
    for i, v in enumerate(res["pi"], 1):
        lo, hi = L.get("pi_min", 1.8), L.get("pi_max", 2.4)
        ok = lo - 1e-9 <= v <= hi + 1e-9
        add(f"π ст.{i} ∈ [{lo}; {hi}]", f"{v:.3f}", f"[{lo}; {hi}]", ok, "hard", _interval_amount(v, lo, hi))

    # прочность и скорость
    sig_dop = res["p"]["sigma"]
    # 1e-4 - допуск сходимости итерации A_т (тот же, что в модели), не физическое послабление
    add("σ_р max <= σ_л", f"{res['sigma_max']:.1f} МПа", f"<= {sig_dop:.0f} МПа",
        res["sigma_max"] <= sig_dop * (1.0 + 1e-4), "hard",
        max(0.0, (res["sigma_max"] - sig_dop) / sig_dop))
    lD = res["p"]["lD"]
    # с. 59: 300…400 обычные; 400…450 при коротких лопатках. Порог l/D<1/6 — проектное допущение.
    u_max = float(L.get("u_max_short", 450.0)) if lD < 1 / 6 else float(L.get("u_max", 400.0))
    add("u_т <= u_max (с. 59)", f"{res['u_t']:.1f} м/с", f"<= {u_max:.0f}",
        res["u_t"] <= u_max + 1e-9, "hard", max(0.0, (res["u_t"] - u_max) / u_max))

    # нижняя граница l_т/D_т: с. 58 даёт 1/4…1/3,5; с. 87 п. 45 — 1/3,5…1/3.
    add("l_т/D_т >= 0.25 (с. 58)", f"{lD:.4f}", ">= 0.25", lD >= 0.25 - 1e-9,
        "recommendation", max(0.0, (0.25 - lD) / 0.25))
    # σ_л: с. 57 — 120…200 обычные, ≤300 особо напряжённые; 400 МПа книгой не подтверждено
    add("σ_л <= 300 МПа (с. 57)", f"{sig_dop:.0f} МПа", "<= 300", sig_dop <= 300.0 + 1e-9,
        "project", max(0.0, (sig_dop - 300.0) / 300.0))
    # n < n_max: q_n = sqrt(σ_р_max/σ_л); при q_n=0.995 запас ≈1 % по напряжениям
    q_n_eff = math.sqrt(max(res["sigma_max"], 0.0) / sig_dop) if sig_dop > 0 else 1.0
    add("n < n_max (q_n)", f"q_n = {q_n_eff:.3f}", "< 1", q_n_eff < 1.0 - 1e-9, "hard",
        max(0.0, q_n_eff - 1.0))

    # длины лопаток
    lengths = [v for pair in zip(res["lc"], res["ll"]) for v in pair]
    ok_len = all(b >= a * 0.999 for a, b in zip(lengths, lengths[1:]))  # 0.1 % — сходимость G/A_т
    add("длины лопаток не убывают", " -> ".join(f"{v*1e3:.1f}" for v in lengths), "не убывают (0.1 % — сходимость)", ok_len,
        "project", 0.0 if ok_len else 0.1)
    add("l_с1 >= 14 мм (проект)", f"{res['lc1']*1e3:.2f} мм", ">= 14 мм", res["lc1"] >= 0.014 - 1e-12, "project",
        max(0.0, (0.014 - res["lc1"]) / 0.014))
    add("l_с1 >= 20 мм (цель)", f"{res['lc1']*1e3:.2f} мм", ">= 20 мм",
        res["lc1"] >= cfg.get("lc1_target", 0.020), "target",
        max(0.0, (cfg.get("lc1_target", 0.020) - res["lc1"]) / cfg.get("lc1_target", 0.020)))

    # мощность, давление, невязки
    Nz = res["p"]["N"]
    add("N на входе редуктора = требуемая", f"{res['N_val']/1e3:.2f} кВт", f"{Nz/1e3:.1f} кВт",
        abs(res["N_val"] / Nz - 1.0) < 1e-3, "hard", abs(res["N_val"] / Nz - 1.0))
    p_t = res["cycle"]["p_t"]
    add("p2z = p_т", f"{st[-1]['p2']/1e3:.2f} кПа", f"{p_t/1e3:.2f} кПа",
        abs(st[-1]["p2"] / p_t - 1.0) < 1e-6, "hard", abs(st[-1]["p2"] / p_t - 1.0))
    add("η*т >= 0.90 (цель)", f"{res['eta']:.4f}", ">= 0.90", res["eta"] >= cfg.get("eta_target", 0.90),
        "target", max(0.0, (cfg.get("eta_target", 0.90) - res["eta"]) / 0.90))

    # невязки уравнений
    cont = []
    for i, s in enumerate(st):
        A = math.pi * s["D_c"] * s["l_c"]
        cont.append(abs(A - s["G_c"] * s["v1"] / s["c1a"]) / max(A, 1e-12))
    add("невязка неразрывности (max)", f"{max(cont):.2e}", "< 1e-6", max(cont) < 1e-6, "hard", max(cont))
    en = []
    for s in st:
        T2p = s["T2"] + s["c2"] ** 2 / (2 * s["cp"])
        en.append(abs(s["L_u"] - s["cp"] * (s["T0"] - T2p)) / max(abs(s["L_u"]), 1e-9))
    add("невязка энергии (max)", f"{max(en):.2e}", "< 1e-6", max(en) < 1e-6, "hard", max(en))

    if full:
        add("γ <= 25°", f"{res.get('gamma_max', 0):.1f}°", "<= 25°", res.get("gamma_max", 0) <= 25.0,
            "hard", max(0.0, (res.get("gamma_max", 0) - 25.0) / 25.0))
        rmin = float(L.get("rho_root_min", 0.05))
        for i, v in enumerate(res.get("rho_root", []), 1):
            add(f"ρ корня ст.{i} >= {rmin:g}", f"{v:.3f}", f">= {rmin:g}", v >= rmin,
                "hard", max(0.0, (rmin - v) / rmin))
        for i, v in enumerate(res.get("lambda_c1", []), 1):
            add(f"λ_c1 ст.{i} <= 1.05", f"{v:.3f}", "<= 1.05", v <= 1.05, "hard",
                max(0.0, (v - 1.05) / 1.05))
        # α1 у корня: книга стр. 73 (13…14°, малые турбины 11…12°)
        for i, v in enumerate(res.get("alpha1_root", []), 1):
            add(f"α1 у корня ст.{i} >= 11°", f"{v:.2f}°", ">= 11°", v >= 11.0 - 1e-9, "hard",
                max(0.0, (11.0 - v) / 11.0))

    # --- единый модуль constraints: геометрия, рис. 18/19, φ/ψ, δ ---
    rho_root_all = res.get("rho_root") or []
    phi_ref_all = res.get("phi_ref") or []
    psi_ref_all = res.get("psi_ref") or []
    for i, s in enumerate(st):
        u_sr = 0.5 * (s["u1"] + s["u2"])
        norm = dict(
            D_c=s["D_c"], D_l=s["D_л"], l_c=s["l_c"], l_l=s["l_л"],
            alpha1=s["α1"], alpha2=s["α2"], pi=s["π_ст"], sigma=s["σ_р"],
            sigma_allow=sig_dop, u=u_sr, rho=s["ρ"],
            rho_root=(rho_root_all[i] if i < len(rho_root_all) else None),
            x_uc0=u_sr / math.sqrt(2.0 * s["H0"]) if s["H0"] > 0 else None,
            x_uc1=u_sr / s["c1"] if s["c1"] > 0 else None,
            eta_u=s["η_u"], delta=s.get("δ_л"),
            phi=res["p"]["phi"], psi=res["p"]["psi"],
            phi_ref=(phi_ref_all[i] if i < len(phi_ref_all) else None),
            psi_ref=(psi_ref_all[i] if i < len(psi_ref_all) else None),
            last=(i == len(st) - 1), lD=(s["l_л"] / s["D_л"] if s["D_л"] else None),
        )
        for chk in cs.stage_checks(norm, cfg):
            if chk["key"].startswith(("geom_", "uc0_", "uc1_", "eta_u_")) or \
               chk["key"] in ("phi_ref", "psi_ref", "delta_abs", "delta_rel"):
                c.append(chk)

    # l/b_a и b_a — из геометрии венцов (яч. 22); в DE (full=False) недоступны
    geom = res.get("geom") or []
    if geom:
        for i, (s, g) in enumerate(zip(st, geom), 1):
            lbc = s["l_c"] / g["b_c"]
            lbl = s["l_л"] / g["b_л"]
            add(f"l/b_a СА ст.{i}", f"{lbc:.2f}", "1.0…2.5",
                1.0 - 1e-9 <= lbc <= 2.5 + 1e-9, "recommendation",
                _interval_amount(lbc, 1.0, 2.5))
            add(f"l/b_a РК ст.{i}", f"{lbl:.2f}", "1.5…3.0",
                1.5 - 1e-9 <= lbl <= 3.0 + 1e-9, "recommendation",
                _interval_amount(lbl, 1.5, 3.0))
            add(f"b_a СА ст.{i} >= 5 мм", f"{g['b_c']*1e3:.1f} мм", ">= 5 мм",
                g["b_c"] >= 0.005 - 1e-12, "project",
                max(0.0, (0.005 - g["b_c"]) / 0.005))
            add(f"b_a РК ст.{i} >= 5 мм", f"{g['b_л']*1e3:.1f} мм", ">= 5 мм",
                g["b_л"] >= 0.005 - 1e-12, "project",
                max(0.0, (0.005 - g["b_л"]) / 0.005))
    else:
        c.append(dict(name="l/b_a, b_a_min", value="нет в полном расчёте (full=False)",
                      limit="NOT_CHECKED", ok=True, kind="info", amount=0.0,
                      status="not_checked", source="[1] с. 61", page=61,
                      note="проверяется при full=True; в DE не считается"))
    k = res["p"]["k"]
    c.append(dict(name="κ = c2a/c1a", value="/".join(f"{v:.2f}" for v in k),
                  limit=">1 — частный случай l_л1=l_л2", ok=all(v > 1.0 for v in k),
                  kind="info", amount=0.0, status="book_recommendation",
                  source="[1] с. 62", page=62,
                  note="в модели D_c=D_л; κ>1 не является универсальным требованием"))
    return c


def penalty(checks, res, cfg) -> float:
    w = {"hard": 1000.0, "project": 100.0, "recommendation": 1.0, "target": 1.0, "info": 0.0}
    p = sum(w[c["kind"]] * c["amount"] for c in checks if not c["ok"])
    p += 1.0 * max(0.0, 1.0 - res["lc1"] / cfg.get("lc1_target", 0.020))
    p += 0.2 * max(0.0, 1.0 - res["eta"] / cfg.get("eta_target", 0.90))
    return float(p)


def is_feasible(checks):
    applicable = [c for c in checks if c["kind"] in ("hard", "project")]
    return bool(applicable) and all(c["ok"] for c in applicable)


def is_strict(checks):
    applicable = [c for c in checks if c["kind"] in ("hard", "project", "target")]
    return bool(applicable) and all(c["ok"] for c in applicable)


def _attempt_error(exc: BaseException) -> dict[str, Any]:
    """Структурированная запись исключения попытки (не RAW-результат)."""

    out: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": str(exc),
    }
    for name in ("status", "stage", "reason", "last_valid_state"):
        value = getattr(exc, name, None)
        if value is not None:
            out[name] = value
    return out


# ---------------------------------------------------------------------------
# Поиск
# ---------------------------------------------------------------------------
@dataclass
class Archive:
    rows: list[dict] = field(default_factory=list)

    def add(self, res, checks, score, **extra):
        feasible = extra.pop("feasible", None)
        strict = extra.pop("strict", None)
        row = dict(
            res=res, checks=checks, score=score,
            feasible=is_feasible(checks) if feasible is None else bool(feasible),
            strict=is_strict(checks) if strict is None else bool(strict),
        )
        row.update(extra)
        self.rows.append(row)

    def best(self, strict=False):
        rows = [r for r in self.rows if (r["strict"] if strict else r["feasible"])]
        if not rows:
            rows = self.rows
        return min(rows, key=lambda r: r["score"])


def _score_x(x, m, cfg, full, archive):
    """Оценить кандидата и сохранить попытку даже при исключении.

    Возвращаемое значение остаётся совместимым (``1e9`` при отказе), но запись
    в архиве больше не теряет вход, конфигурацию и уже посчитанные ``res`` /
    ``checks``. ``phase`` показывает, где именно возник отказ. Это диагностика
    попытки, а не заявка на валидированный RAW-результат.
    """

    x_attempt = np.array(x, dtype=float, copy=True)
    cfg_attempt = copy.deepcopy(cfg)
    res = None
    checks: list[dict[str, Any]] = []
    phase = "calculate"
    try:
        res = evaluate(x, m, cfg, full=full)
        phase = "validate"
        checks = validate(res, cfg, full=full)
        phase = "penalty"
        sc = penalty(checks, res, cfg)
    except Exception as exc:  # noqa: BLE001
        # KeyboardInterrupt/SystemExit наследуют BaseException и сюда не
        # попадают: прерывание не превращается в запись архива.
        archive.add(
            res, checks, 1e9,
            x=x_attempt, cfg=cfg_attempt, full=bool(full), phase=phase,
            error=_attempt_error(exc), attempt=True,
            feasible=False, strict=False,
        )
        return 1e9
    archive.add(
        res, checks, sc,
        x=x_attempt, cfg=cfg_attempt, full=bool(full), phase="score",
        attempt=True,
    )
    return sc


def run_random(m, cfg, n=40, seed=1, full=False):
    rng = np.random.default_rng(seed)
    b = границы(cfg)
    arch = Archive()
    x0 = warm_start(cfg)
    _score_x(np.clip(x0, b[:, 0], b[:, 1]), m, cfg, full, arch)
    for _ in range(n):
        x = rng.uniform(b[:, 0], b[:, 1])
        _score_x(x, m, cfg, full, arch)
    return arch


def run_global(m, cfg, maxiter=3, popsize=3, seed=1, full=False):
    from scipy.optimize import differential_evolution

    b = границы(cfg)
    dim = len(b)
    rng = np.random.default_rng(seed)
    size = max(popsize * dim, 2)
    init = rng.uniform(b[:, 0], b[:, 1], size=(size, dim))
    init[0] = np.clip(warm_start(cfg), b[:, 0], b[:, 1])
    arch = Archive()

    def f(x):
        return _score_x(x, m, cfg, full, arch)

    differential_evolution(f, b, maxiter=maxiter, popsize=popsize, seed=seed,
                           init=init, polish=False, tol=1e-3, workers=1, updating="immediate")
    return arch


def run_local(m, cfg, x0=None, steps=2, seed=1, full=False):
    b = границы(cfg)
    arch = Archive()
    best = np.clip(warm_start(cfg) if x0 is None else np.asarray(x0, float), b[:, 0], b[:, 1])
    best_score = _score_x(best, m, cfg, full, arch)
    for _ in range(steps):
        улучшилось = False
        for j in range(len(best)):
            for s in (+0.04, -0.04):
                x = best.copy()
                x[j] = np.clip(x[j] + s * (b[j, 1] - b[j, 0]), b[j, 0], b[j, 1])
                sc = _score_x(x, m, cfg, full, arch)
                if sc < best_score - 1e-9:
                    best, best_score, улучшилось = x, sc, True
        if not улучшилось:
            break
    return best, best_score, arch


def pareto(rows):
    """Недоминируемые по (max lc1, max η, min σ, min loss)."""
    ok = [r for r in rows if r["res"] is not None and r["feasible"]]
    if len(ok) < 2:
        ok = [r for r in rows if r["res"] is not None]
    if not ok:
        return []
    vals = np.array([[r["res"]["lc1"], r["res"]["eta"], -r["res"]["sigma_max"], -r["res"]["loss"]] for r in ok])
    out = []
    for i, r in enumerate(ok):
        dominated = any(all(vals[j] >= vals[i]) and any(vals[j] > vals[i]) for j in range(len(ok)) if j != i)
        if not dominated:
            out.append(r)
    return out


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def checks_table(checks):
    import pandas as pd

    return pd.DataFrame([dict(проверка=c["name"], значение=c["value"], допуск=c["limit"],
                              тип=c["kind"], статус="OK" if c["ok"] else "НЕ выполнено") for c in checks])


def stages_table(res):
    import pandas as pd

    rows = []
    for i, s in enumerate(res["stages"], 1):
        rows.append({
            "Ступень": i, "D_ср, мм": s["D_c"] * 1e3, "u, м/с": s["u1"],
            "l_с, мм": s["l_c"] * 1e3, "l_л, мм": s["l_л"] * 1e3,
            "α1, °": s["α1"], "α2, °": s["α2"], "β1, °": s["β1"], "β2, °": s["β2"],
            "ρ": s["ρ"], "π": s["π_ст"], "T*0, К": s["T0"], "T*2, К": s["T_ст_п"],
            "p*0, кПа": s["p0"] / 1e3, "p2, кПа": s["p2"] / 1e3,
            "η*т": s["η_т_п"], "L_т, кДж/кг": s["L_т"] / 1e3, "σ_р, МПа": s["σ_р"],
        })
    return pd.DataFrame(rows).set_index("Ступень").T
