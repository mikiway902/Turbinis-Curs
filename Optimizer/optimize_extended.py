"""Расширенная оптимизация: n, D_вт, w1, w2, α1, κ при D_вт=const.

Сценарии:
  A87_strict / A87_relaxed  — α2 строгие; l_т/D_т ≤ 1/3.5 или ≤ 1/3 (без нижней границы)
  A58_strict / A58_relaxed  — α2 строгие; 0.25 ≤ l_т/D_т ≤ 1/3.5 или ≤ 1/3
  B87_* / B58_*             — α2_1 = 20…90, остальные книжные

Пилот:
    python optimize_extended.py --scenario A87_strict --seeds 1 --maxiter 2 --popsize 3
Полный прогон:
    python optimize_extended.py --scenario all --seeds 10 --maxiter 20 --popsize 6

Важно: D_вт — геометрическая неизвестная, не проектный вход. n — переменная;
q_n = n/n_max — отчётный запас, не множитель. Все результаты сохраняются.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import bisect
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Optimizer"))
sys.path.insert(0, str(ROOT / "Turbinis-Calc"))
import optimizer_core as oc  # noqa: E402
import constraints as cs  # noqa: E402
from turbine_engine.gas import (  # noqa: E402
    GAS_PROPERTY_BASES,
    GAS_PROPERTY_BASIS_LEGACY,
    GAS_PROPERTY_BASIS_TEXTBOOK,
)
from turbine_engine.cycle import (  # noqa: E402
    ALPHA_BASIS_LEGACY,
    CYCLE_GAS_PROPERTY_BASES,
    CYCLE_GAS_PROPERTY_BASIS_LEGACY,
    CYCLE_GAS_PROPERTY_BASIS_TEXTBOOK,
)
from turbine_engine.orchestration import build_const_hub_orchestration  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)
K_N = 6.8e-6
SIGMA_LIM = 300.0
LC1_MIN = 0.014
N_TARGET = 110e3
U_MAX = 400.0
# Явные метки метода. Базовый вариант сохраняет исходную notebook-аппроксимацию
# среднего cp; textbook_true_cp — только явно выбранный диагностический вариант.
CYCLE_GAS_PROPERTY_BASIS = CYCLE_GAS_PROPERTY_BASIS_LEGACY
TEXTBOOK_DIAGNOSTIC_MODEL_VARIANT = (
    "extended_const_hub+textbook_true_cp_diagnostic"
)
TEXTBOOK_CYCLE_DIAGNOSTIC_MODEL_VARIANT = (
    "extended_const_hub+textbook_true_cp_cycle_diagnostic"
)
TEXTBOOK_BOTH_DIAGNOSTIC_MODEL_VARIANT = (
    "extended_const_hub+textbook_true_cp_cycle_and_stage_diagnostic"
)
# Отсутствие необязательной диагностики рис. 18 не делает точку лучше:
# в soft-ранжировании такие точки штрафуются консервативно.
SOFT_MISSING_DIAGNOSTIC_PENALTY = 1e3


def _installed_gas_property_basis(function) -> str | None:
    """Basis recorded by the factory that created an installed helper.

    ``inspect.unwrap`` sees through the optional ``lru_cache`` wrappers.  A
    missing attribute means the entry is a notebook/legacy or synthetic
    function rather than one of the extracted adapters.
    """

    if function is None:
        return None
    try:
        function = inspect.unwrap(function)
    except (TypeError, ValueError):
        pass
    return getattr(function, oc.GAS_PROPERTY_BASIS_ATTR, None)


def _gas_property_basis_state(ns: dict) -> tuple[str, str | None]:
    """Resolve the actual basis from installed wiring and check the marker.

    The mutable ``ns["_gas_property_basis"]`` marker is not trusted on its own.
    The return value's first item is the construction-bound basis recovered from
    the installed ``газ`` adapter; a marker without verifiable wiring, or a
    marker that disagrees with it, is rejected.  The second item is a
    configuration error to reject before any physical evaluation, or ``None``.
    """

    marker = ns.get("_gas_property_basis")
    wiring = _installed_gas_property_basis(ns.get("газ"))

    if marker is not None and (
        not isinstance(marker, str) or marker not in GAS_PROPERTY_BASES
    ):
        basis = (
            wiring
            if wiring in GAS_PROPERTY_BASES
            else GAS_PROPERTY_BASIS_LEGACY
        )
        return basis, (
            f"unknown gas_property_basis marker {marker!r}; "
            f"expected one of {GAS_PROPERTY_BASES!r}"
        )
    if wiring is not None and wiring not in GAS_PROPERTY_BASES:
        return wiring, (
            f"unknown installed gas_property_basis {wiring!r}; "
            f"expected one of {GAS_PROPERTY_BASES!r}"
        )
    if wiring is None:
        if marker is not None:
            return marker, (
                "gas_property_basis marker is present but installed газ "
                "wiring has no construction metadata; rebuild the namespace"
            )
        return GAS_PROPERTY_BASIS_LEGACY, None

    for name in ("H0_по_p", "ступень"):
        other = _installed_gas_property_basis(ns.get(name))
        if other is not None and other != wiring:
            return wiring, (
                f"installed {name} basis {other!r} does not match "
                f"installed газ basis {wiring!r}; rebuild the namespace"
            )
    if marker is not None and marker != wiring:
        return wiring, (
            f"gas_property_basis marker {marker!r} does not match installed "
            f"wiring {wiring!r}; rebuild the namespace instead of mutating "
            "the marker"
        )
    return wiring, None


def _installed_cycle_metadata(function) -> dict | None:
    """Cycle method metadata recorded by the factory that built ``газ``.

    It is deliberately a separate set of attributes from the stage gas basis:
    the namespace may select the stage and cycle bases independently.
    """

    if function is None:
        return None
    try:
        function = inspect.unwrap(function)
    except (TypeError, ValueError):
        pass
    basis = getattr(function, oc.CYCLE_GAS_PROPERTY_BASIS_ATTR, None)
    if basis is None:
        return None
    return {
        "gas_property_basis": basis,
        "alpha_basis": getattr(
            function, oc.CYCLE_ALPHA_BASIS_ATTR, ALPHA_BASIS_LEGACY
        ),
        "method_diagnostics": getattr(
            function, oc.CYCLE_METHOD_DIAGNOSTICS_ATTR, None
        ),
    }


def _cycle_gas_property_basis_state(
    ns: dict,
) -> tuple[str, str, object | None, str | None]:
    """Resolve the actual cycle basis from installed construction metadata.

    The mutable ``ns["_cycle_gas_property_basis"]`` marker is only a coherence
    check, exactly as for the stage basis.  If it disagrees with the metadata
    fixed when the namespace was built, the namespace is rejected before any
    physical evaluation.  The default legacy path has no method dictionary, so
    ``method_diagnostics`` is ``None``.
    """

    marker = ns.get("_cycle_gas_property_basis")
    metadata = _installed_cycle_metadata(ns.get("газ"))
    installed = metadata["gas_property_basis"] if metadata is not None else None
    alpha_basis = (
        metadata["alpha_basis"] if metadata is not None else ALPHA_BASIS_LEGACY
    )
    diagnostics = (
        metadata["method_diagnostics"] if metadata is not None else None
    )

    if installed is not None and installed not in CYCLE_GAS_PROPERTY_BASES:
        return installed, alpha_basis, diagnostics, (
            f"unknown installed cycle gas_property_basis {installed!r}; "
            f"expected one of {CYCLE_GAS_PROPERTY_BASES!r}"
        )
    if marker is not None and (
        not isinstance(marker, str) or marker not in CYCLE_GAS_PROPERTY_BASES
    ):
        basis = installed if installed in CYCLE_GAS_PROPERTY_BASES else (
            CYCLE_GAS_PROPERTY_BASIS_LEGACY
        )
        return basis, alpha_basis, diagnostics, (
            f"unknown cycle gas_property_basis marker {marker!r}; "
            f"expected one of {CYCLE_GAS_PROPERTY_BASES!r}"
        )
    if metadata is None:
        if marker is not None:
            return marker, alpha_basis, None, (
                "cycle gas_property_basis marker is present but installed газ "
                "wiring has no cycle construction metadata; rebuild the namespace"
            )
        return (
            CYCLE_GAS_PROPERTY_BASIS_LEGACY,
            ALPHA_BASIS_LEGACY,
            None,
            None,
        )
    if marker is not None and marker != installed:
        return installed, alpha_basis, diagnostics, (
            f"cycle gas_property_basis marker {marker!r} does not match "
            f"installed cycle wiring {installed!r}; rebuild the namespace "
            "instead of mutating the marker"
        )
    return installed, alpha_basis, diagnostics, None


SCENARIOS = {
    "A87_strict":  dict(a2=[(70, 90), (70, 90), (80, 90)], lD=(None, 1 / 3.5)),
    "A87_relaxed": dict(a2=[(70, 90), (70, 90), (80, 90)], lD=(None, 1 / 3)),
    "A58_strict":  dict(a2=[(70, 90), (70, 90), (80, 90)], lD=(1 / 4, 1 / 3.5)),
    "A58_relaxed": dict(a2=[(70, 90), (70, 90), (80, 90)], lD=(1 / 4, 1 / 3)),
    "B87_strict":  dict(a2=[(20, 90), (70, 90), (80, 90)], lD=(None, 1 / 3.5)),
    "B87_relaxed": dict(a2=[(20, 90), (70, 90), (80, 90)], lD=(None, 1 / 3)),
    "B58_strict":  dict(a2=[(20, 90), (70, 90), (80, 90)], lD=(1 / 4, 1 / 3.5)),
    "B58_relaxed": dict(a2=[(20, 90), (70, 90), (80, 90)], lD=(1 / 4, 1 / 3)),
}

BOUNDS = [
    (60e3, 95e3),   # n
    (0.040, 0.090),  # D_вт, м
    (0.50, 1.20),   # w1
    (0.50, 1.20),   # w2
    (11.0, 20.0),   # α1_1
    (11.0, 20.0),   # α1_2
    (15.0, 24.0),   # α1_3
    (1.00, 1.40),   # κ1
    (1.00, 1.35),   # κ2
    (1.00, 1.35),   # κ3
]
# контрольная точка из ручной сетки (n=84k, D=65, w=0.75/0.80);
# α1_1 поднят до 11 (книжное ограничение). Это только старт, не результат.
WARM = np.array([84e3, 0.065, 0.75, 0.80, 11.0, 11.0, 19.0, 1.38, 1.12, 1.15])


def _линейно(x, xp, fp):
    """np.interp для одной точки; те же операции, бит-в-бит как у numpy."""
    if x <= xp[0]:
        return fp[0]
    if x >= xp[-1]:
        return fp[-1]
    i = bisect.bisect_right(xp, x) - 1
    return (fp[i + 1] - fp[i]) / (xp[i + 1] - xp[i]) * (x - xp[i]) + fp[i]


def _graph_diagnostic(rho: float) -> tuple[float | None, str, str]:
    """Необязательный оптимум (u/c0) по рис. 18 без экстраполяции.

    Выход за оцифрованный диапазон ρ — это ``NOT_CHECKED``, а не отказ
    физического расчёта. Неожиданные ошибки не перехватываются: они должны
    остаться видимыми, а уже посчитанная физика — сохраниться вызывающим кодом.
    """

    try:
        return float(cs.uc0_opt_from_rho(rho)), "ok", ""
    except ValueError as exc:
        return None, "NOT_CHECKED", str(exc)


def _const_hub_provenance(
    st: list[dict],
    requested_a1,
    gas_property_basis: str = GAS_PROPERTY_BASIS_LEGACY,
    *,
    cycle_gas_property_basis: str = CYCLE_GAS_PROPERTY_BASIS_LEGACY,
    cycle_alpha_basis: str = ALPHA_BASIS_LEGACY,
    cycle_method_diagnostics: object | None = None,
) -> dict:
    """Запрошенные и фактические параметры const-hub кандидата.

    ``ρ`` здесь не вход, а результат внутренней итерации; ``α1`` передаётся
    явно. Ветка не запускает ячейку-14 ``ступень_авто``, поэтому её инженерные
    подборы ρ/α1 не приписываются этому результату. При этом const-hub сам
    является инженерным замыканием, а не подтверждением RAW.

    ``stage_gas_property_basis`` — основа ``cp``/``k`` именно ступени и
    давления. ``cycle_gas_property_basis`` — отдельно выбранная и фактически
    использованная основа цикла; ``cycle_alpha_basis`` и
    ``cycle_method_diagnostics`` берутся из того же построения. Опции не
    связаны: разрешён как stage-only, так и cycle-only диагностический вариант;
    по умолчанию обе остаются ``legacy``. Для любого неlegacy варианта
    добавляется ``NOT_PHYSICALLY_VALIDATED``: это диагностика метода, а не
    подтверждение полной физики. Охлаждение учитывается только балансом
    расхода, без смешения температур.
    """

    actual_a1 = [float(s["α1"]) for s in st]
    actual_rho = [float(s["ρ"]) for s in st]
    stage_textbook = gas_property_basis == GAS_PROPERTY_BASIS_TEXTBOOK
    cycle_textbook = (
        cycle_gas_property_basis == CYCLE_GAS_PROPERTY_BASIS_TEXTBOOK
    )
    textbook = stage_textbook or cycle_textbook
    if stage_textbook and cycle_textbook:
        model_variant = TEXTBOOK_BOTH_DIAGNOSTIC_MODEL_VARIANT
    elif stage_textbook:
        model_variant = TEXTBOOK_DIAGNOSTIC_MODEL_VARIANT
    elif cycle_textbook:
        model_variant = TEXTBOOK_CYCLE_DIAGNOSTIC_MODEL_VARIANT
    else:
        model_variant = "extended_const_hub"
    fuel_relation = None
    if isinstance(cycle_method_diagnostics, dict):
        fuel_relation = cycle_method_diagnostics.get("fuel", {}).get("relation")
    if fuel_relation is None:
        fuel_relation = "source calc_g_m mean-cp balance at T_ref"
    provenance = {
        "model_variant": model_variant,
        "stage_gas_property_basis": gas_property_basis,
        "cycle_gas_property_basis": cycle_gas_property_basis,
        "cycle_alpha_basis": cycle_alpha_basis,
        "cycle_fuel_relation": fuel_relation,
        "cycle_method_diagnostics": cycle_method_diagnostics,
        "a1_requested": [float(v) for v in requested_a1],
        "a1_actual": actual_a1,
        "rho_requested": None,
        "rho_actual": actual_rho,
        "engineering_adjustment": {
            "active": False,
            "legacy_cell14_repair": False,
            "model_construction": ["const_hub_reaction_fixed_point"],
            "mechanisms": [],
            "requested": {"a1": [float(v) for v in requested_a1], "rho": None},
            "actual": {"a1": actual_a1, "rho": actual_rho},
            "note": (
                "ρ получен внутренней итерацией const-hub, а не входом; "
                "ячейка-14 ступень_авто здесь не используется. "
                "Это не утверждение о полном RAW."
            ),
        },
    }
    if textbook:
        provenance["NOT_PHYSICALLY_VALIDATED"] = True
        if stage_textbook and cycle_textbook:
            diagnostic_label = (
                "textbook_true_cp_cycle_and_stage_properties_"
                "not_physically_validated"
            )
        elif stage_textbook:
            # Preserve the pre-existing stage-only label.
            diagnostic_label = (
                "textbook_true_cp_stage_properties_not_physically_validated"
            )
        else:
            diagnostic_label = (
                "textbook_true_cp_cycle_properties_not_physically_validated"
            )
        provenance["gas_property_diagnostic_label"] = diagnostic_label
        provenance["cycle_assumptions"] = {
            "cooling": "cooling_bookkeeping_only_no_temperature_mixing",
            "physical_validation": "NOT_PHYSICALLY_VALIDATED",
        }
    return provenance


def build_ns(m: oc.Model, q_n_limit: float = 0.995, sigma_lim: float = SIGMA_LIM,
             fast: bool = True, orchestration_core: bool = True, *,
             gas_property_basis: str = GAS_PROPERTY_BASIS_LEGACY,
             cycle_gas_property_basis: str = CYCLE_GAS_PROPERTY_BASIS_LEGACY) -> tuple[dict, dict]:
    """Пространство имён модели + функции const-hub.

    ``orchestration_core=True`` (по умолчанию) сразу после исполнения
    исходных ячеек подменяет четыре физические функции их извлечёнными
    адаптерами: ``газ``/``H0_по_p`` (ячейка 7), ``ступень`` (ячейка 12) и
    ``ρ_корня`` (ячейка 14). Следующие ячейки (16 и 20) читают подменённые
    определения из пространства имён так же, как в
    ``optimizer_core._evaluate(stage_core=True)``. Дополнительно заменяются
    три функции ``ступень_const_hub`` / ``турбина_const_hub`` /
    ``распределить_π_const_hub`` на извлечённую реализацию
    ``turbine_engine.orchestration``. Ячейка блокнота всё равно исполняется:
    её ``ρ_таблица`` используется извлечённой реализацией, а при
    ``orchestration_core=False`` её определения остаются независимым
    эталоном (никаких подмен). Верхний семиточечный перебор ``D_hub`` из
    ячейки в обоих случаях исключён (источник обрезается до ``rows = []``).

    ``gas_property_basis`` — явная опция выбора основы ``cp``/``k`` для
    ступени, ``H0_по_p`` и функции ``газ``. Допустимы только
    ``legacy_derivative`` (по умолчанию: исходная notebook-аппроксимация
    производной ``calc_cp``/среднего ``cp``, поведение не меняется) и
    диагностическая ``textbook_true_cp`` (опубликованный прямой fit
    ``cp(T)``). Выбор фиксируется в этом собранном пространстве имён
    (``ns["_gas_property_basis"]`` и атрибут установленных адаптеров);
    не меняйте маркер на месте — для другой основы пересоберите пространство
    имён, иначе ``eval_candidate`` отклонит несовпадение до физического
    расчёта. Кэши чистых функций принадлежат только этому построению и между
    построениями не разделяются. ``textbook_true_cp`` требует
    ``orchestration_core=True``: эталонная ветка блокнота не имеет такой
    подмены.

    ``cycle_gas_property_basis`` — независимая keyword-only опция метода
    цикла. По умолчанию ``legacy_mean_cp`` сохраняет прежний расчёт;
    ``textbook_true_cp`` использует тот же явно выбранный вариант в
    ``oc.cycle_point`` и переносит его ``G``, ``g_t``, ``alpha``, ``L_k``,
    давления и ``c_t`` во всё построение. Любая textbook-опция (цикл или
    ступень) требует ``orchestration_core=True``. Основы не связываются
    автоматически: stage-only и cycle-only варианты остаются явными.
    """
    if (not isinstance(gas_property_basis, str)
            or gas_property_basis not in GAS_PROPERTY_BASES):
        raise ValueError(
            f"unknown gas_property_basis {gas_property_basis!r}; "
            f"expected one of {GAS_PROPERTY_BASES!r}"
        )
    if (not isinstance(cycle_gas_property_basis, str)
            or cycle_gas_property_basis not in CYCLE_GAS_PROPERTY_BASES):
        raise ValueError(
            f"unknown cycle gas_property_basis {cycle_gas_property_basis!r}; "
            f"expected one of {CYCLE_GAS_PROPERTY_BASES!r}"
        )
    if ((gas_property_basis == GAS_PROPERTY_BASIS_TEXTBOOK
         or cycle_gas_property_basis == CYCLE_GAS_PROPERTY_BASIS_TEXTBOOK)
            and not orchestration_core):
        raise ValueError(
            "textbook_true_cp (stage or cycle) requires "
            "orchestration_core=True; the notebook reference branch has no "
            "textbook gas/property wiring"
        )
    if cycle_gas_property_basis == CYCLE_GAS_PROPERTY_BASIS_LEGACY:
        # Preserve the historical call shape on the default path.
        c = oc.cycle_point(m, 11.0, 1250.0, 110e3)
    else:
        c = oc.cycle_point(
            m, 11.0, 1250.0, 110e3,
            gas_property_basis=cycle_gas_property_basis,
        )
    ns = dict(m.base)
    ns["_gas_property_basis"] = gas_property_basis
    ns["_cycle_gas_property_basis"] = c.get(
        "cycle_gas_property_basis", CYCLE_GAS_PROPERTY_BASIS_LEGACY
    )
    ns["T_г_полн"] = 1250.0
    ns["охлаждение_лопаток"] = c["охлаждение"]
    ns["g_охл"] = c["g_ohl"]
    ns["g_воз"] = c["g_voz"]
    exec(compile(m.code[m.param_idx], "параметры", "exec"), ns)
    ns["q_n"] = float(q_n_limit)
    ns["σ_л"] = float(sigma_lim)
    ns["δ_отн"], ns["δ_мин"] = 0.005, 0.2e-3
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
    Ц["T_г_полн"] = 1250.0
    ns["Ц"] = Ц
    ns.update({"D_ст": [0.67, 1.0, 1.0], "l_т_D_т": 0.250586,
               "α1_ст": [11.0, 11.0, 19.0], "π_веса": [0.91, 0.89, 1.0],
               "κ_ст": [1.38, 1.12, 1.15], "φ": 0.97, "ψ": 0.96})
    gas_cell = stage_cell = reaction_cell = None
    if orchestration_core:
        # семантические маркеры — те же, что в optimizer_core._evaluate
        gas_cell = next(
            (i for i in m.core_idx
             if "def газ(" in m.code[i] and "def H0_по_p(" in m.code[i]),
            None,
        )
        stage_cell = next((i for i in m.core_idx if "def ступень(" in m.code[i]), None)
        reaction_cell = next(
            (i for i in m.core_idx if "def ρ_корня(" in m.code[i]), None
        )
        if gas_cell is None:
            raise RuntimeError("notebook gas cell (cell 7) not found in core_idx")
        if stage_cell is None:
            raise RuntimeError("notebook stage cell not found in core_idx")
        if reaction_cell is None:
            raise RuntimeError("notebook root-reaction cell (cell 14) not found in core_idx")
    for i in m.core_idx:
        exec(compile(m.code[i], f"ядро {i}", "exec"), ns)
        # те же адаптеры и тот же порядок, что в _evaluate(stage_core=True):
        # ячейки 16/20 дальше видят уже подменённые определения
        if orchestration_core and i == gas_cell:
            if gas_property_basis == GAS_PROPERTY_BASIS_LEGACY:
                # Keep the historical no-keyword factory calls so existing
                # adapter/mock seams observe exactly the same construction.
                ns["газ"] = oc._газ_ядро(ns)
                ns["H0_по_p"] = oc._h0_по_p_ядро(ns)
            else:
                ns["газ"] = oc._газ_ядро(
                    ns, gas_property_basis=gas_property_basis
                )
                ns["H0_по_p"] = oc._h0_по_p_ядро(
                    ns, gas_property_basis=gas_property_basis
                )
            # Cycle metadata is construction-bound on the installed gas
            # closure, separately from the stage basis above.
            oc._mark_cycle_metadata(ns["газ"], c)
        if orchestration_core and i == stage_cell:
            if gas_property_basis == GAS_PROPERTY_BASIS_LEGACY:
                ns["ступень"] = oc._ступень_ядро(ns)
            else:
                ns["ступень"] = oc._ступень_ядро(
                    ns, gas_property_basis=gas_property_basis
                )
        if orchestration_core and i == reaction_cell:
            ns["ρ_корня"] = oc._ρ_корня_ядро(ns)
    if not orchestration_core:
        # Reference branch: keep the notebook functions, but record the
        # construction-time basis on them so the mutable namespace marker
        # cannot mislabel this namespace later.
        for _name in ("газ", "H0_по_p", "ступень"):
            _function = ns.get(_name)
            if _function is not None and _installed_gas_property_basis(_function) is None:
                setattr(
                    _function,
                    oc.GAS_PROPERTY_BASIS_ATTR,
                    gas_property_basis,
                )
        if ns.get("газ") is not None:
            oc._mark_cycle_metadata(ns["газ"], c)
    # базовый расход — из цикла, не из старого G-фита
    ns["G_в"] = c["G"]
    ns["G_c1"] = c["G"] * c["g_t"]
    ns["G_л_ост"] = c["G"] * (c["g_t"] + c["g_voz"])
    # фиксированный базовый расход: каждый кандидат стартует отсюда
    ns["G_base"] = (c["G"], c["G"] * c["g_t"], c["G"] * (c["g_t"] + c["g_voz"]))
    # функции const-hub из блокнота
    cell = next(x for x in m.nb["cells"] if x.get("id") == "const-hub-scan")
    src = "".join(cell["source"]).split("rows = []")[0]
    if fast:  # тот же линейный интерполянт, без накладных расходов np.interp
        src = src.replace("np.interp(l_l / D_л_n, *ρ_таблица)",
                          "_линейно(l_l / D_л_n, *ρ_таблица)")
    exec(compile(src, "const_hub", "exec"), ns)
    if fast:
        # чистые функции от своих аргументов: кэш не меняет результат,
        # только убирает повторный счёт одинаковых (T0, H0) внутри итераций
        ns["_линейно"] = _линейно
        ns["газ"] = lru_cache(maxsize=8192)(ns["газ"])
        ns["cp_k_ступени"] = lru_cache(maxsize=8192)(ns["cp_k_ступени"])
        ns["H0_по_p"] = lru_cache(maxsize=8192)(ns["H0_по_p"])
    if orchestration_core:
        # те же формулы/порядок/допуски; зависимости читаются из ns на каждом
        # вызове, поэтому π_веса/α1_ст/κ_ст/G_* и подменённый ступень видны
        # ровно так же, как у блокнотных функций.
        orchestration = build_const_hub_orchestration(
            ns,
            interp=_линейно if fast else np.interp,
            clip=np.clip,
        )
        orchestration.install()
    return ns, c


def eval_candidate(x, ns: dict, scenario: str, q_n_limit: float, soft: bool = False) -> dict:
    x = np.asarray(x, dtype=float)
    n, Dh, w1, w2, a11, a12, a13, k1, k2, k3 = map(float, x)
    sc = SCENARIOS[scenario]
    lD_lo, lD_hi = sc["lD"]
    gas_basis, gas_basis_error = _gas_property_basis_state(ns)
    cycle_basis, cycle_alpha_basis, cycle_diagnostics, cycle_basis_error = (
        _cycle_gas_property_basis_state(ns)
    )
    requested_a1 = (a11, a12, a13)
    st: list[dict] | None = None
    opt: list[float | None] = []
    opt_status: list[str] = []
    opt_reason: list[str] = []
    try:
        if gas_basis_error:
            raise ValueError(gas_basis_error)
        if cycle_basis_error:
            raise ValueError(cycle_basis_error)
        ns["π_веса"] = [w1, w2, 1.0]
        ns["α1_ст"] = [a11, a12, a13]
        ns["κ_ст"] = [k1, k2, k3]
        # каждый кандидат — от фиксированного расхода цикла, не от предыдущего
        baseG = tuple(ns["G_base"]) if "G_base" in ns else (ns["G_в"], ns["G_c1"], ns["G_л_ост"])
        # масштабирование расхода до N=110 кВт на входе редуктора
        # (η_р не входит в целевую плоскость)
        converged = False
        for _ in range(20):
            ns["G_в"], ns["G_c1"], ns["G_л_ост"] = baseG
            _, st = ns["распределить_π_const_hub"](n, Dh, ns["π_т"] ** (1 / ns["z"]))
            N = (sum(s["G_c"] * s["L_т"] for s in st) * ns["η_м"]
                 - ns["G_в"] * ns["L_к"])
            if not np.isfinite(N) or N <= 0:
                raise FloatingPointError(f"N={N}")
            if abs(N / N_TARGET - 1.0) < 1e-12:
                converged = True
                break
            baseG = tuple(v * (N_TARGET / N) for v in baseG)
        if not converged:
            # последний пересчёт нужен только если сходимость не достигнута
            ns["G_в"], ns["G_c1"], ns["G_л_ост"] = baseG
            _, st = ns["распределить_π_const_hub"](n, Dh, ns["π_т"] ** (1 / ns["z"]))
            N = (sum(s["G_c"] * s["L_т"] for s in st) * ns["η_м"]
                 - ns["G_в"] * ns["L_к"])
        lc1 = st[0]["l_c"]
        a2 = [s["α2"] for s in st]
        sig_all = [s["σ_р"] for s in st]
        sig = max(sig_all)
        u = max(max(s["u1"], s["u2"]) for s in st)
        lD = st[-1]["l_л"] / st[-1]["D_л"]
        A_all = [s["A_лa"] for s in st]
        n_max_all = min(math.sqrt(SIGMA_LIM / (K_N * A)) for A in A_all if A > 0)
        qn = n / n_max_all
        qn_sigma = max(math.sqrt(max(s, 0.0) / SIGMA_LIM) for s in sig_all)
        sig_cons = max(abs(sig_all[i] - K_N * n * n * A_all[i]) / max(sig_all[i], 1e-9)
                       for i in range(len(st)))
        # η_t* по полным параметрам (не η_т_полн_цикл)
        g_i = [s["G_c"] / ns["G_c1"] for s in st]
        L_sum = sum(g * s["L_т"] for g, s in zip(g_i, st))
        T_out, p_out = st[-1]["T_ст_п"], st[-1]["p_ст_п"]
        cp_t, k_t = ns["газ"](0.5 * (ns["T_г_полн"] + T_out))
        H_t = cp_t * ns["T_г_полн"] * (1 - (p_out / ns["p_г_полн"]) ** ((k_t - 1) / k_t))
        eta_t = L_sum / H_t if H_t > 0 else float("nan")
        eta_below = max(0.0, 0.90 - eta_t) / 0.90
        eta_ok = bool(eta_t >= 0.90 - 1e-9)
        uc0 = [0.5 * (s["u1"] + s["u2"]) / math.sqrt(2 * s["H0"]) for s in st]
        opt = []
        opt_status = []
        opt_reason = []
        for s in st:
            value, status, reason = _graph_diagnostic(s["ρ"])
            opt.append(value)
            opt_status.append(status)
            opt_reason.append(reason)
        uc_dev = [
            (abs(v / o - 1.0) if o is not None else None)
            for v, o in zip(uc0, opt)
        ]
        missing_opt = [i for i, value in enumerate(opt) if value is None]
        rho_cp = [s["ρ"] for s in st]
        rho_pen = sum(max(0.0, 0.30 - r) / 0.30 + max(0.0, r - 0.35) / 0.35 for r in rho_cp)
        # корневая реактивность — только α1=const (ст.1/2); ст.3 Γ — NOT_CHECKED
        rho_root = []
        for s in st[:2]:
            Dm, Dhub = s["D_c"], s["D_c"] - s["l_c"]
            a = math.radians(s["α1"])
            rho_root.append(1.0 - (1.0 - s["ρ"]) * (Dm / Dhub) ** (2 * math.cos(a) ** 2))
        cont = max(abs(math.pi * s["D_л"] * s["l_л"] - s["G_л"] * s["v2"] / s["c2a"])
                   / (math.pi * s["D_л"] * s["l_л"]) for s in st)
        en = []
        for s in st:
            T2p = s["T2"] + s["c2"] ** 2 / (2 * s["cp"])
            en.append(abs(s["L_u"] - s["cp"] * (s["T0"] - T2p)) / max(abs(s["L_u"]), 1e-9))
        en_max = max(en)
        p2z = abs(st[-1]["p2"] / ns["p_т"] - 1.0)
        # NaN-защита: критические физические величины должны быть конечными.
        # Необязательная диагностика uc0_opt сюда не входит: её отсутствие —
        # NOT_CHECKED, а не физический/численный отказ.
        crit = [lc1, sig, u, lD, qn, qn_sigma, N, cont, en_max, p2z, eta_t,
                *a2, *rho_cp, *rho_root, *uc0]
        if not all(np.isfinite(v) for v in crit):
            bad = [name for name, v in zip(
                ["lc1", "sig", "u", "lD", "qn", "qn_sigma", "N", "cont", "en", "p2z", "eta_t"],
                [lc1, sig, u, lD, qn, qn_sigma, N, cont, en_max, p2z, eta_t]) if not np.isfinite(v)]
            return dict(score=1e9, score1=1e9, score2=1e9, feasible=False, hard=1e9,
                        status="numerical", err=f"non-finite: {bad}",
                        x=x, n=n, D_hub=Dh, w1=w1, w2=w2, a1=(a11, a12, a13), k=(k1, k2, k3),
                        lc1=float("nan"), a2=[float("nan")] * 3, sigma=float("nan"),
                        u=float("nan"), lD=float("nan"), qn=float("nan"), N=float("nan"),
                        eta_t=float("nan"), eta_ok=False, eta_below=float("nan"),
                        uc0=[], uc0_opt=opt, rho=[], rho_root=[], cont=float("nan"),
                        energy=float("nan"), p2z=float("nan"), violations=[],
                        hard_max=1e9, violation_count=0,
                        stages=[dict(s) for s in st],
                        uc0_opt_status=opt_status,
                        uc0_opt_reason=opt_reason,
                        scoring_status="not_scored",
                        **_const_hub_provenance(
                            st, requested_a1, gas_basis,
                            cycle_gas_property_basis=cycle_basis,
                            cycle_alpha_basis=cycle_alpha_basis,
                            cycle_method_diagnostics=cycle_diagnostics,
                        ))
        # нарушения: имя, значение, предел, нормированная величина
        violations = []

        def vadd(name, value, limit, amount):
            violations.append(dict(name=name, value=float(value),
                                   limit=str(limit), amount=float(amount)))

        for i, (v, (lo, hi)) in enumerate(zip(a2, sc["a2"]), 1):
            if v < lo:
                vadd(f"α2_{i} < {lo:g}", v, f">={lo:g}", (lo - v) / lo)
            elif v > hi:
                vadd(f"α2_{i} > {hi:g}", v, f"<={hi:g}", (v - hi) / hi)
        if lc1 < LC1_MIN:
            vadd("l_c1 < 14 мм", lc1 * 1e3, ">=14 мм", (LC1_MIN - lc1) / LC1_MIN)
        if sig > SIGMA_LIM:
            vadd("σ > 300 МПа", sig, "<=300 МПа", (sig - SIGMA_LIM) / SIGMA_LIM)
        if qn > q_n_limit:
            vadd("q_n > лимит", qn, f"<={q_n_limit:g}", (qn - q_n_limit) / q_n_limit)
        if abs(qn - qn_sigma) / max(qn, 1e-9) > 1e-3:
            vadd("q_n != √(σ/300)", qn, f"√(σ/300)={qn_sigma:.5f}",
                 abs(qn - qn_sigma) / max(qn, 1e-9))
        if sig_cons > 1e-9:
            vadd("σ != K_N n² A", sig_cons, "<=1e-9", sig_cons)
        if u > U_MAX:
            vadd("u > 400 м/с", u, "<=400", (u - U_MAX) / U_MAX)
        if abs(N / N_TARGET - 1.0) > 1e-4:
            vadd("N != 110 кВт (вход редуктора)", N / 1e3, "110 кВт", abs(N / N_TARGET - 1.0))
        if p2z > 1e-6:
            vadd("p2z != p_т", p2z, "<=1e-6", p2z)
        if cont > 1e-6:
            vadd("неразрывность", cont, "<=1e-6", cont)
        if en_max > 1e-6:
            vadd("энергия", en_max, "<=1e-6", en_max)
        for i, r in enumerate(rho_root, 1):
            if r < 0.05:
                vadd(f"ρ_корня_{i} < 0.05", r, ">=0.05", (0.05 - r) / 0.05)
        if lD_hi is not None and lD > lD_hi:
            vadd("l_t/D_t > верх", lD, f"<={lD_hi:.4f}", (lD - lD_hi) / lD_hi)
        if lD_lo is not None and lD < lD_lo:
            vadd("l_t/D_t < низ", lD, f">={lD_lo:.4f}", (lD_lo - lD) / lD_lo)
        hard = max((v["amount"] for v in violations), default=0.0)
        # Фаза 1: только hard. Фаза 2: hard=0 + η_t* + max l_c1.
        score1 = hard
        # фаза 2: чистая max l_c1 при η_t*≥0.90 и hard=0.
        # u/c0 и ρ_ср — диагностика; в цель входят только с --soft.
        score2 = (1e6 * hard + 1e6 * eta_below
                  + 1.0 * (1.0 - lc1 / 0.020))
        scoring_status = "ok"
        score2_soft_applied = False
        if soft:
            score2_soft_applied = True
            score2 += 1.0 * sum(d for d in uc_dev if d is not None) + 0.2 * rho_pen
            if missing_opt:
                # Не превращаем отсутствие диагностики в хороший soft-балл:
                # такие точки не должны выигрывать условное ранжирование.
                score2 += SOFT_MISSING_DIAGNOSTIC_PENALTY * len(missing_opt)
                scoring_status = "unavailable_diagnostic"
        return dict(
            score=score1, score1=float(score1), score2=float(score2),
            feasible=bool(hard <= 1e-9), hard=float(hard), hard_max=float(hard),
            violation_count=int(sum(1 for v in violations if v["amount"] > 1e-9)),
            status="ok", x=x,
            n=n, D_hub=Dh, w1=w1, w2=w2, a1=(a11, a12, a13), k=(k1, k2, k3),
            lc1=lc1, a2=a2, sigma=sig, u=u, lD=lD, qn=qn, N=N,
            eta_t=eta_t, eta_ok=eta_ok, eta_below=eta_below,
            uc0=uc0, uc0_opt=opt, rho=rho_cp, rho_root=rho_root,
            cont=cont, energy=en_max, p2z=p2z,
            violations=violations, err="",
            stages=[dict(s) for s in st],
            uc0_opt_status=opt_status,
            uc0_opt_reason=opt_reason,
            uc0_opt_missing=missing_opt,
            scoring_status=scoring_status,
            soft_score_applied=score2_soft_applied,
            **_const_hub_provenance(
                st, requested_a1, gas_basis,
                cycle_gas_property_basis=cycle_basis,
                cycle_alpha_basis=cycle_alpha_basis,
                cycle_method_diagnostics=cycle_diagnostics,
            ),
        )
    except Exception as e:  # noqa: BLE001
        return dict(score=1e9, score1=1e9, score2=1e9, feasible=False, hard=1e9,
                    status="model_error", err=f"{type(e).__name__}: {e}",
                    x=x, n=n, D_hub=Dh, w1=w1, w2=w2, a1=(a11, a12, a13), k=(k1, k2, k3),
                    lc1=float("nan"), a2=[float("nan")] * 3, sigma=float("nan"),
                    u=float("nan"), lD=float("nan"), qn=float("nan"), N=float("nan"),
                    eta_t=float("nan"), eta_ok=False, eta_below=float("nan"),
                    uc0=[], uc0_opt=opt, rho=[], rho_root=[], cont=float("nan"),
                    energy=float("nan"), p2z=float("nan"), violations=[],
                    hard_max=1e9, violation_count=0,
                    stages=[dict(s) for s in (st or [])],
                    uc0_opt_status=opt_status,
                    uc0_opt_reason=opt_reason,
                    scoring_status="not_scored",
                    **_const_hub_provenance(
                        st or [], requested_a1, gas_basis,
                        cycle_gas_property_basis=cycle_basis,
                        cycle_alpha_basis=cycle_alpha_basis,
                        cycle_method_diagnostics=cycle_diagnostics,
                    ))


def run_scenario(m, scenario: str, seeds: int, maxiter: int, popsize: int,
                 q_n_limit: float, soft: bool = False) -> list[dict]:
    # σ_л в ns = предел, эквивалентный выбранному q_n (legacy-путь; модель его не читает)
    ns, _ = build_ns(m, q_n_limit, q_n_limit ** 2 * SIGMA_LIM)
    rows: list[dict] = []
    lo = np.array([b[0] for b in BOUNDS])
    hi = np.array([b[1] for b in BOUNDS])

    def f1(x):
        r = eval_candidate(x, ns, scenario, q_n_limit, soft=soft)
        r["phase"] = 1
        r["seed"] = cur_seed
        r["scenario"] = scenario
        rows.append(r)
        return r["score1"]

    def f2(x):
        r = eval_candidate(x, ns, scenario, q_n_limit, soft=soft)
        r["phase"] = 2
        r["seed"] = cur_seed
        r["scenario"] = scenario
        rows.append(r)
        return r["score2"]

    for seed in range(1, seeds + 1):
        cur_seed = seed
        rng = np.random.default_rng(seed)
        size = max(popsize * len(BOUNDS), 2)
        init = rng.uniform(lo, hi, size=(size, len(BOUNDS)))
        init[0] = np.clip(WARM, lo, hi)
        differential_evolution(
            f1, BOUNDS, seed=seed, maxiter=maxiter, popsize=popsize,
            init=init, polish=False, workers=1, updating="immediate", tol=1e-3,
        )
        feas = [r for r in rows if r.get("seed") == seed and r["phase"] == 1 and r["feasible"]]
        if feas:
            # старт фазы 2: все допустимые точки фазы 1 попадают в начальную популяцию
            start = sorted(feas, key=lambda r: r["score2"])
            init2 = rng.uniform(lo, hi, size=(size, len(BOUNDS)))
            for j, r0 in enumerate(start[:size]):
                init2[j] = np.clip(r0["x"], lo, hi)
            differential_evolution(
                f2, BOUNDS, seed=seed + 1000, maxiter=max(2, maxiter // 2),
                popsize=popsize, init=init2, polish=False, workers=1,
                updating="immediate", tol=1e-3,
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="A87_strict",
                    help="A87_strict/A87_relaxed/A58_strict/A58_relaxed/"
                         "B87_strict/B87_relaxed/B58_strict/B58_relaxed/all")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--maxiter", type=int, default=2)
    ap.add_argument("--popsize", type=int, default=3)
    ap.add_argument("--q-n", type=float, default=0.995)
    ap.add_argument("--tag", default="", help="суффикс имени CSV, чтобы не перезаписать пилот")
    ap.add_argument("--soft", action="store_true", help="добавить u/c0 и ρ_ср в фазу 2 как предпочтения")
    args = ap.parse_args()
    m = oc.load_model()
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for name in names:
        if name not in SCENARIOS:
            raise SystemExit(f"unknown scenario {name}")
    all_rows: list[dict] = []
    for name in names:
        print(f"\n=== {name}: seeds={args.seeds}, DE {args.maxiter}x{args.popsize}, "
              f"q_n<={args.q_n} ===")
        rows = run_scenario(m, name, args.seeds, args.maxiter, args.popsize, args.q_n, args.soft)
        all_rows += rows
        ph1 = [r for r in rows if r["phase"] == 1]
        ph2 = [r for r in rows if r["phase"] == 2]
        ok1 = [r for r in ph1 if r["feasible"]]
        best1 = min(ph1, key=lambda r: r["score1"])
        print(f"фаза 1: оценено {len(ph1)}; допустимых {len(ok1)}")
        status = "допустим" if best1["feasible"] else "НЕДОПУСТИМ (лучший по hard)"
        print(f"best1 [{status}] hard={best1['hard']:.3e} "
              f"l_c1={best1['lc1']*1e3 if np.isfinite(best1['lc1']) else float('nan'):.2f} мм "
              f"α2={[round(v,1) for v in best1['a2']]} σ={best1['sigma']:.0f} "
              f"lD={best1['lD']:.3f} q_n={best1['qn']:.4f} N={best1['N']/1e3:.2f} "
              f"η_t*={best1['eta_t']:.4f}")
        for v in best1.get("violations", []):
            print(f"   нарушено: {v['name']} = {v['value']:.4g} (предел {v['limit']}), "
                  f"норм. {v['amount']:.3f}")
        if ok1:
            ok2 = [r for r in ph2 if r["feasible"] and r["eta_ok"]]
            label = "η_t*≥0.90"
            if not ok2:
                ok2 = [r for r in ph2 if r["feasible"]]
                label = "η_t*<0.90 (нет допустимых по η)"
            if ok2:
                best2 = min(ok2, key=lambda r: r["score2"])
                print(f"фаза 2: оценено {len(ph2)}; допустимых {len([r for r in ph2 if r['feasible']])}; "
                      f"по {label}: {len(ok2)}")
                print(f"best2: l_c1={best2['lc1']*1e3:.2f} мм; "
                      f"α2={[round(v,1) for v in best2['a2']]}; η_t*={best2['eta_t']:.4f}; "
                      f"σ={best2['sigma']:.0f}; l/D={best2['lD']:.3f}; q_n={best2['qn']:.4f}")
            else:
                print("фаза 2: допустимых точек нет")
        else:
            print("фаза 2 не запускалась: нет допустимых в фазе 1")
    suffix = f"_{args.tag}" if args.tag else ""
    out = RESULTS / f"extended_{args.scenario}_q{args.q_n:g}{suffix}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "seed", "phase", "status", "score1", "score2",
                    "feasible", "hard", "hard_max", "violation_count",
                    "n", "D_hub", "w1", "w2", "a1", "k", "lc1", "a2", "sigma",
                    "u", "lD", "qn", "N", "eta_t", "eta_ok", "eta_below",
                    "uc0", "uc0_opt", "rho", "rho_root", "cont", "energy", "p2z",
                    "violations", "err", "x", "model_variant", "a1_requested",
                    "a1_actual", "rho_actual", "uc0_opt_status", "uc0_opt_reason",
                    "scoring_status", "stages"])
        for r in all_rows:
            xstr = ";".join(f"{v:.17g}" for v in r.get("x", []))
            w.writerow([
                r["scenario"], r["seed"], r["phase"], r.get("status", ""),
                r.get("score1", r["score"]), r.get("score2", ""),
                r["feasible"], r["hard"], r.get("hard_max", r["hard"]),
                r.get("violation_count", ""),
                r["n"], r["D_hub"], r["w1"], r["w2"], r["a1"], r["k"], r["lc1"],
                r["a2"], r["sigma"], r["u"], r["lD"], r["qn"], r["N"],
                r.get("eta_t", ""), r.get("eta_ok", ""), r.get("eta_below", ""),
                r["uc0"], r["uc0_opt"], r["rho"], r["rho_root"],
                r["cont"], r["energy"], r["p2z"],
                "; ".join(f"{v['name']}:{v['amount']:.3f}" for v in r.get("violations", [])),
                r["err"], xstr,
                r.get("model_variant", ""),
                r.get("a1_requested", ""), r.get("a1_actual", ""),
                r.get("rho_actual", ""), r.get("uc0_opt_status", ""),
                r.get("uc0_opt_reason", ""), r.get("scoring_status", ""),
                json.dumps(r.get("stages", []), ensure_ascii=False, default=float),
            ])
    print(f"\nсохранено: {out}")


if __name__ == "__main__":
    main()
