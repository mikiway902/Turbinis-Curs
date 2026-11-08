"""Ядро параметрической оптимизации ТВД.

Использует существующую расчётную модель проекта:
  Turbinis-Cycle/Turbinis-Cycle.ipynb  - цикл;
  Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb - поступенчатый расчёт.

Новая физическая модель не создаётся: код ячеек блокнота исполняется так же,
как в mathcad/make_mathcad.py и podbor_parametrov.py. Здесь добавлены только:
  * пересчёт цикла для произвольных pi*к, T*г, N;
  * независимая валидация решения;
  * численная оптимизация (DE / случайный поиск / локальная доводка);
  * исправление неустойчивого внешнего итератора по расходу G (см. _patch_cell16).

Единицы СИ. Углы в градусах, где так принято в исходной модели.
"""
from __future__ import annotations

import json
import math
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
def cycle_point(m: Model, pi: float = 11.0, T_g: float = 1250.0, N: float = 110000.0) -> dict[str, Any]:
    """Пересчитывает цикл существующими функциями Turbinis-Cycle."""
    C = m.cycle
    keys = ("T_г_полн", "охлаждение_лопаток", "g_охл", "g_воз")
    old = {k: C[k] for k in keys if k in C}
    try:
        C["T_г_полн"] = float(T_g)
        охл = bool(T_g > 1250.0)
        C["охлаждение_лопаток"] = охл
        C["g_охл"] = 0.05 if охл else 0.015
        C["g_воз"] = 0.6 * C["g_охл"]
        d = C["calc_tvd"](float(pi))
    finally:
        for k, v in old.items():
            C[k] = v

    M = float(C["M_т"])
    k_t = C["calc_k_kerosene"](d["T_т"], d["α"])
    T_st = d["T_т"] / (1.0 + (k_t - 1.0) / 2.0 * M**2)
    c_t = M * math.sqrt(k_t * C["R"] * T_st)
    G = N / d["N_в_уд"]
    g_ohl = 0.05 if T_g > 1250.0 else 0.015
    g_voz = 0.6 * g_ohl
    C_e = d["g_m"] * (1.0 - C["g_ут"] - g_ohl) * 3600.0 / (d["N_в_уд"] / 1e3)
    eta_e = 3600.0 / (C_e * float(C["Q_н_р"]))
    p_g = float(C["σ_г"]) * float(C["σ_вх"]) * float(C["p_а_полн"]) * pi
    p_t = float(C["π_с"]) * float(C["p_а"])
    return {
        "pi": float(pi), "T_g": float(T_g), "N": float(N),
        "G": float(G), "p_g": p_g, "p_t": p_t, "c_t": float(c_t),
        "g_t": float(d["g_т"]), "alpha_g": float(d["α"]),
        "L_k": float(d["L_к"]), "L_t": float(d["L_т"]), "T_t": float(d["T_т"]),
        "eta_t_cycle": float(d["η_т"]), "N_ud": float(d["N_в_уд"]),
        "C_e": float(C_e), "eta_e": float(eta_e),
        "g_ohl": g_ohl, "g_voz": g_voz, "охлаждение": bool(T_g > 1250.0),
        "d": d,
    }


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
    "k": [1.12, 1.04, 1.02], "sigma": 285.0, "phi": 0.965, "psi": 0.95,
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
        "D1": (0.50, 1.00), "D2": (0.60, 1.00), "lD": (0.20, 1.0 / 3.0),
        "a11": (9.0, 20.0), "a12": (9.0, 20.0), "a13": (10.0, 24.0),
        "w1": (0.60, 1.20), "w2": (0.60, 1.20),
        "k1": (0.95, 1.40), "k2": (0.95, 1.30), "k3": (0.95, 1.30),
        "sigma": (250.0, 400.0), "pi": (6.0, 18.0), "N": (80000.0, 500000.0),
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
def evaluate(x, m: Model, cfg, full: bool = False) -> dict[str, Any]:
    p = декодировать(x, cfg)
    c = cycle_point(m, p["pi"], p["Tg"], p["N"])
    ns = dict(m.base)
    ns["T_г_полн"] = p["Tg"]
    ns["охлаждение_лопаток"] = c["охлаждение"]
    ns["g_охл"] = c["g_ohl"]
    ns["g_воз"] = c["g_voz"]
    exec(compile(m.code[m.param_idx], "параметры", "exec"), ns)
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
    for i in m.core_idx:
        exec(compile(m.code[i], f"ядро {i}", "exec"), ns)
    if full:
        for i in m.full_idx:
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
    return res


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
    add("u_т <= 400 м/с", f"{res['u_t']:.1f} м/с", "<= 400", res["u_t"] <= 400.0 + 1e-9, "hard",
        max(0.0, (res["u_t"] - 400.0) / 400.0))

    # длины лопаток
    lengths = [v for pair in zip(res["lc"], res["ll"]) for v in pair]
    ok_len = all(b >= a - 1e-12 for a, b in zip(lengths, lengths[1:]))
    add("длины лопаток не убывают", " -> ".join(f"{v*1e3:.1f}" for v in lengths), "не убывают", ok_len,
        "project", 0.0 if ok_len else 0.1)
    add("l_с1 >= 14 мм (проект)", f"{res['lc1']*1e3:.2f} мм", ">= 14 мм", res["lc1"] >= 0.014 - 1e-12, "project",
        max(0.0, (0.014 - res["lc1"]) / 0.014))
    add("l_с1 >= 20 мм (цель)", f"{res['lc1']*1e3:.2f} мм", ">= 20 мм",
        res["lc1"] >= cfg.get("lc1_target", 0.020), "target",
        max(0.0, (cfg.get("lc1_target", 0.020) - res["lc1"]) / cfg.get("lc1_target", 0.020)))

    # мощность, давление, невязки
    Nz = res["p"]["N"]
    add("N на валу = требуемая", f"{res['N_val']/1e3:.2f} кВт", f"{Nz/1e3:.1f} кВт",
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
        for i, v in enumerate(res.get("rho_root", []), 1):
            add(f"ρ корня ст.{i} >= 0.05", f"{v:.3f}", ">= 0.05", v >= 0.05, "hard",
                max(0.0, (0.05 - v) / 0.05))
        for i, v in enumerate(res.get("lambda_c1", []), 1):
            add(f"λ_c1 ст.{i} <= 1.05", f"{v:.3f}", "<= 1.05", v <= 1.05, "hard",
                max(0.0, (v - 1.05) / 1.05))
        # α1 у корня: книга стр. 73 (13…14°, малые турбины 11…12°)
        for i, v in enumerate(res.get("alpha1_root", []), 1):
            add(f"α1 у корня ст.{i} >= 11°", f"{v:.2f}°", ">= 11°", v >= 11.0 - 1e-9, "hard",
                max(0.0, (11.0 - v) / 11.0))
    return c


def penalty(checks, res, cfg) -> float:
    w = {"hard": 1000.0, "project": 100.0, "target": 1.0, "info": 0.0}
    p = sum(w[c["kind"]] * c["amount"] for c in checks if not c["ok"])
    p += 1.0 * max(0.0, 1.0 - res["lc1"] / cfg.get("lc1_target", 0.020))
    p += 0.2 * max(0.0, 1.0 - res["eta"] / cfg.get("eta_target", 0.90))
    return float(p)


def is_feasible(checks):
    return all(c["ok"] for c in checks if c["kind"] in ("hard", "project"))


def is_strict(checks):
    return all(c["ok"] for c in checks if c["kind"] in ("hard", "project", "target"))


# ---------------------------------------------------------------------------
# Поиск
# ---------------------------------------------------------------------------
@dataclass
class Archive:
    rows: list[dict] = field(default_factory=list)

    def add(self, res, checks, score):
        self.rows.append(dict(res=res, checks=checks, score=score,
                              feasible=is_feasible(checks), strict=is_strict(checks)))

    def best(self, strict=False):
        rows = [r for r in self.rows if (r["strict"] if strict else r["feasible"])]
        if not rows:
            rows = self.rows
        return min(rows, key=lambda r: r["score"])


def _score_x(x, m, cfg, full, archive):
    try:
        res = evaluate(x, m, cfg, full=full)
        checks = validate(res, cfg, full=full)
        sc = penalty(checks, res, cfg)
        archive.add(res, checks, sc)
        return sc
    except Exception:
        archive.rows.append(dict(res=None, checks=[], score=1e9, feasible=False, strict=False))
        return 1e9


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
