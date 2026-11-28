"""Этап 4: сетка (n, D_вт, w2, w1) 15×15×15×15 = 50 625 точек.

Остальные переменные — из лучшей точки B87_relaxed, кроме:
κ3 = 1.30, α1_3 = 23°, α1_2 = 20°.

Допустимость считается в двух категориях:
  full  — все ограничения ВКЛЮЧАЯ η_t* ≥ 0.90;
  hard0 — без η_t* (только для диагностики конфликта).

    python scan4d.py --workers 8
    python scan4d.py --report        # сводка по готовому scan4d.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import optimize_extended as oe
import analyze_extended as an
import optimizer_core as oc
import scan3d

RESULTS = oe.RESULTS
STEPS = 15
OVERRIDES = dict(k3=1.30, a13=23.0, a12=20.0)
IDX = scan3d.IDX
FIELDS = ["n", "D_vt", "w2", "w1", "status", "hard", "hard_no_strength",
          "ok_count", "lc1_mm", "rho_root_1", "rho_root_2", "a2_1", "a2_2", "a2_3",
          "sigma", "qn", "eta_t", "lD", "u", "N_kW", "l_ok", "rho_ok",
          "a2_ok_B", "a2_ok_A", "sigma_ok", "qn_ok", "eta_ok",
          "feasible", "feasible_A", "full", "full_A", "ok_no_strength",
          "violation_count", "binding", "violations"]


def config_point() -> np.ndarray:
    x = scan3d.base_point()
    for k, v in OVERRIDES.items():
        x[IDX[k]] = v
    return x


def eval4(x, ns, ql: float) -> dict:
    r = oe.eval_candidate(x, ns, "B87_relaxed", ql)
    a2 = (r["a2"] + [float("nan")] * 3)[:3]
    rr = (r["rho_root"] + [float("nan")] * 2)[:2]
    groups = {scan3d.group(v["name"]) for v in r["violations"]}
    nonstrength = [v for v in r["violations"] if scan3d.group(v["name"]) != "strength"]
    a2_ok_b = all(lo <= v <= hi for v, (lo, hi) in zip(a2, oe.SCENARIOS["B87_relaxed"]["a2"]))
    a2_ok_a = all(lo <= v <= hi for v, (lo, hi) in zip(a2, [(70, 90), (70, 90), (80, 90)]))
    l_ok = bool(r["lc1"] >= 0.014)
    rho_ok = bool(rr[0] >= 0.05 and rr[1] >= 0.05)
    sigma_ok = bool(r["sigma"] <= 300.0)
    qn_ok = bool(r["qn"] <= ql)
    eta_ok = bool(r["eta_t"] >= 0.90 - 1e-9)
    feasible = bool(r["feasible"])
    return {
        "n": float(x[0]), "D_vt": float(x[1]), "w2": float(x[3]), "w1": float(x[2]),
        "status": r["status"], "hard": r["hard"],
        "hard_no_strength": max((v["amount"] for v in nonstrength), default=0.0),
        "ok_count": 10 - len(groups),
        "lc1_mm": r["lc1"] * 1e3, "rho_root_1": rr[0], "rho_root_2": rr[1],
        "a2_1": a2[0], "a2_2": a2[1], "a2_3": a2[2],
        "sigma": r["sigma"], "qn": r["qn"], "eta_t": r["eta_t"],
        "lD": r["lD"], "u": r["u"], "N_kW": r["N"] / 1e3,
        "l_ok": l_ok, "rho_ok": rho_ok, "a2_ok_B": a2_ok_b, "a2_ok_A": a2_ok_a,
        "sigma_ok": sigma_ok, "qn_ok": qn_ok, "eta_ok": eta_ok,
        "feasible": feasible, "feasible_A": bool(feasible and a2_ok_a),
        "full": bool(feasible and eta_ok),
        "full_A": bool(feasible and a2_ok_a and eta_ok),
        "ok_no_strength": bool(nonstrength == []),
        "violation_count": r["violation_count"],
        "binding": (max(r["violations"], key=lambda v: v["amount"])["name"]
                    if r["violations"] else "—"),
        "violations": "; ".join(f"{v['name']}:{v['amount']:.3f}" for v in r["violations"]),
    }


_W: dict = {}


def _init(ql: float) -> None:
    m = oc.load_model()
    _W["ql"] = ql
    _W["x0"] = config_point()
    _W["ns"], _ = oe.build_ns(m, ql, ql ** 2 * oe.SIGMA_LIM)


def _work(nvals) -> list[dict]:
    dvals = np.linspace(*oe.BOUNDS[1], STEPS)
    w2vals = np.linspace(*oe.BOUNDS[3], STEPS)
    w1vals = np.linspace(*oe.BOUNDS[2], STEPS)
    out = []
    for nv in nvals:
        for dv in dvals:
            for w2v in w2vals:
                for w1v in w1vals:
                    x = _W["x0"].copy()
                    x[0], x[1], x[3], x[2] = nv, dv, w2v, w1v
                    out.append(eval4(x, _W["ns"], _W["ql"]))
    return out


def run(workers: int, ql: float) -> None:
    nvals = np.linspace(*oe.BOUNDS[0], STEPS)
    chunks = [nvals[i::workers] for i in range(workers)]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(ql,)) as ex:
        parts = list(ex.map(_work, chunks))
    rows = [r for part in parts for r in part]
    rows.sort(key=lambda r: (r["n"], r["D_vt"], r["w2"], r["w1"]))
    with (RESULTS / "scan4d.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({f: r.get(f, "") for f in FIELDS})
    print(f"scan4d: {len(rows)} точек -> scan4d.csv", flush=True)


def load() -> list[dict]:
    rows = []
    with (RESULTS / "scan4d.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k in ("n", "D_vt", "w2", "w1", "hard", "hard_no_strength", "lc1_mm",
                      "rho_root_1", "rho_root_2", "a2_1", "a2_2", "a2_3", "sigma",
                      "qn", "eta_t", "lD", "u", "N_kW"):
                r[k] = float(r[k])
            for k in ("l_ok", "rho_ok", "a2_ok_B", "a2_ok_A", "sigma_ok", "qn_ok",
                      "eta_ok", "feasible", "feasible_A", "full", "full_A",
                      "ok_no_strength"):
                r[k] = r[k] == "True"
            r["ok_count"] = int(float(r["ok_count"]))
            r["violation_count"] = int(float(r["violation_count"]))
            rows.append(r)
    return rows


def fnum(v, nd=4):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def point_line(r, tag):
    return (f"- **{tag}**: n={r['n']/1e3:.1f} тыс., D_вт={r['D_vt']*1e3:.1f} мм, "
            f"w2={r['w2']:.3f}, w1={r['w1']:.3f}; hard={r['hard']:.3f}, "
            f"l_c1={r['lc1_mm']:.2f} мм, ρ={fnum(r['rho_root_1'])}/{fnum(r['rho_root_2'])}, "
            f"α2={r['a2_1']:.1f}/{r['a2_2']:.1f}/{r['a2_3']:.1f}, "
            f"σ={r['sigma']:.0f}, q_n={r['qn']:.4f}, η_t*={r['eta_t']:.4f}; "
            f"связывающее: {r['binding']}")


CHECKS = [("l", "l_ok", "l_c1 ≥ 14 мм"), ("rho", "rho_ok", "ρ1,ρ2 ≥ 0.05"),
          ("a2", "a2_ok_B", "α2 в норме"), ("sig", "sigma_ok", "σ ≤ 300"),
          ("qn", "qn_ok", "q_n ≤ 0.995"), ("eta", "eta_ok", "η_t* ≥ 0.90")]


def report(ql: float) -> None:
    rows = load()
    ok = [r for r in rows if r["status"] == "ok"]
    lines = ["# Этап 4: сетка (n, D_вт, w2, w1)", "",
             f"50 625 точек (15⁴), ошибок {len(rows) - len(ok)}. "
             f"κ3={OVERRIDES['k3']}, α1_3={OVERRIDES['a13']}°, "
             f"α1_2={OVERRIDES['a12']}°, остальное — лучшая точка B87_relaxed.", "",
             "Две категории: **full** — все ограничения и η_t* ≥ 0.90; "
             "**hard0** — все ограничения фазы 1 без η (диагностика).", ""]
    lines += ["| категория | точек |", "|---|---|",
              f"| full (всё + η) | {sum(r['full'] for r in ok)} |",
              f"| full при строгих α2 (70/70/80) | {sum(r['full_A'] for r in ok)} |",
              f"| hard0 (всё без η) | {sum(r['feasible'] for r in ok)} |",
              f"| hard0 при строгих α2 | {sum(r['feasible_A'] for r in ok)} |", ""]
    if not any(r["feasible"] for r in ok):
        lines += ["Точек с `hard=0` нет: η не единственное, чего не хватает.", ""]
    # одиночные ограничения
    lines += ["## Отдельные ограничения", "", "| ограничение | точек |", "|---|---|"]
    for _, key, label in CHECKS:
        lines.append(f"| {label} | {sum(r[key] for r in ok)} |")
    lines.append("")
    # пары
    lines += ["## Пары ограничений", "",
              "| пара | точек |", "|---|---|"]
    for (n1, k1, _), (n2, k2, _) in itertools.combinations(CHECKS, 2):
        lines.append(f"| {n1} + {n2} | {sum(r[k1] and r[k2] for r in ok)} |")
    lines += ["", f"Все шесть вместе: {sum(all(r[k] for _, k, _ in CHECKS) for r in ok)}; "
              f"со строгими α2 (70/70/80): "
              f"{sum(all(r[k] for k in ('l_ok','rho_ok','a2_ok_A','sigma_ok','qn_ok','eta_ok')) for r in ok)}", ""]
    # лучшие точки
    best = min(ok, key=lambda r: r["hard"])
    lines += ["## Ближайшие к допустимости", "",
              "| # | hard | n | D_вт | w2 | w1 | l_c1 | ρ1/ρ2 | α2 | σ | η_t* | связывающее |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(sorted(ok, key=lambda r: (r["hard"], -r["lc1_mm"]))[:8], 1):
        lines.append(
            f"| {i} | {r['hard']:.3f} | {r['n']/1e3:.1f} | {r['D_vt']*1e3:.1f} | "
            f"{r['w2']:.3f} | {r['w1']:.3f} | {r['lc1_mm']:.2f} | "
            f"{r['rho_root_1']:.4f}/{r['rho_root_2']:.4f} | "
            f"{r['a2_1']:.1f}/{r['a2_2']:.1f}/{r['a2_3']:.1f} | "
            f"{r['sigma']:.0f} | {r['eta_t']:.4f} | {r['binding']} |")
    lines.append("")
    # лучшие по длине при разных наборах
    for keys, label in (
        (("l_ok", "rho_ok"), "max l при l≥14 и ρ ok"),
        (("l_ok", "rho_ok", "a2_ok_B"), "max l при l≥14, ρ, α2"),
        (("l_ok", "rho_ok", "a2_ok_B", "eta_ok"), "max l при l≥14, ρ, α2, η"),
    ):
        sub = [r for r in ok if all(r[k] for k in keys)]
        if sub:
            b = max(sub, key=lambda r: r["lc1_mm"])
            lines.append(point_line(b, label + f" ({len(sub)} точек)"))
        else:
            lines.append(f"- **{label}**: точек нет")
    lines.append("")
    # повторная проверка top-20
    m = oc.load_model()
    ns, _ = oe.build_ns(m, ql, ql ** 2 * oe.SIGMA_LIM)
    x0 = config_point()
    worst = 0.0
    for b in sorted(ok, key=lambda r: r["hard"])[:20]:
        x = x0.copy()
        x[0], x[1], x[3], x[2] = b["n"], b["D_vt"], b["w2"], b["w1"]
        r = oe.eval_candidate(x, ns, "B87_relaxed", ql)
        worst = max(worst, abs(r["hard"] - b["hard"]), abs(r["lc1"] * 1e3 - b["lc1_mm"]),
                    abs(r["sigma"] - b["sigma"]), abs(r["qn"] - b["qn"]),
                    abs(r["eta_t"] - b["eta_t"]),
                    abs(r["rho_root"][0] - b["rho_root_1"]),
                    abs(r["rho_root"][1] - b["rho_root_2"]))
    lines += [f"Повторная оценка top-20 на свежем `ns`: max |Δ| = {worst:.2e}", ""]
    (RESULTS / "REPORT_SCAN4D.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--q-n", type=float, default=0.995)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report(args.q_n)
    else:
        run(args.workers, args.q_n)


if __name__ == "__main__":
    main()
