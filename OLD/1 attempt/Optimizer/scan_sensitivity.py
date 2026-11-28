"""Этап 1: одномерная чувствительность от лучшей точки полного прогона.

Для каждого сценария берётся лучший по hard кандидат из
`results/extended_<сценарий>_q0.995_full10.csv`; каждая переменная
сканируется по своим границам при фиксированных остальных.
Сохраняются все ограничения, не только hard.

    python scan_sensitivity.py                 # B87_relaxed и A87_relaxed
    python scan_sensitivity.py --steps 41
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import optimize_extended as oe
import analyze_extended as an
import optimizer_core as oc

RESULTS = oe.RESULTS
VARS = [("n", 0), ("D_вт", 1), ("w1", 2), ("w2", 3), ("α1_1", 4), ("α1_2", 5),
        ("α1_3", 6), ("κ1", 7), ("κ2", 8), ("κ3", 9)]
FIELDS = ["scenario", "variable", "value", "status", "hard", "lc1_mm", "a2_1", "a2_2",
          "a2_3", "rho_root_1", "rho_root_2", "rho_1", "rho_2", "rho_3", "sigma",
          "qn", "lD", "eta_t", "u", "N_kW", "violation_count", "binding", "violations"]


def best_point(scenario: str) -> tuple[np.ndarray, dict]:
    path = RESULTS / f"extended_{scenario}_q0.995_full10.csv"
    rows = an.load_csv(path)
    b = min(rows, key=lambda r: r["hard"])
    return np.asarray(b["x"], dtype=float), b


def binding(r: dict) -> str:
    if r["status"] != "ok":
        return r["status"]
    if not r["violations"]:
        return "— (допустим)"
    top = max(r["violations"], key=lambda v: v["amount"])
    return f"{top['name']} ({top['amount']:.3f})"


def evaluate(x, ns, scenario, ql):
    r = oe.eval_candidate(x, ns, scenario, ql)
    a2 = (r["a2"] + [float("nan")] * 3)[:3]
    rr = (r["rho_root"] + [float("nan")] * 2)[:2]
    rho = (r["rho"] + [float("nan")] * 3)[:3]
    return {
        "scenario": scenario, "status": r["status"], "hard": r["hard"],
        "lc1_mm": r["lc1"] * 1e3, "a2_1": a2[0], "a2_2": a2[1], "a2_3": a2[2],
        "rho_root_1": rr[0], "rho_root_2": rr[1],
        "rho_1": rho[0], "rho_2": rho[1], "rho_3": rho[2],
        "sigma": r["sigma"], "qn": r["qn"], "lD": r["lD"],
        "eta_t": r["eta_t"], "u": r["u"], "N_kW": r["N"] / 1e3,
        "violation_count": r["violation_count"],
        "binding": binding(r),
        "violations": "; ".join(f"{v['name']}:{v['amount']:.3f}" for v in r["violations"]),
    }


def fnum(v: float, nd: int = 3) -> str:
    return "—" if not np.isfinite(v) else f"{v:.{nd}f}"


def scenario_scan(scenario: str, steps: int, ql: float, m) -> tuple[list[dict], dict]:
    x0, base = best_point(scenario)
    ns, _ = oe.build_ns(m, ql, ql ** 2 * oe.SIGMA_LIM)
    r0 = evaluate(x0, ns, scenario, ql)
    assert abs(r0["hard"] - base["hard"]) < 1e-9, "базовая точка не воспроизвелась"
    rows = []
    for name, i in VARS:
        lo, hi = oe.BOUNDS[i]
        for v in np.linspace(lo, hi, steps):
            x = x0.copy()
            x[i] = v
            r = evaluate(x, ns, scenario, ql)
            r["variable"] = name
            r["value"] = float(v)
            rows.append(r)
    print(f"{scenario}: базовая точка hard={r0['hard']:.4f}, l_c1={r0['lc1_mm']:.2f} мм, "
          f"{len(rows)} оценок скана")
    return rows, r0


def report(scenario: str, rows: list[dict], base: dict) -> str:
    lines = [f"# Этап 1: одномерная чувствительность — {scenario}", "",
             f"База (лучший hard полного прогона): `hard={base['hard']:.4f}`, "
             f"`l_c1={base['lc1_mm']:.2f} мм`, η_t*={fnum(base['eta_t'], 4)}, "
             f"σ={fnum(base['sigma'], 0)} МПа, q_n={fnum(base['qn'], 4)}, "
             f"ρ_корня={fnum(base['rho_root_1'], 4)}/{fnum(base['rho_root_2'], 4)}, "
             f"α2={fnum(base['a2_1'], 1)}/{fnum(base['a2_2'], 1)}/{fnum(base['a2_3'], 1)}.",
             ""]
    bad = [r for r in rows if r["status"] != "ok"]
    if bad:
        lines.append(f"Численно некорректных точек: {len(bad)} "
                     f"({', '.join(sorted({r['variable'] for r in bad}))}) — исключены из выводов.")
        lines.append("")
    ok = [r for r in rows if r["status"] == "ok"]

    lines += ["| переменная | лучший hard (значение) | l_c1 там, мм | ρ_корня 1 / 2 | α2 | σ, МПа | q_n | η_t* | связывающее |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, _ in VARS:
        rs = [r for r in ok if r["variable"] == name]
        if not rs:
            lines.append(f"| {name} | нет данных | | | | | | | |")
            continue
        b = min(rs, key=lambda r: r["hard"])
        lines.append(
            f"| {name} | {b['hard']:.3f} ({b['value']:.4g}) | {fnum(b['lc1_mm'], 2)} | "
            f"{fnum(b['rho_root_1'], 4)} / {fnum(b['rho_root_2'], 4)} | "
            f"{fnum(b['a2_1'], 1)}/{fnum(b['a2_2'], 1)}/{fnum(b['a2_3'], 1)} | "
            f"{fnum(b['sigma'], 0)} | {fnum(b['qn'], 4)} | {fnum(b['eta_t'], 4)} | {b['binding']} |")
    lines += ["", "## Диапазоны по переменным (все точки скана)", "",
              "| переменная | hard | l_c1, мм | ρ_корня 1 | σ, МПа | η_t* | нарушений в скане |",
              "|---|---|---|---|---|---|---|"]
    for name, _ in VARS:
        rs = [r for r in ok if r["variable"] == name]
        if not rs:
            continue
        l = [r["lc1_mm"] for r in rs]
        rr = [r["rho_root_1"] for r in rs]
        sg = [r["sigma"] for r in rs]
        et = [r["eta_t"] for r in rs]
        lines.append(
            f"| {name} | {min(r['hard'] for r in rs):.3f}…{max(r['hard'] for r in rs):.3f} | "
            f"{min(l):.2f}…{max(l):.2f} | {fnum(min(rr), 4)}…{fnum(max(rr), 4)} | "
            f"{fnum(min(sg), 0)}…{fnum(max(sg), 0)} | {fnum(min(et), 4)}…{fnum(max(et), 4)} | "
            f"{min(r['violation_count'] for r in rs)}…{max(r['violation_count'] for r in rs)} |")
    # какие ограничения появляются при отклонении от базы
    lines += ["", "## Что меняется при отклонении от базы", ""]
    for name, i in VARS:
        rs = [r for r in ok if r["variable"] == name]
        if not rs:
            continue
        lo, hi = oe.BOUNDS[i]
        go_lo = min(rs, key=lambda r: abs(r["value"] - lo))
        go_hi = min(rs, key=lambda r: abs(r["value"] - hi))
        better = [r for r in rs if r["hard"] < base["hard"] - 1e-9]
        verdict = (f"hard ниже базы в {len(better)} точках, минимум {min(r['hard'] for r in better):.3f}"
                   if better else "hard нигде не ниже базы")
        lines.append(f"- **{name}** ({lo:g}…{hi:g}): {verdict}; "
                     f"на границах hard={go_lo['hard']:.3f}/{go_hi['hard']:.3f}, "
                     f"l_c1={go_lo['lc1_mm']:.2f}/{go_hi['lc1_mm']:.2f} мм, "
                     f"связывающее: {go_lo['binding']} ⟷ {go_hi['binding']}")
    top = sorted(ok, key=lambda r: r["hard"])[:5]
    lines += ["", "## Пять лучших точек скана", ""]
    for r in top:
        lines.append(f"- {r['variable']}={r['value']:.4g}: hard={r['hard']:.3f}, "
                     f"l_c1={r['lc1_mm']:.2f} мм, ρ1={fnum(r['rho_root_1'], 4)}, "
                     f"σ={fnum(r['sigma'], 0)}, q_n={fnum(r['qn'], 4)}, {r['binding']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["B87_relaxed", "A87_relaxed"])
    ap.add_argument("--steps", type=int, default=31)
    ap.add_argument("--q-n", type=float, default=0.995)
    args = ap.parse_args()
    m = oc.load_model()
    all_rows = []
    for sc in args.scenarios:
        rows, base = scenario_scan(sc, args.steps, args.q_n, m)
        all_rows += rows
        text = report(sc, rows, base)
        (RESULTS / f"SCAN1D_{sc}.md").write_text(text, encoding="utf-8")
        print(text)
    with (RESULTS / "scan1d.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"сохранено: {RESULTS / 'scan1d.csv'}")


if __name__ == "__main__":
    main()
