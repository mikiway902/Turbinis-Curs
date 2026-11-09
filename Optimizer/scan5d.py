"""Этап 5: сетка (n, D_вт, w2, w1, κ2) 9⁵ = 59 049 точек.

Остальные переменные — из лучшей точки B87_relaxed, кроме:
κ3 = 1.30, α1_3 = 23°, α1_2 = 20°.

Категории допустимости:
  A  l_c1 ≥ 14
  B  A + ρ1,ρ2 ≥ 0.05
  C  B + все ограничения α2
  D  C + σ ≤ 300 и q_n ≤ 0.995
  E  D + η_t* ≥ 0.90

    python scan5d.py --workers 8
    python scan5d.py --report
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import optimize_extended as oe
import analyze_extended as an
import optimizer_core as oc
import scan3d

RESULTS = oe.RESULTS
STEPS = 9
OVERRIDES = dict(k3=1.30, a13=23.0, a12=20.0)
# границы этапа 5 (уже исходных; сужение — осознанный выбор, не ослабление)
GRID = {
    "n": (60e3, 90e3),
    "D": (0.045, 0.062),
    "w2": (0.50, 1.00),
    "w1": (0.70, 1.20),
    "k2": (1.00, 1.35),
}
IDX = scan3d.IDX
FIELDS = ["n", "D_vt", "w2", "w1", "k2", "status", "hard", "hard_no_strength",
          "ok_count", "lc1_mm", "rho_root_1", "rho_root_2", "a2_1", "a2_2", "a2_3",
          "sigma", "qn", "eta_t", "lD", "u", "N_kW", "l_ok", "rho_ok",
          "a2_ok_B", "a2_ok_A", "sigma_ok", "qn_ok", "eta_ok",
          "feasible", "full", "ok_except_l", "violation_count", "binding",
          "violations", "idx"]


def base_point() -> np.ndarray:
    x = scan3d.base_point()
    for k, v in OVERRIDES.items():
        x[IDX[k]] = v
    return x


def eval5(x, ns, ql: float) -> dict:
    r = oe.eval_candidate(x, ns, "B87_relaxed", ql)
    a2 = (r["a2"] + [float("nan")] * 3)[:3]
    rr = (r["rho_root"] + [float("nan")] * 2)[:2]
    groups = {scan3d.group(v["name"]) for v in r["violations"]}
    nonstrength = [v for v in r["violations"] if scan3d.group(v["name"]) != "strength"]
    nonlen = [v for v in r["violations"] if scan3d.group(v["name"]) != "l_c1"]
    a2_ok_b = all(lo <= v <= hi for v, (lo, hi) in zip(a2, oe.SCENARIOS["B87_relaxed"]["a2"]))
    a2_ok_a = all(lo <= v <= hi for v, (lo, hi) in zip(a2, [(70, 90), (70, 90), (80, 90)]))
    eta_ok = bool(r["eta_t"] >= 0.90 - 1e-9)
    feasible = bool(r["feasible"])
    return {
        "n": float(x[0]), "D_vt": float(x[1]), "w2": float(x[3]), "w1": float(x[2]),
        "k2": float(x[8]),
        "status": r["status"], "hard": r["hard"],
        "hard_no_strength": max((v["amount"] for v in nonstrength), default=0.0),
        "ok_count": 10 - len(groups),
        "lc1_mm": r["lc1"] * 1e3, "rho_root_1": rr[0], "rho_root_2": rr[1],
        "a2_1": a2[0], "a2_2": a2[1], "a2_3": a2[2],
        "sigma": r["sigma"], "qn": r["qn"], "eta_t": r["eta_t"],
        "lD": r["lD"], "u": r["u"], "N_kW": r["N"] / 1e3,
        "l_ok": bool(r["lc1"] >= 0.014),
        "rho_ok": bool(rr[0] >= 0.05 and rr[1] >= 0.05),
        "a2_ok_B": a2_ok_b, "a2_ok_A": a2_ok_a,
        "sigma_ok": bool(r["sigma"] <= 300.0),
        "qn_ok": bool(r["qn"] <= ql),
        "eta_ok": eta_ok,
        "feasible": feasible,
        "full": bool(feasible and eta_ok),
        "ok_except_l": bool(nonlen == []),
        "violation_count": r["violation_count"],
        "binding": (max(r["violations"], key=lambda v: v["amount"])["name"]
                    if r["violations"] else "—"),
        "violations": "; ".join(f"{v['name']}:{v['amount']:.3f}" for v in r["violations"]),
    }


_W: dict = {}


def _init(ql: float) -> None:
    m = oc.load_model()
    _W["ql"] = ql
    _W["x0"] = base_point()
    _W["ns"], _ = oe.build_ns(m, ql, ql ** 2 * oe.SIGMA_LIM)
    _W["axes"] = [np.linspace(*GRID[k], STEPS) for k in ("n", "D", "w2", "w1", "k2")]


def _work(rng) -> list[dict]:
    start, end = rng
    ns, x0 = _W["ns"], _W["x0"]
    nv, dv, w2v, w1v, k2v = _W["axes"]
    shape = (STEPS,) * 5
    out = []
    for idx in range(start, end):
        i, j, k, l, m = np.unravel_index(idx, shape)
        x = x0.copy()
        x[0], x[1], x[3], x[2], x[8] = nv[i], dv[j], w2v[k], w1v[l], k2v[m]
        r = eval5(x, ns, _W["ql"])
        r["idx"] = int(idx)
        out.append(r)
    return out


def run(workers: int, ql: float) -> None:
    total = STEPS ** 5
    bounds = np.linspace(0, total, workers + 1).astype(int)
    chunks = [(int(bounds[i]), int(bounds[i + 1])) for i in range(workers)]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(ql,)) as ex:
        parts = list(ex.map(_work, chunks))
    rows = [r for part in parts for r in part]
    rows.sort(key=lambda r: r["idx"])
    with (RESULTS / "scan5d.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({f: r.get(f, "") for f in FIELDS})
    print(f"scan5d: {len(rows)} точек -> scan5d.csv", flush=True)


def load() -> list[dict]:
    rows = []
    with (RESULTS / "scan5d.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k in ("n", "D_vt", "w2", "w1", "k2", "hard", "hard_no_strength",
                      "lc1_mm", "rho_root_1", "rho_root_2", "a2_1", "a2_2", "a2_3",
                      "sigma", "qn", "eta_t", "lD", "u", "N_kW"):
                r[k] = float(r[k])
            for k in ("l_ok", "rho_ok", "a2_ok_B", "a2_ok_A", "sigma_ok", "qn_ok",
                      "eta_ok", "feasible", "full", "ok_except_l"):
                r[k] = r[k] == "True"
            r["ok_count"] = int(float(r["ok_count"]))
            r["violation_count"] = int(float(r["violation_count"]))
            rows.append(r)
    return rows


def fnum(v, nd=4):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def point_line(r, tag):
    return (f"- **{tag}**: n={r['n']/1e3:.2f} тыс., D_вт={r['D_vt']*1e3:.2f} мм, "
            f"w2={r['w2']:.4f}, w1={r['w1']:.4f}, κ2={r['k2']:.4f}; "
            f"hard={r['hard']:.3f}, l_c1={r['lc1_mm']:.2f} мм, "
            f"ρ={fnum(r['rho_root_1'])}/{fnum(r['rho_root_2'])}, "
            f"α2={r['a2_1']:.1f}/{r['a2_2']:.1f}/{r['a2_3']:.1f}, "
            f"σ={r['sigma']:.0f}, q_n={r['qn']:.4f}, η_t*={r['eta_t']:.4f}; "
            f"связывающее: {r['binding']}")


def report(ql: float) -> None:
    rows = load()
    ok = [r for r in rows if r["status"] == "ok"]
    A = [r for r in ok if r["l_ok"]]
    B = [r for r in A if r["rho_ok"]]
    C = [r for r in B if r["a2_ok_B"]]
    D = [r for r in C if r["sigma_ok"] and r["qn_ok"]]
    E = [r for r in D if r["eta_ok"]]
    lines = ["# Этап 5: сетка (n, D_вт, w2, w1, κ2)", "",
             f"{len(rows)} точек (9⁵), ошибок {len(rows) - len(ok)}. "
             f"κ3={OVERRIDES['k3']}, α1_3={OVERRIDES['a13']}°, "
             f"α1_2={OVERRIDES['a12']}°, остальное — лучшая точка B87_relaxed.", "",
             "Границы: n 60…90 тыс., D_вт 45…62 мм, w2 0.5…1.0, w1 0.7…1.2, "
             "κ2 1.0…1.35.", "",
             "| категория | точек |", "|---|---|",
             f"| A: l_c1 ≥ 14 | {len(A)} |",
             f"| B: A + ρ1,ρ2 ≥ 0.05 | {len(B)} |",
             f"| C: B + все α2 (B) | {len(C)} |",
             f"| C при строгих α2 70/70/80 | {sum(r['a2_ok_A'] for r in B)} |",
             f"| D: C + σ ≤ 300 и q_n ≤ 0.995 | {len(D)} |",
             f"| E: D + η_t* ≥ 0.90 | {len(E)} |",
             f"| E = все ограничения (hard=0) + η | {sum(r['full'] for r in ok)} |",
             f"| всё, кроме длины (ok_except_l) | {sum(r['ok_except_l'] for r in ok)} |",
             ""]
    # Чувствительность к κ2
    lines += ["## Чувствительность к κ2", "",
              "| κ2 | точек | A | B | C | D | E | max α2_2 при B | min hard | max l (ok_except_l) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    k2vals = sorted({round(r["k2"], 6) for r in ok})
    for k2 in k2vals:
        sub = [r for r in ok if round(r["k2"], 6) == k2]
        a = [r for r in sub if r["l_ok"]]
        b = [r for r in a if r["rho_ok"]]
        c = [r for r in b if r["a2_ok_B"]]
        d = [r for r in c if r["sigma_ok"] and r["qn_ok"]]
        e = [r for r in d if r["eta_ok"]]
        lines.append(
            f"| {k2:.4f} | {len(sub)} | {len(a)} | {len(b)} | {len(c)} | {len(d)} | {len(e)} | "
            f"{fnum(max((r['a2_2'] for r in b), default=float('nan')), 1)} | "
            f"{min(r['hard'] for r in sub):.3f} | "
            f"{fnum(max((r['lc1_mm'] for r in sub if r['ok_except_l']), default=float('nan')), 2)} |")
    lines.append("")
    # Запрошенные величины
    lines += ["## Ключевые величины", ""]
    ab = [r for r in A if r["rho_ok"]]
    if ab:
        b = max(ab, key=lambda r: r["a2_2"])
        lines.append(point_line(b, "max α2_2 при l ≥ 14 и ρ1,ρ2 ≥ 0.05"))
    else:
        lines.append("- **max α2_2 при l ≥ 14 и ρ ok**: точек нет")
    oel = [r for r in ok if r["ok_except_l"]]
    if oel:
        b = max(oel, key=lambda r: r["lc1_mm"])
        lines.append(point_line(b, "max l при всех остальных ограничениях"))
        lines.append(f"  - κ2 в этих точках: {min(r['k2'] for r in oel):.3f}…{max(r['k2'] for r in oel):.3f}")
    else:
        lines.append("- **max l при всех остальных ограничениях**: точек нет")
    if A:
        b = max(A, key=lambda r: r["eta_t"])
        lines.append(point_line(b, "max η_t* при l ≥ 14"))
    # top-10 по hard
    lines += ["", "## Ближайшие к допустимости", "",
              "| # | hard | n | D_вт | w2 | w1 | κ2 | l_c1 | ρ1/ρ2 | α2 | σ | η_t* | связывающее |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(sorted(ok, key=lambda r: (r["hard"], -r["lc1_mm"]))[:10], 1):
        lines.append(
            f"| {i} | {r['hard']:.3f} | {r['n']/1e3:.1f} | {r['D_vt']*1e3:.1f} | "
            f"{r['w2']:.3f} | {r['w1']:.3f} | {r['k2']:.3f} | {r['lc1_mm']:.2f} | "
            f"{r['rho_root_1']:.4f}/{r['rho_root_2']:.4f} | "
            f"{r['a2_1']:.1f}/{r['a2_2']:.1f}/{r['a2_3']:.1f} | "
            f"{r['sigma']:.0f} | {r['eta_t']:.4f} | {r['binding']} |")
    lines.append("")
    # повторная проверка
    m = oc.load_model()
    ns, _ = oe.build_ns(m, ql, ql ** 2 * oe.SIGMA_LIM)
    x0 = base_point()
    worst = 0.0
    for b in sorted(ok, key=lambda r: r["hard"])[:20]:
        x = x0.copy()
        x[0], x[1], x[3], x[2], x[8] = b["n"], b["D_vt"], b["w2"], b["w1"], b["k2"]
        r = oe.eval_candidate(x, ns, "B87_relaxed", ql)
        worst = max(worst, abs(r["hard"] - b["hard"]), abs(r["lc1"] * 1e3 - b["lc1_mm"]),
                    abs(r["sigma"] - b["sigma"]), abs(r["qn"] - b["qn"]),
                    abs(r["eta_t"] - b["eta_t"]),
                    abs(r["rho_root"][0] - b["rho_root_1"]),
                    abs(r["rho_root"][1] - b["rho_root_2"]),
                    abs(r["a2"][1] - b["a2_2"]))
    lines += [f"Повторная оценка top-20 на свежем `ns`: max |Δ| = {worst:.2e}", ""]
    (RESULTS / "REPORT_SCAN5D.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
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
