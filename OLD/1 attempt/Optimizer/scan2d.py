"""Этап 2: пять двумерных сканов от лучших точек полного прогона.

    1. D_вт × w2        4. n × D_вт
    2. w1 × α1_2        5. α1_1 × α1_2
    3. κ1 × w1

Сетка 31×31 по исходным границам переменных, остальные — на базовой точке
(лучший hard из full10 для A87_relaxed и B87_relaxed). Ограничения и ядро
не меняются. Сохраняются все ограничения, а не только hard.

    python scan2d.py                 # оба сценария, все пары
    python scan2d.py --steps 21
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import optimize_extended as oe
import analyze_extended as an
import optimizer_core as oc

RESULTS = oe.RESULTS
PAIRS = [
    ("D_vt_w2", "D_вт", 1, "w2", 3),
    ("w1_a12", "w1", 2, "α1_2", 5),
    ("k1_w1", "κ1", 7, "w1", 2),
    ("n_D_vt", "n", 0, "D_вт", 1),
    ("a11_a12", "α1_1", 4, "α1_2", 5),
]
FIELDS = ["scenario", "pair", "x_name", "y_name", "x", "y", "status", "hard",
          "hard_no_strength", "ok_count", "lc1_mm", "rho_root_1", "rho_root_2",
          "a2_1", "a2_2", "a2_3", "sigma", "qn", "eta_t", "lD", "u", "N_kW",
          "set_len14", "set_len_rho", "feasible", "ok_no_strength",
          "violation_count", "binding", "violations", "x_full"]


def group(name: str) -> str:
    for g in ("α2_1", "α2_2", "α2_3"):
        if name.startswith(g):
            return g
    if name.startswith("l_c1"):
        return "l_c1"
    if name.startswith("ρ_корня_1"):
        return "ρ1"
    if name.startswith("ρ_корня_2"):
        return "ρ2"
    if name.startswith("σ > ") or name.startswith("q_n > "):
        return "strength"
    if name.startswith("l_t/D_t"):
        return "lD"
    if name.startswith("u >"):
        return "u"
    return "model"


def eval_row(x, ns, scenario, ql):
    r = oe.eval_candidate(x, ns, scenario, ql)
    a2 = (r["a2"] + [float("nan")] * 3)[:3]
    rr = (r["rho_root"] + [float("nan")] * 2)[:2]
    groups = {group(v["name"]) for v in r["violations"]}
    nonstrength = [v for v in r["violations"] if group(v["name"]) != "strength"]
    binding = (max(r["violations"], key=lambda v: v["amount"])["name"]
               if r["violations"] else "—")
    return {
        "status": r["status"], "hard": r["hard"],
        "hard_no_strength": max((v["amount"] for v in nonstrength), default=0.0),
        "ok_count": 10 - len(groups),
        "lc1_mm": r["lc1"] * 1e3, "rho_root_1": rr[0], "rho_root_2": rr[1],
        "a2_1": a2[0], "a2_2": a2[1], "a2_3": a2[2],
        "sigma": r["sigma"], "qn": r["qn"], "eta_t": r["eta_t"],
        "lD": r["lD"], "u": r["u"], "N_kW": r["N"] / 1e3,
        "set_len14": bool(r["lc1"] >= 0.014),
        "set_len_rho": bool(r["lc1"] >= 0.014 and rr[0] >= 0.05 and rr[1] >= 0.05),
        "feasible": bool(r["feasible"]),
        "ok_no_strength": bool(nonstrength == []),
        "violation_count": r["violation_count"],
        "binding": binding,
        "violations": "; ".join(f"{v['name']}:{v['amount']:.3f}" for v in r["violations"]),
        "x_full": ";".join(f"{v:.17g}" for v in x),
    }


def fnum(v, nd=4):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def run_pair(m, scenario, x0, pair, steps, ql):
    pid, xn, ix, yn, iy = pair
    xs = np.linspace(*oe.BOUNDS[ix], steps)
    ys = np.linspace(*oe.BOUNDS[iy], steps)
    ns, _ = oe.build_ns(m, ql, ql ** 2 * oe.SIGMA_LIM)
    rows = []
    for yv in ys:
        for xv in xs:
            x = x0.copy()
            x[ix], x[iy] = xv, yv
            r = eval_row(x, ns, scenario, ql)
            r.update(scenario=scenario, pair=pid, x_name=xn, y_name=yn,
                     x=float(xv), y=float(yv))
            rows.append(r)
    return rows, xs, ys


def recheck(row, scenario, ql, m):
    x = np.array([float(v) for v in row["x_full"].split(";")])
    ns, _ = oe.build_ns(m, ql, ql ** 2 * oe.SIGMA_LIM)
    r = eval_row(x, ns, scenario, ql)
    keys = ("hard", "lc1_mm", "sigma", "qn", "eta_t", "rho_root_1", "rho_root_2")
    return max(abs(r[k] - row[k]) for k in keys), r


def plot(scenario, pid, xn, yn, xs, ys, rows, path: Path):
    z = {k: np.array([r[k] for r in rows], dtype=float).reshape(len(ys), len(xs))
         for k in ("hard", "lc1_mm", "rho_root_1", "rho_root_2", "sigma")}
    z["rho_min"] = np.minimum(z["rho_root_1"], z["rho_root_2"])
    X, Y = np.meshgrid(xs, ys)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    panels = [("hard", "hard", None), ("lc1_mm", "l_c1, mm", 14.0),
              ("rho_min", "min(rho_root)", 0.05), ("sigma", "sigma, MPa", 300.0)]
    for ax, (key, label, level) in zip(axs.ravel(), panels):
        m = ax.pcolormesh(X, Y, z[key], shading="auto", cmap="viridis")
        fig.colorbar(m, ax=ax, shrink=0.9)
        if level is not None:
            ax.contour(X, Y, z[key], levels=[level], colors="red", linewidths=1.5)
        ax.set_xlabel(xn)
        ax.set_ylabel(yn)
        ax.set_title(label)
    fig.suptitle(f"{scenario}: {xn} × {yn}")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def pair_metrics(rows):
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        return None
    best = min(ok, key=lambda r: r["hard"])
    longest = max(ok, key=lambda r: r["lc1_mm"])
    lr = [r for r in ok if r["set_len_rho"]]
    longest_rho = max(lr, key=lambda r: r["lc1_mm"]) if lr else None
    ns_ = [r for r in ok if r["ok_no_strength"]]
    longest_nostr = max(ns_, key=lambda r: r["lc1_mm"]) if ns_ else None
    min_sig = min(ns_, key=lambda r: r["sigma"]) if ns_ else None
    return dict(
        n_ok=len(ok), n_err=len(rows) - len(ok),
        best=best, longest=longest, longest_rho=longest_rho,
        longest_nostr=longest_nostr, min_sig=min_sig,
        n_len14=sum(r["set_len14"] for r in ok),
        n_len_rho=len(lr),
        n_feasible=sum(r["feasible"] for r in ok),
        n_nostr=len(ns_),
        hard_min=best["hard"], hard_max=max(r["hard"] for r in ok),
    )


def point_line(r, tag):
    return (f"- **{tag}**: {r['x_name']}={r['x']:.4g}, {r['y_name']}={r['y']:.4g}; "
            f"hard={r['hard']:.3f}, l_c1={r['lc1_mm']:.2f} мм, "
            f"ρ={fnum(r['rho_root_1'])}/{fnum(r['rho_root_2'])}, "
            f"α2={r['a2_1']:.1f}/{r['a2_2']:.1f}/{r['a2_3']:.1f}, "
            f"σ={r['sigma']:.0f}, q_n={r['qn']:.4f}, η_t*={r['eta_t']:.4f}; "
            f"связывающее: {r['binding']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["A87_relaxed", "B87_relaxed"])
    ap.add_argument("--steps", type=int, default=31)
    ap.add_argument("--q-n", type=float, default=0.995)
    args = ap.parse_args()
    m = oc.load_model()
    all_rows, report, summary = [], [], []
    for scenario in args.scenarios:
        base_rows = an.load_csv(RESULTS / f"extended_{scenario}_q0.995_full10.csv")
        b = min(base_rows, key=lambda r: r["hard"])
        x0 = np.asarray(b["x"], dtype=float)
        report += [f"# Этап 2: двумерные сканы — {scenario}", "",
                   f"База: hard={b['hard']:.4f}, l_c1={b['lc1']*1e3:.2f} мм, "
                   f"ρ={b['rho_root'][0]:.4f}/{b['rho_root'][1]:.4f}, "
                   f"σ={b['sigma']:.0f}, q_n={b['qn']:.4f}", ""]
        for pair in PAIRS:
            pid, xn, ix, yn, iy = pair
            rows, xs, ys = run_pair(m, scenario, x0, pair, args.steps, args.q_n)
            for r in rows:
                all_rows.append(r)
            with (RESULTS / f"scan2d_{scenario}_{pid}.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in FIELDS})
            plot(scenario, pid, xn, yn, xs, ys, rows,
                 RESULTS / f"scan2d_{scenario}_{pid}.png")
            met = pair_metrics(rows)
            if met is None:
                report += [f"## {xn} × {yn}", "все точки численно некорректны", ""]
                continue
            report += [f"## {xn} × {yn}", "",
                       f"Точек: {met['n_ok']} ok, {met['n_err']} numerical/model_error. "
                       f"hard {met['hard_min']:.3f}…{met['hard_max']:.3f}.", ""]
            report.append(point_line(met["best"], "мин. hard"))
            report.append(point_line(met["longest"], "макс. l_c1"))
            if met["longest_rho"]:
                report.append(point_line(met["longest_rho"], "макс. l_c1 при ρ1,ρ2 ≥ 0.05"))
            if met["longest_nostr"]:
                report.append(point_line(met["longest_nostr"], "макс. l_c1 при всех, кроме прочности"))
            if met["min_sig"]:
                report.append(point_line(met["min_sig"], "мин. σ при всех, кроме прочности"))
            report += ["",
                       f"Множества: l_c1 ≥ 14 мм — {met['n_len14']}; "
                       f"l_c1 ≥ 14 и ρ1,ρ2 ≥ 0.05 — {met['n_len_rho']}; "
                       f"все ограничения — {met['n_feasible']}; "
                       f"все кроме прочности — {met['n_nostr']}.", ""]
            # повторная проверка кандидатов
            cands = [met["best"], met["longest_rho"], met["min_sig"]]
            worst = 0.0
            for c in cands:
                if c is None:
                    continue
                d, _ = recheck(c, scenario, args.q_n, m)
                worst = max(worst, d)
            report += [f"Повторная оценка кандидатов исходной функцией: max |Δ| = {worst:.2e}", ""]
            summary.append(dict(
                scenario=scenario, pair=f"{xn}×{yn}", steps=args.steps,
                ok=met["n_ok"], err=met["n_err"],
                hard_min=met["hard_min"],
                l_max=met["longest"]["lc1_mm"],
                l_max_rho=(met["longest_rho"]["lc1_mm"] if met["longest_rho"] else float("nan")),
                l_max_nostr=(met["longest_nostr"]["lc1_mm"] if met["longest_nostr"] else float("nan")),
                sigma_min_nostr=(met["min_sig"]["sigma"] if met["min_sig"] else float("nan")),
                n_len14=met["n_len14"], n_len_rho=met["n_len_rho"],
                n_feasible=met["n_feasible"], n_nostr=met["n_nostr"],
            ))
            print(f"{scenario} {xn}×{yn}: hard_min={met['hard_min']:.3f}, "
                  f"l_max={met['longest']['lc1_mm']:.2f}, len14={met['n_len14']}, "
                  f"len+rho={met['n_len_rho']}, feasible={met['n_feasible']}", flush=True)
    with (RESULTS / "scan2d.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    if summary:
        with (RESULTS / "scan2d_summary.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
    report += ["# Сводка", "",
               "| сценарий | пара | hard min | max l_c1 | max l_c1 (ρ ok) | "
               "max l_c1 (без прочности) | min σ (без прочности) | l≥14 | l≥14+ρ | всё |",
               "|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary:
        report.append(
            f"| {r['scenario']} | {r['pair']} | {r['hard_min']:.3f} | {fnum(r['l_max'], 2)} | "
            f"{fnum(r['l_max_rho'], 2)} | {fnum(r['l_max_nostr'], 2)} | "
            f"{fnum(r['sigma_min_nostr'], 0)} | {r['n_len14']} | {r['n_len_rho']} | {r['n_feasible']} |")
    report.append("")
    (RESULTS / "REPORT_SCAN2D.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"сохранено: {RESULTS / 'scan2d.csv'}, REPORT_SCAN2D.md", flush=True)


if __name__ == "__main__":
    main()
