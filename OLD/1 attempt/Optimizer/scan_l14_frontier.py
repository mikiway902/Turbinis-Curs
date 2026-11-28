"""Дополнение к этапу 2: фронт при фиксированной длине l_c1 = 14 мм.

Для каждой пары (n, w2) диаметр втулки подбирается делением пополам так,
чтобы l_c1 = 14 мм; остальные переменные — на базовой точке. Показывает,
какой максимум α2 и какая цена по ρ/σ возможны ровно на границе длины.

    python scan_l14_frontier.py
"""
from __future__ import annotations

import csv

import numpy as np

import optimize_extended as oe
import analyze_extended as an
import optimizer_core as oc

RESULTS = oe.RESULTS
FIELDS = ["scenario", "n", "w2", "D_vt", "lc1_mm", "a2_1", "a2_2", "a2_3",
          "a2_ok", "rho_root_1", "rho_root_2", "rho_ok", "sigma", "qn", "eta_t",
          "lD", "hard", "binding"]


def base_x(scenario):
    rows = an.load_csv(RESULTS / f"extended_{scenario}_q0.995_full10.csv")
    return np.asarray(min(rows, key=lambda r: r["hard"])["x"], dtype=float)


def main() -> None:
    m = oc.load_model()
    out = []
    for scenario in ("A87_relaxed", "B87_relaxed"):
        lo, hi = oe.BOUNDS[1]
        x0 = base_x(scenario)
        ns, _ = oe.build_ns(m, 0.995, 0.995 ** 2 * oe.SIGMA_LIM)
        a2_rng = oe.SCENARIOS[scenario]["a2"]
        for n in np.linspace(60e3, 95e3, 15):
            for w2 in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2):
                x = x0.copy()
                x[0], x[3] = n, w2
                a, b = lo, hi
                x[1] = a
                if oe.eval_candidate(x, ns, scenario, 0.995)["lc1"] < 0.014:
                    continue
                for _ in range(14):
                    mid = 0.5 * (a + b)
                    x[1] = mid
                    if oe.eval_candidate(x, ns, scenario, 0.995)["lc1"] > 0.014:
                        a = mid
                    else:
                        b = mid
                x[1] = 0.5 * (a + b)
                r = oe.eval_candidate(x, ns, scenario, 0.995)
                if r["status"] != "ok":
                    continue
                rr = r["rho_root"]
                a2_ok = all(lo_ <= v <= hi_ for v, (lo_, hi_) in zip(r["a2"], a2_rng))
                row = dict(
                    scenario=scenario, n=float(n), w2=float(w2), D_vt=float(x[1]),
                    lc1_mm=r["lc1"] * 1e3, a2_1=r["a2"][0], a2_2=r["a2"][1],
                    a2_3=r["a2"][2], a2_ok=a2_ok,
                    rho_root_1=rr[0], rho_root_2=rr[1],
                    rho_ok=bool(rr[0] >= 0.05 and rr[1] >= 0.05),
                    sigma=r["sigma"], qn=r["qn"], eta_t=r["eta_t"], lD=r["lD"],
                    hard=r["hard"],
                    binding=(max(r["violations"], key=lambda v: v["amount"])["name"]
                             if r["violations"] else "—"),
                )
                out.append(row)
        print(f"{scenario}: {sum(1 for r in out if r['scenario'] == scenario)} строк", flush=True)
    with (RESULTS / "scan_l14_frontier.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    lines = ["# Фронт l_c1 = 14 мм (дополнение к этапу 2)", "",
             "Для каждой пары (n, w2) D_вт подобран так, чтобы l_c1 = 14.00 мм; "
             "остальные переменные — на базовой точке. Ни одно ограничение не менялось.", ""]
    for scenario in ("A87_relaxed", "B87_relaxed"):
        rs = [r for r in out if r["scenario"] == scenario]
        if not rs:
            continue
        best = min(rs, key=lambda r: r["hard"])
        a2max = max(rs, key=lambda r: r["a2_2"])
        ok_rho = [r for r in rs if r["rho_ok"]]
        a2max_rho = max(ok_rho, key=lambda r: r["a2_2"]) if ok_rho else None
        a2ok = [r for r in rs if r["a2_ok"]]
        both = [r for r in rs if r["a2_ok"] and r["rho_ok"]]
        lines += [f"## {scenario}", "",
                  f"- точек на фронте: {len(rs)}; min hard = {best['hard']:.3f} "
                  f"(n={best['n']/1e3:.0f} тыс., w2={best['w2']:.2f}, "
                  f"α2={best['a2_1']:.1f}/{best['a2_2']:.1f}/{best['a2_3']:.1f}, "
                  f"ρ={best['rho_root_1']:.4f}/{best['rho_root_2']:.4f}, σ={best['sigma']:.0f})",
                  f"- max α2_2 = {a2max['a2_2']:.1f}° при n={a2max['n']/1e3:.0f} тыс., "
                  f"w2={a2max['w2']:.2f}; там ρ={a2max['rho_root_1']:.4f}/{a2max['rho_root_2']:.4f}, "
                  f"σ={a2max['sigma']:.0f}, hard={a2max['hard']:.3f}"]
        if a2max_rho:
            lines.append(f"- max α2_2 при ρ1,ρ2 ≥ 0.05: {a2max_rho['a2_2']:.1f}° "
                         f"(n={a2max_rho['n']/1e3:.0f} тыс., w2={a2max_rho['w2']:.2f}, "
                         f"σ={a2max_rho['sigma']:.0f}, hard={a2max_rho['hard']:.3f})")
        lines += [f"- точки со всеми α2 в норме: {len(a2ok)}; "
                  f"с α2 и ρ в норме одновременно: {len(both)}; "
                  f"допустимых (hard=0): {sum(r['hard'] <= 1e-9 for r in rs)}", ""]
    (RESULTS / "SCAN_L14_FRONTIER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
