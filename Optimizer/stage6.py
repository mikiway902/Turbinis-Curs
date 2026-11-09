"""Этап 6: цель — рабочая лопатка 1-й ступени 16…20 мм и рост по ступеням.

6А: контроль κ1 на семействе этапа 5 (все ограничения, кроме длины СА).
Плюс оценка новой цели по роторным лопаткам l_л (РК), а не l_c (СА).

    python stage6.py
"""
from __future__ import annotations

import csv

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import optimize_extended as oe
import scan5d as s5
import optimizer_core as oc

RESULTS = oe.RESULTS
K1_BASE = 1.0124
K1_NEW = 1.0
FIELDS = ["n", "D_vt", "w2", "w1", "k2", "k1", "rk1", "rk2", "rk3", "sa1", "sa2", "sa3",
          "a2_1", "a2_2", "a2_3", "rho1", "rho2", "sigma", "qn", "eta_t", "N_kW",
          "hard", "monotonic", "A16", "B", "C_B", "C_strict", "D", "E"]


def measure(x, ns, ql):
    r = oe.eval_candidate(x, ns, "B87_relaxed", ql)
    _, st = ns["распределить_π_const_hub"](x[0], x[1], ns["π_т"] ** (1 / ns["z"]))
    rk = [s_["l_л"] * 1e3 for s_ in st]
    sa = [s_["l_c"] * 1e3 for s_ in st]
    a2 = r["a2"]
    mono = rk[1] > rk[0] and rk[2] > rk[1]
    a2_b = all(lo <= v <= hi for v, (lo, hi) in zip(a2, oe.SCENARIOS["B87_relaxed"]["a2"]))
    a2_s = all(lo <= v <= hi for v, (lo, hi) in zip(a2, [(70, 90)] * 2 + [(80, 90)]))
    rho_ok = r["rho_root"][0] >= 0.05 and r["rho_root"][1] >= 0.05
    sig_ok = r["sigma"] <= 300.0 and r["qn"] <= ql
    eta_ok = r["eta_t"] >= 0.90 - 1e-9
    A16 = rk[0] >= 16.0
    return dict(rk1=rk[0], rk2=rk[1], rk3=rk[2], sa1=sa[0], sa2=sa[1], sa3=sa[2],
                a2_1=a2[0], a2_2=a2[1], a2_3=a2[2],
                rho1=r["rho_root"][0], rho2=r["rho_root"][1],
                sigma=r["sigma"], qn=r["qn"], eta_t=r["eta_t"], N_kW=r["N"] / 1e3,
                hard=r["hard"], monotonic=mono, A16=A16,
                B=bool(A16 and mono),
                C_B=bool(A16 and mono and a2_b and rho_ok),
                C_strict=bool(A16 and mono and a2_s and rho_ok),
                D=bool(A16 and mono and a2_b and rho_ok and sig_ok),
                E=bool(A16 and mono and a2_b and rho_ok and sig_ok and eta_ok))


def main() -> None:
    m = oc.load_model()
    ns, _ = oe.build_ns(m, 0.995, 0.995 ** 2 * oe.SIGMA_LIM)
    fam = [r for r in s5.load() if r["ok_except_l"]]
    rows = []
    for b in fam:
        x = s5.base_point()
        x[0], x[1], x[3], x[2], x[8] = b["n"], b["D_vt"], b["w2"], b["w1"], b["k2"]
        for k1 in (K1_BASE, K1_NEW):
            x[7] = k1
            r = measure(x, ns, 0.995)
            r.update(n=b["n"], D_vt=b["D_vt"], w2=b["w2"], w1=b["w1"], k2=b["k2"], k1=k1)
            rows.append(r)
    with (RESULTS / "stage6_results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    new = [r for r in rows if r["k1"] == K1_NEW]
    base = [r for r in rows if r["k1"] == K1_BASE]
    cats = {k: sum(r[k] for r in new) for k in ("A16", "B", "C_B", "C_strict", "D", "E")}
    summary = [dict(category=k, k1_new=v, k1_base=sum(r[k] for r in base)) for k, v in cats.items()]
    with (RESULTS / "stage6_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["category", "k1_new", "k1_base"])
        w.writeheader()
        w.writerows(summary)
    best = sorted(new, key=lambda r: (-r["rk1"], -r["eta_t"]))[:10]
    with (RESULTS / "stage6_best.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(best)
    # график: длины по ступеням у лучших точек
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    idx = np.arange(len(best))
    for j, key in enumerate(("rk1", "rk2", "rk3")):
        ax.bar(idx + (j - 1) * 0.26, [r[key] for r in best], width=0.26,
               label=f"l_рк{j+1}")
    ax.axhline(16, color="red", ls="--", lw=1)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{i+1}" for i in idx])
    ax.set_xlabel("лучшие точки (по l_рк1)")
    ax.set_ylabel("высота рабочей лопатки, мм")
    ax.legend()
    ax.set_title("Этап 6: рабочие лопатки по ступеням (κ1=1.0)")
    fig.savefig(RESULTS / "stage6_rk_by_stage.png", dpi=110)
    plt.close(fig)
    # график: rk1 от w1 и n
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axs[0].scatter([r["w1"] for r in new], [r["rk1"] for r in new], s=10)
    axs[0].axhline(16, color="red", ls="--", lw=1)
    axs[0].set_xlabel("w1")
    axs[0].set_ylabel("l_рк1, мм")
    axs[1].scatter([r["n"] / 1e3 for r in new], [r["rk1"] for r in new], s=10)
    axs[1].axhline(16, color="red", ls="--", lw=1)
    axs[1].set_xlabel("n, тыс. об/мин")
    axs[1].set_ylabel("l_рк1, мм")
    fig.suptitle("Этап 6: l_рк1 в семействе 308 точек")
    fig.savefig(RESULTS / "stage6_rk1_vs_params.png", dpi=110)
    plt.close(fig)
    print("категории (κ1=1.0 / 1.0124):", {k: (cats[k], sum(r[k] for r in base)) for k in cats})
    print("max rk1:", round(max(r["rk1"] for r in new), 2),
          "; max rk1 при eta>=0.9:", round(max((r["rk1"] for r in new if r["eta_t"] >= 0.9),
                                              default=float("nan")), 2))


if __name__ == "__main__":
    main()
