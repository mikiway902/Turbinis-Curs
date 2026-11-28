"""6Б-5 (шаг 1): интегральная проверка радиальной модели.

Для каждой ступени по её закону профилирования (ns["закон_ст"]):
  - распределения c1u(r), c1a(r), p1(r), T1(r), ρ_г(r) по всему каналу;
  - интегральный расход G = ∫ρ_г·c1a·2πr dr и невязка к G_c;
  - радиальное равновесие dp/dr = ρ_г·c1u²/r;
  - постоянство работы L_u(r) = c1u·u1 + c2u·u2;
  - сверка корневой ρ с эталонной ns["ρ_корня"].

    python radial_integral.py
"""
from __future__ import annotations

import argparse
import csv
import math

import numpy as np

import optimize_extended as oe
import scan5d as s5
import radial_check as rc
import optimizer_core as oc

RESULTS = oe.RESULTS


def span(st, n, law, R, φ, npts=201):
    k, cp, T0, p0, H0, p2 = st["k"], st["cp"], st["T0"], st["p0"], st["H0"], st["p2"]
    r_c, r_l = st["D_c"] / 2, st["D_л"] / 2
    r1 = np.linspace(r_c - 0.5 * st["l_c"], r_c + 0.5 * st["l_c"], npts)
    r2 = np.linspace(r_l - 0.5 * st["l_л"], r_l + 0.5 * st["l_л"], npts)
    u1 = math.pi * n * r1 / 30
    u2 = math.pi * n * r2 / 30
    if law == "Γ":
        c1u = st["c1u"] * r_c / r1
        c1a = np.full_like(r1, st["c1a"])
        M1u = st["c1u"] / math.sqrt(k * R * st["T1_ад"])
        p1 = st["p1"] * (1 + (k - 1) / 2 * M1u ** 2 * (1 - (r_c / r1) ** 2)) ** (k / (k - 1))
        c2u = st["c2u"] * r_l / r2
        c2a = np.full_like(r2, st["c2a"])
    else:
        q = (r_c / r1) ** (math.cos(math.radians(st["α1"])) ** 2)
        c1u, c1a = st["c1u"] * q, st["c1a"] * q
        c1 = np.hypot(c1u, c1a)
        p1 = p0 * (1 - (c1 / φ) ** 2 / (2 * cp * T0)) ** (k / (k - 1))
        c2u = (st["L_u"] - c1u * u1) / u2
        c2a = np.full_like(r2, st["c2a"])
    c1 = np.hypot(c1u, c1a)
    T1 = T0 - c1 ** 2 / (2 * cp)
    rho = p1 / (R * T1)
    G = float(np.trapezoid(rho * c1a * 2 * math.pi * r1, r1))
    # радиальное равновесие: (1/ρ)dp/dr = c_u²/r
    dp = np.gradient(p1, r1)
    eq = dp - rho * c1u ** 2 / r1
    eq_rel = float(np.max(np.abs(eq) / np.maximum(np.abs(dp), 1e-9)))
    Lu = c1u * u1 + c2u * u2
    Lu_rel = float(np.max(np.abs(Lu / st["L_u"] - 1.0)))
    return dict(G=G, eq_rel=eq_rel, Lu_rel=Lu_rel,
                rho_hub=float(rho[0]), rho_tip=float(rho[-1]),
                c1a_hub=float(c1a[0]), c1a_tip=float(c1a[-1]),
                a1_hub=float(np.degrees(np.arctan2(c1a[0], c1u[0]))),
                a1_tip=float(np.degrees(np.arctan2(c1a[-1], c1u[-1]))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=float, default=90000)
    ap.add_argument("--D", type=float, default=0.062)
    ap.add_argument("--w2", type=float, default=0.8125)
    ap.add_argument("--w1", type=float, default=0.70)
    ap.add_argument("--k2", type=float, default=1.04375)
    ap.add_argument("--k3", type=float, default=1.25)
    ap.add_argument("--law1", default="Γ", choices=["Γ", "α1"])
    args = ap.parse_args()
    m = oc.load_model()
    ns, _ = oe.build_ns(m, 0.995, 0.995 ** 2 * oe.SIGMA_LIM)
    R, φ = ns["R"], ns["φ"]
    laws = [args.law1, "α1", "Γ"]
    x = s5.base_point()
    x[0], x[1], x[3], x[2], x[8], x[9] = args.n, args.D, args.w2, args.w1, args.k2, args.k3
    x[4] = 11.0
    for _ in range(60):
        r = oe.eval_candidate(x, ns, "B87_relaxed", 0.995)
        _, st = ns["распределить_π_const_hub"](x[0], x[1], ns["π_т"] ** (1 / ns["z"]))
        p = rc.profile(st[0], x[0], "Γ", R, φ)
        if p["корень"]["a1"] >= 11.15:
            break
        x[4] = min(20.0, x[4] + 0.05)
    r = oe.eval_candidate(x, ns, "B87_relaxed", 0.995)
    _, st = ns["распределить_π_const_hub"](x[0], x[1], ns["π_т"] ** (1 / ns["z"]))
    print(f"контрольная точка: n={x[0]/1e3:.2f}k D={x[1]*1e3:.1f} w2={x[3]:.3f} "
          f"w1={x[2]:.3f} k2={x[8]:.4f} k3={x[9]:.3f} a1_1={x[4]:.2f}")
    print(f"  hard={r['hard']:.3f}, a2={[round(v,1) for v in r['a2']]}, "
          f"eta={r['eta_t']:.4f}, sigma={r['sigma']:.0f}, qn={r['qn']:.4f}")
    rows = []
    for i, (s_, law) in enumerate(zip(st, laws), 1):
        d = span(s_, x[0], law, R, φ)
        ref = ns["ρ_корня"](s_, law)
        p3 = rc.profile(s_, x[0], law, R, φ)
        rows.append(dict(stage=i, law=law, G_c=s_["G_c"], G_int=d["G"],
                         eps_G=d["G"] / s_["G_c"] - 1.0,
                         G_l=s_["G_л"],
                         eps_G_rotor=d["G"] / s_["G_л"] - 1.0,
                         eq_rel=d["eq_rel"], Lu_rel=d["Lu_rel"],
                         rho_hub_ref=ref, rho_hub_check=p3["корень"]["rho"],
                         d_rho=abs(ref - p3["корень"]["rho"]),
                         a1_hub=d["a1_hub"], a1_tip=d["a1_tip"],
                         rho_g_hub=d["rho_hub"], rho_g_tip=d["rho_tip"]))
    with (RESULTS / "stage6b5_radial.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for d in rows:
        print(f"  ст.{d['stage']} ({d['law']}): G_расч/G_c−1 = {d['eps_G']:+.3e}; "
              f"G_расч/G_л−1 = {d['eps_G_rotor']:+.3e}; равновесие {d['eq_rel']:.1e}; "
              f"L_u {d['Lu_rel']:.1e}; ρ_корня эталон {d['rho_hub_ref']:.4f} "
              f"сверка Δ={d['d_rho']:.1e}; α1 корень {d['a1_hub']:.1f}° / периферия {d['a1_tip']:.1f}°")
    print("\nдля справки: G_c =", [round(s_["G_c"], 4) for s_ in st])


if __name__ == "__main__":
    main()
