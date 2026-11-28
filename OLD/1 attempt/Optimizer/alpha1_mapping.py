"""Проверка гипотезы: сопоставление линий тока по накопленному расходу.

1) На исходном (equal-span) решении считаются два residual:
   R_full = (1/ρ)dp/dr − c2u²/r + T·ds/dr   [м/с²]
   R_h    = cp·dT/dr − c2u²/r               [м/с²]
2) Сопоставление r1↔r2 по равенству накопленного массового расхода
   (2–3 итерации), затем те же residuals.

    python alpha1_mapping.py
"""
from __future__ import annotations

import math

import numpy as np

import optimize_extended as oe
import scan5d as s5
import radial_check as rc
import optimizer_core as oc

RESULTS = oe.RESULTS
NPTS = 801


def stator(st, n, law, R, φ, r):
    r_c = st["D_c"] / 2
    if law == "Γ":
        cu = st["c1u"] * r_c / r
        ca = np.full_like(r, st["c1a"])
        k = st["k"]
        M1u = st["c1u"] / math.sqrt(k * R * st["T1_ад"])
        p = st["p1"] * (1 + (k - 1) / 2 * M1u ** 2 * (1 - (r_c / r) ** 2)) ** (k / (k - 1))
    else:
        q = (r_c / r) ** (math.cos(math.radians(st["α1"])) ** 2)
        cu, ca = st["c1u"] * q, st["c1a"] * q
        c1 = np.hypot(cu, ca)
        p = st["p0"] * (1 - (c1 / φ) ** 2 / (2 * st["cp"] * st["T0"])) ** (st["k"] / (st["k"] - 1))
    c1 = np.hypot(cu, ca)
    T = st["T0"] - c1 ** 2 / (2 * st["cp"])
    return cu, ca, c1, T, p, p / (R * T)


def mu_of(r, rho, ca):
    g = rho * ca * 2 * math.pi * r
    m = np.concatenate(([0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(r))))
    return m / m[-1]


def build(st, n, law, R, φ, npts=NPTS, mapping="span", iters=4):
    r_c, r_l = st["D_c"] / 2, st["D_л"] / 2
    r1 = np.linspace(r_c - 0.5 * st["l_c"], r_c + 0.5 * st["l_c"], npts)
    r2 = np.linspace(r_l - 0.5 * st["l_л"], r_l + 0.5 * st["l_л"], npts)
    cu1, ca1, c1, T1, p1, rho1 = stator(st, n, law, R, φ, r1)
    mu1 = mu_of(r1, rho1, ca1)
    u2 = math.pi * n * r2 / 30
    ca2 = np.full_like(r2, st["c2a"])
    k, cp = st["k"], st["cp"]
    # стартовое поле (equal-span)
    idx = np.arange(npts)
    rho2 = np.full_like(r2, st["p2"] / (R * st["T2"]))
    for _ in range(1 if mapping == "span" else iters):
        cu1m, ca1m, c1m, T1m = cu1[idx], ca1[idx], c1[idx], T1[idx]
        u1m = math.pi * n * r1[idx] / 30
        w1 = np.hypot(ca1m, cu1m - u1m)
        cu2 = (st["L_u"] - cu1m * u1m) / u2
        w2 = np.hypot(ca2, cu2 + u2)
        T2 = T1m + (w1 ** 2 - w2 ** 2 + u2 ** 2 - u1m ** 2) / (2 * cp)
        s1 = (1 / φ ** 2 - 1) * c1m ** 2 / 2 / T1m
        s2 = s1 + (1 / 0.96 ** 2 - 1) * w2 ** 2 / 2 / T2
        m = npts // 2
        p2 = st["p2"] * (T2 / T2[m]) ** (k / (k - 1)) * np.exp(-(s2 - s2[m]) / R)
        rho2 = p2 / (R * T2)
        if mapping == "span":
            break
        mu2 = mu_of(r2, rho2, ca2)
        idx = np.clip(np.searchsorted(mu1, mu2), 0, npts - 1)
    dp = np.gradient(p2, r2)
    ds = np.gradient(s2, r2)
    dT = np.gradient(T2, r2)
    R_full = dp / rho2 - cu2 ** 2 / r2 + T2 * ds
    R_h = cp * dT - cu2 ** 2 / r2
    ref = max(float(np.max(np.abs(dp / rho2))), 1e-9)
    return dict(eps_full=float(np.max(np.abs(R_full))) / ref,
                eps_h=float(np.max(np.abs(R_h))) / ref,
                max_diff=float(np.max(np.abs(R_full - R_h))),
                max_ref=ref,
                T_range=(float(T2.min()), float(T2.max())),
                p_range=(float(p2.min()), float(p2.max())),
                a2=(float(np.degrees(np.arctan2(ca2[0], cu2[0]))),
                    float(np.degrees(np.arctan2(ca2[-1], cu2[-1])))))


def main() -> None:
    m = oc.load_model()
    ns, _ = oe.build_ns(m, 0.995, 0.995 ** 2 * oe.SIGMA_LIM)
    R, φ = ns["R"], ns["φ"]
    x = s5.base_point()
    x[0], x[1], x[3], x[2], x[8], x[9] = 90000, 0.062, 0.8125, 0.70, 1.04375, 1.25
    x[4] = 11.0
    for _ in range(60):
        oe.eval_candidate(x, ns, "B87_relaxed", 0.995)
        _, st = ns["распределить_π_const_hub"](x[0], x[1], ns["π_т"] ** (1 / ns["z"]))
        if rc.profile(st[0], x[0], "Γ", R, φ)["корень"]["a1"] >= 11.15:
            break
        x[4] = min(20.0, x[4] + 0.05)
    oe.eval_candidate(x, ns, "B87_relaxed", 0.995)
    _, st = ns["распределить_π_const_hub"](x[0], x[1], ns["π_т"] ** (1 / ns["z"]))
    print("a1_1=%.2f" % x[4])
    for i, law in ((2, "α1"), (1, "Γ"), (3, "Γ")):
        print(f"\nст.{i} ({law}):")
        for name, mp in (("equal-span", "span"), ("equal-mass", "mass")):
            d = build(st[i - 1], x[0], law, R, φ, mapping=mp)
            print(f"  {name:10s}: ε_full={d['eps_full']:7.2%}  ε_h={d['eps_h']:7.2%}  "
                  f"max|R_full−R_h|={d['max_diff']:8.1f} при max|dp/(ρdr)|={d['max_ref']:9.1f} м/с²; "
                  f"α2 {d['a2'][0]:.1f}/{d['a2'][1]:.1f}°")


if __name__ == "__main__":
    main()
