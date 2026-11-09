"""Правильный residual радиального равновесия (шаг 7.1).

Для статического давления (осесимметричный поток, c_r ≈ 0):

    R_p = (1/ρ_г)·dp/dr − c_u²/r                      [м/с²]

Эквивалент через Гиббса:

    R_T = cp·dT/dr − T·ds/dr − c_u²/r

Общая форма через полную энтальпию:

    R_GRE = dh0/dr − T·ds/dr − c_a·dc_a/dr − (c_u/r)·d(r·c_u)/dr

Должно быть R_p ≈ R_T ≈ R_GRE.

    python radial_residual_correct.py
"""
from __future__ import annotations

import math

import numpy as np

import optimize_extended as oe
import scan5d as s5
import radial_check as rc
import radial_exit as rx
import optimizer_core as oc

RESULTS = oe.RESULTS
NPTS = 801


def fields(st, n, law, R, φ, r1, r2):
    cu1, ca1, c1, T1 = rx.stator(st, n, law, R, φ, r1)
    u1 = math.pi * n * r1 / 30
    u2 = math.pi * n * r2 / 30
    r_c, r_l = st["D_c"] / 2, st["D_л"] / 2
    if law == "Γ":
        cu2 = st["c2u"] * r_l / r2
        M1u = st["c1u"] / math.sqrt(st["k"] * R * st["T1_ад"])
        p1 = st["p1"] * (1 + (st["k"] - 1) / 2 * M1u ** 2 * (1 - (r_c / r1) ** 2)) ** (st["k"] / (st["k"] - 1))
    else:
        cu2 = (st["L_u"] - cu1 * u1) / u2
        p1 = st["p0"] * (1 - (c1 / φ) ** 2 / (2 * st["cp"] * st["T0"])) ** (st["k"] / (st["k"] - 1))
    ca2 = np.full_like(r2, st["c2a"])
    rho1 = p1 / (R * T1)
    w1 = np.hypot(ca1, cu1 - u1)
    w2 = np.hypot(ca2, cu2 + u2)
    T2 = T1 + (w1 ** 2 - w2 ** 2 + u2 ** 2 - u1 ** 2) / (2 * st["cp"])
    s1 = (1 / φ ** 2 - 1) * c1 ** 2 / 2 / T1
    s2 = s1 + (1 / 0.96 ** 2 - 1) * w2 ** 2 / 2 / T2
    m = NPTS // 2
    p2 = st["p2"] * (T2 / T2[m]) ** (st["k"] / (st["k"] - 1)) * np.exp(-(s2 - s2[m]) / R)
    rho2 = p2 / (R * T2)
    return dict(r1=r1, r2=r2, p1=p1, T1=T1, rho1=rho1, cu1=cu1, ca1=ca1, s1=s1,
                p2=p2, T2=T2, rho2=rho2, cu2=cu2, ca2=ca2, s2=s2,
                cp=st["cp"], k=st["k"])


def residuals(d, side="rotor"):
    if side == "rotor":
        r, p, T, s, rho, cu, ca = (d["r2"], d["p2"], d["T2"], d["s2"],
                                   d["rho2"], d["cu2"], d["ca2"])
    else:
        r, p, T, s, rho, cu, ca = (d["r1"], d["p1"], d["T1"], d["s1"],
                                   d["rho1"], d["cu1"], d["ca1"])
    cp = d["cp"]
    dp = np.gradient(p, r)
    dT = np.gradient(T, r)
    ds = np.gradient(s, r)
    dca = np.gradient(ca, r)
    dcu = np.gradient(cu, r)
    h0 = cp * T + (ca ** 2 + cu ** 2) / 2
    dh0 = np.gradient(h0, r)
    R_p = dp / rho - cu ** 2 / r
    R_T = cp * dT - T * ds - cu ** 2 / r
    R_GRE = dh0 - T * ds - ca * dca - (cu / r) * np.gradient(r * cu, r)
    ref = max(float(np.max(np.abs(dp / rho))), 1e-9)
    return dict(eps_p=float(np.max(np.abs(R_p))) / ref,
                eps_T=float(np.max(np.abs(R_T))) / ref,
                eps_GRE=float(np.max(np.abs(R_GRE))) / ref,
                d_TP=float(np.max(np.abs(R_T - R_p))),
                d_GP=float(np.max(np.abs(R_GRE - R_p))),
                ref=ref, maxRp=float(np.max(np.abs(R_p))))


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
    r_c = lambda s_: s_["D_c"] / 2
    print("a1_1=%.2f" % x[4])
    for i, law in ((1, "Γ"), (2, "α1"), (3, "Γ")):
        s_ = st[i - 1]
        r1 = np.linspace(r_c(s_) - 0.5 * s_["l_c"], r_c(s_) + 0.5 * s_["l_c"], NPTS)
        r_l = s_["D_л"] / 2
        r2 = np.linspace(r_l - 0.5 * s_["l_л"], r_l + 0.5 * s_["l_л"], NPTS)
        d = fields(s_, x[0], law, R, φ, r1, r2)
        for side in ("stator", "rotor"):
            q = residuals(d, side)
            print(f"ст.{i} ({law}) {side:6s}: ε_p={q['eps_p']:7.2%}  ε_T={q['eps_T']:7.2%}  "
                  f"ε_GRE={q['eps_GRE']:7.2%}  |R_T−R_p|={q['d_TP']:7.1f}  "
                  f"|R_GRE−R_p|={q['d_GP']:7.1f}  при max|dp/ρdr|={q['ref']:.0f} м/с²")


if __name__ == "__main__":
    main()
