"""Диагностика А для α1-ступени: независимые контроли и норма невязки.

Невязка:  R(r) = (1/ρ_г)·dp/dr − c2u²/r + T·ds/dr
Норма:    ε_RRE = max|R| / max|dp/dr|
Единицы:  p — Па, r — м, ρ_г — кг/м³, s — Дж/(кг·К).

    python alpha1_diag.py
"""
from __future__ import annotations

import math

import numpy as np

import optimize_extended as oe
import scan5d as s5
import radial_exit as rx
import radial_check as rc
import optimizer_core as oc

RESULTS = oe.RESULTS


def build(st, n, law, R, φ, npts=801, s_const=False, T_const=False):
    r_c, r_l = st["D_c"] / 2, st["D_л"] / 2
    r1 = np.linspace(r_c - 0.5 * st["l_c"], r_c + 0.5 * st["l_c"], npts)
    r2 = np.linspace(r_l - 0.5 * st["l_л"], r_l + 0.5 * st["l_л"], npts)
    cu1, ca1, c1, T1 = rx.stator(st, n, law, R, φ, r1)
    u1 = math.pi * n * r1 / 30
    w1 = np.hypot(ca1, cu1 - u1)
    s1 = (1 / φ ** 2 - 1) * c1 ** 2 / 2 / T1
    u2 = math.pi * n * r2 / 30
    if law == "Γ":
        cu2 = st["c2u"] * r_l / r2
    else:
        cu2 = (st["L_u"] - cu1 * u1) / u2
    ca2 = np.full_like(r2, st["c2a"])
    w2 = np.hypot(ca2, cu2 + u2)
    cp, k = st["cp"], st["k"]
    T2 = T1 + (w1 ** 2 - w2 ** 2 + u2 ** 2 - u1 ** 2) / (2 * cp)
    m = npts // 2
    if T_const:
        T2 = np.full_like(T2, T2[m])
    hl = (1 / 0.96 ** 2 - 1) * w2 ** 2 / 2
    s2 = s1 + hl / T2
    if s_const:
        s2 = np.full_like(s2, s2[m])
    p2 = st["p2"] * (T2 / T2[m]) ** (k / (k - 1)) * np.exp(-(s2 - s2[m]) / R)
    rho2 = p2 / (R * T2)
    dp = np.gradient(p2, r2)
    ds = np.gradient(s2, r2)
    resid = dp / rho2 - cu2 ** 2 / r2 + T2 * ds
    eps = float(np.max(np.abs(resid)) / max(np.max(np.abs(dp / rho2)), 1e-12))
    zero = [float((r2[i] - r2[0]) / (r2[-1] - r2[0]))
            for i in range(1, npts) if dp[i - 1] * dp[i] < 0]
    imax = int(np.argmax(np.abs(resid)))
    return dict(eps=eps, maxR=float(np.max(np.abs(resid))),
                maxdp=float(np.max(np.abs(dp / rho2))),
                zeros=zero, imax_frac=float((r2[imax] - r2[0]) / (r2[-1] - r2[0])),
                a2_hub=float(np.degrees(np.arctan2(ca2[0], cu2[0]))),
                a2_tip=float(np.degrees(np.arctan2(ca2[-1], cu2[-1]))),
                p_min=float(p2.min()), p_max=float(p2.max()),
                T_min=float(T2.min()), T_max=float(T2.max()))


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
    print("контрольная точка, a1_1=%.2f" % x[4])
    cases = [("исходный (s(r), T(r))", {}),
             ("s2 = const", dict(s_const=True)),
             ("T2 = const", dict(T_const=True)),
             ("s2 = const и T2 = const", dict(s_const=True, T_const=True))]
    for i, law in ((2, "α1"), (1, "Γ"), (3, "Γ")):
        print(f"\n== ступень {i} ({law}) ==")
        for name, kw in cases:
            d = build(st[i - 1], x[0], law, R, φ, **kw)
            z = ", ".join(f"{v:.3f}" for v in d["zeros"]) or "нет"
            eps = "н/д (dp/dr=0)" if d["maxdp"] < 1.0 else f"{d['eps']:7.2%}"
            print(f"  {name:24s}: ε_RRE={eps}  max|R|={d['maxR']:10.1f} Па/м  "
                  f"max|dp/ρdr|={d['maxdp']:10.1f}  нули dp/dr при x/l: {z[:60]}; "
                  f"max|R| при x/l={d['imax_frac']:.2f}")


if __name__ == "__main__":
    main()
