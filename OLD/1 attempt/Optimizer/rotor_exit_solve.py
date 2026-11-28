"""Система Б (минимальная): замыкание выхода РК.

Неизвестные: c2u(r) на N узлах и масштаб осевой скорости λ_a.
Уравнения:
  (1) dp2/dr = (p2/(R·T2))·c2u²/r          — N−2 внутренних узлов
  (2) T2(r) из энергии РК                   — исключается
  (3) s2 = s(T2, p2) = s1 + Δs_л(w2)        — задаёт p2 по состоянию
  (4) средняя L_u = L_u ступени             — 1 уравнение
  (5) ∫ρ2 c2a 2πr dr = G_л                  — 1 уравнение
BC: p2(r_ср) = p2 ступени (входит в (3)).
Гиббс отдельно не добавляется — он следствие (3).

    python rotor_exit_solve.py [--stage 1|2|3]
"""
from __future__ import annotations

import argparse
import math

import numpy as np
from scipy.optimize import least_squares

import optimize_extended as oe
import scan5d as s5
import radial_check as rc
import radial_exit as rx
import optimizer_core as oc

N = 41
DEG = 8  # степень полинома для c2u(r); 4 -> 4.6-100 %, 6-8 -> 0.4-15 %


def setup(st, n, law, R, φ):
    r_c, r_l = st["D_c"] / 2, st["D_л"] / 2
    r1 = np.linspace(r_c - 0.5 * st["l_c"], r_c + 0.5 * st["l_c"], N)
    r2 = np.linspace(r_l - 0.5 * st["l_л"], r_l + 0.5 * st["l_л"], N)
    cu1, ca1, c1, T1 = rx.stator(st, n, law, R, φ, r1)
    u1 = math.pi * n * r1 / 30
    u2 = math.pi * n * r2 / 30
    w1 = np.hypot(ca1, cu1 - u1)
    s1 = (1 / φ ** 2 - 1) * c1 ** 2 / 2 / T1
    if law == "Γ":
        cu2_0 = st["c2u"] * r_l / r2
    else:
        cu2_0 = (st["L_u"] - cu1 * u1) / u2
    return dict(r1=r1, r2=r2, cu1=cu1, ca1=ca1, T1=T1, u1=u1, u2=u2,
                w1=w1, s1=s1, cu2_0=cu2_0, cp=st["cp"], k=st["k"],
                p2m=st["p2"], L_u=st["L_u"], G=st["G_л"], ca20=st["c2a"])


def fields(g, cu2, lam):
    ca2 = lam * g["ca20"]
    w2 = np.hypot(ca2, cu2 + g["u2"])
    T2 = g["T1"] + (g["w1"] ** 2 - w2 ** 2 + g["u2"] ** 2 - g["u1"] ** 2) / (2 * g["cp"])
    s2 = g["s1"] + (1 / 0.96 ** 2 - 1) * w2 ** 2 / 2 / T2
    m = N // 2
    p2 = g["p2m"] * (T2 / T2[m]) ** (g["k"] / (g["k"] - 1)) * np.exp(-(s2 - s2[m]) / R_GAS)
    return ca2, w2, T2, s2, p2


R_GAS = None


def poly_coeffs(x):
    return x[:DEG + 1]


def cu2_of(x, g):
    xi = (g["r2"] - g["r2"][0]) / (g["r2"][-1] - g["r2"][0])
    return np.polyval(poly_coeffs(x), xi)


def residuals(x, g):
    cu2 = cu2_of(x, g)
    lam = x[DEG + 1]
    ca2, w2, T2, s2, p2 = fields(g, cu2, lam)
    dp = np.gradient(p2, g["r2"])
    eq = dp - p2 / (R_GAS * T2) * cu2 ** 2 / g["r2"]
    L_u = g["cu1"] * g["u1"] + cu2 * g["u2"]
    rho2 = p2 / (R_GAS * T2)
    Gm = float(np.trapezoid(rho2 * ca2 * 2 * math.pi * g["r2"], g["r2"]))
    eqi = eq[1:-1] / max(float(np.max(np.abs(dp))), 1e-9)
    return np.concatenate([eqi,
                           [(float(np.mean(L_u)) / g["L_u"] - 1.0) * 10],
                           [Gm / g["G"] - 1.0]])


def solve(st, n, law, R, φ):
    g = setup(st, n, law, R, φ)
    xi = (g["r2"] - g["r2"][0]) / (g["r2"][-1] - g["r2"][0])
    coef0 = np.polyfit(xi, g["cu2_0"], DEG)
    x0 = np.concatenate([coef0, [1.0]])
    res = least_squares(residuals, x0, args=(g,), xtol=1e-13, ftol=1e-13,
                        gtol=1e-13, max_nfev=20000, method="lm")
    cu2, lam = cu2_of(res.x, g), res.x[DEG + 1]
    ca2, w2, T2, s2, p2 = fields(g, cu2, lam)
    dp = np.gradient(p2, g["r2"])
    eq = dp - p2 / (R_GAS * T2) * cu2 ** 2 / g["r2"]
    rho2 = p2 / (R_GAS * T2)
    Gm = float(np.trapezoid(rho2 * ca2 * 2 * math.pi * g["r2"], g["r2"]))
    L_u = g["cu1"] * g["u1"] + cu2 * g["u2"]
    a2 = np.degrees(np.arctan2(ca2, cu2))
    return dict(ok=res.success, cost=res.cost, nfev=res.nfev,
                eps=float(np.max(np.abs(eq)) / max(float(np.max(np.abs(dp))), 1e-9)),
                eps_mass=Gm / g["G"] - 1.0,
                Lu_spread=float(np.max(np.abs(L_u / g["L_u"] - 1.0))),
                lam=lam, a2=(float(a2[0]), float(a2[-1])),
                w2_ok=bool(np.all(w2 > ca2)),
                p=(float(p2[0]), float(p2[-1])),
                cu2=(float(cu2[0]), float(cu2[-1])),
                rho=(float(rho2[0]), float(rho2[-1])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=2)
    args = ap.parse_args()
    global R_GAS
    m = oc.load_model()
    ns, _ = oe.build_ns(m, 0.995, 0.995 ** 2 * oe.SIGMA_LIM)
    R, φ = ns["R"], ns["φ"]
    R_GAS = R
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
    law = ["Γ", "α1", "Γ"][args.stage - 1]
    d = solve(st[args.stage - 1], x[0], law, R, φ)
    print(f"ст.{args.stage} ({law}): success={d['ok']} nfev={d['nfev']} cost={d['cost']:.2e}")
    print(f"  ε_eq={d['eps']:.3%}  ε_mass={d['eps_mass']:+.3%}  разброс L_u={d['Lu_spread']:.3%}")
    print(f"  λ_a={d['lam']:.4f}; α2 вт/пер={d['a2'][0]:.1f}/{d['a2'][1]:.1f}°; "
          f"w2>c2a={d['w2_ok']}; p2 {d['p'][0]/1e3:.1f}…{d['p'][1]/1e3:.1f} кПа; "
          f"c2u {d['cu2'][0]:.1f}…{d['cu2'][1]:.1f} м/с")


if __name__ == "__main__":
    main()
