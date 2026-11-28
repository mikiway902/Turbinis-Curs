"""6Б-4.5: замыкание выхода РК и чувствительность к BC.

Выход РК: c2u(r), c2a(r), w2(r), T2(r) из энергии в относительном движении,
s2(r) = s1(r) + (ψ⁻²−1)w2²/(2T2), p2(r) из состояния при известном p2 на
среднем сечении. Проверки: расход ∫ρ2·c2a·2πr dr, радиальное равновесие
dp2/dr = ρ2(c2u²/r − T2·ds2/dr), невязки.

BC: якорь p на среднем сечении против якоря на втулке — абсолютные p, ρ
и невязка равновесия.

    python radial_exit.py
"""
from __future__ import annotations

import math

import numpy as np

import optimize_extended as oe
import scan5d as s5
import radial_check as rc
import optimizer_core as oc

RESULTS = oe.RESULTS


def stator(st, n, law, R, φ, r):
    r_c = st["D_c"] / 2
    if law == "Γ":
        cu = st["c1u"] * r_c / r
        ca = np.full_like(r, st["c1a"])
    else:
        q = (r_c / r) ** (math.cos(math.radians(st["α1"])) ** 2)
        cu, ca = st["c1u"] * q, st["c1a"] * q
    c1 = np.hypot(cu, ca)
    T1 = st["T0"] - c1 ** 2 / (2 * st["cp"])
    return cu, ca, c1, T1


def rotor_exit(st, n, law, R, φ, npts=401):
    r_c, r_l = st["D_c"] / 2, st["D_л"] / 2
    r1 = np.linspace(r_c - 0.5 * st["l_c"], r_c + 0.5 * st["l_c"], npts)
    r2 = np.linspace(r_l - 0.5 * st["l_л"], r_l + 0.5 * st["l_л"], npts)
    cu1, ca1, c1, T1 = stator(st, n, law, R, φ, r1)
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
    hl = (1 / st.get("ψ", 0.96) ** 2 - 1) * w2 ** 2 / 2
    s2 = s1 + hl / T2
    m = npts // 2
    p2 = st["p2"] * (T2 / T2[m]) ** (k / (k - 1)) * np.exp(-(s2 - s2[m]) / R)
    rho2 = p2 / (R * T2)
    G = float(np.trapezoid(rho2 * ca2 * 2 * math.pi * r2, r2))
    dp = np.gradient(p2, r2)
    dsdr = np.gradient(s2, r2)
    eq = dp - rho2 * (cu2 ** 2 / r2 - T2 * dsdr)
    # невязка нормируется на max|dp| по всему сечению: поточечное деление
    # взрывается там, где dp/dr проходит через ноль
    return dict(G=G, G_stage=st["G_л"],
                eps_G=G / st["G_л"] - 1,
                eq_rel=float(np.max(np.abs(eq)) / max(float(np.max(np.abs(dp))), 1e-9)),
                rho_hub=float(rho2[0]), rho_tip=float(rho2[-1]),
                p_hub=float(p2[0]), p_tip=float(p2[-1]),
                T_hub=float(T2[0]), T_tip=float(T2[-1]),
                T2_mid_err=float(T2[m] / st["T2"] - 1),
                w2_ok=bool(np.all(w2 > ca2)),
                a2_hub=float(np.degrees(np.arctan2(ca2[0], cu2[0]))),
                a2_tip=float(np.degrees(np.arctan2(ca2[-1], cu2[-1]))))


def bc_table(st, n, law, R, φ, npts=401):
    """Сравнение якорей: среднее сечение против втулки."""
    r_c = st["D_c"] / 2
    r = np.linspace(r_c - 0.5 * st["l_c"], r_c + 0.5 * st["l_c"], npts)
    cu, ca, c1, T = stator(st, n, law, R, φ, r)
    k = st["k"]
    # якорь A: p(r_ср) = p1, интегрируем в обе стороны с T(энергия)
    dlnp = cu ** 2 / (R * T * r)
    m = npts // 2
    integ = np.concatenate(([0.0], np.cumsum(0.5 * (dlnp[1:] + dlnp[:-1]) * np.diff(r))))
    pA = st["p1"] * np.exp(integ - integ[m])
    # якорь B: p(втулка) = p_book(втулка) по (71), интегрируем наружу
    if law == "Γ":
        M1u = st["c1u"] / math.sqrt(k * R * st["T1_ад"])
        p_hub_book = st["p1"] * (1 + (k - 1) / 2 * M1u ** 2 * (1 - (r_c / r[0]) ** 2)) ** (k / (k - 1))
    else:
        c1h = c1[0]
        p_hub_book = st["p0"] * (1 - (c1h / φ) ** 2 / (2 * st["cp"] * st["T0"])) ** (k / (k - 1))
    pB = p_hub_book * np.exp(integ - integ[0])
    rhoA, rhoB = pA / (R * T), pB / (R * T)
    def react(p_hub):
        T1ad = st["T0"] * (p_hub / st["p0"]) ** ((k - 1) / k)
        return st["cp"] * T1ad / st["H0"] * (1 - (st["p2"] / p_hub) ** ((k - 1) / k))
    return dict(p_hubA=float(pA[0]), p_midA=float(pA[m]), p_tipA=float(pA[-1]),
                p_hubB=float(pB[0]), p_midB=float(pB[m]), p_tipB=float(pB[-1]),
                rho_hubA=float(rhoA[0]), rho_midA=float(rhoA[m]), rho_tipA=float(rhoA[-1]),
                rho_hubB=float(rhoB[0]), rho_midB=float(rhoB[m]), rho_tipB=float(rhoB[-1]),
                rho_reactA=react(float(pA[0])), rho_reactB=react(float(p_hub_book)),
                eqA=float(np.max(np.abs(np.gradient(pA, r) - rhoA * cu ** 2 / r)
                                 / np.maximum(np.abs(np.gradient(pA, r)), 1e-9))),
                eqB=float(np.max(np.abs(np.gradient(pB, r) - rhoB * cu ** 2 / r)
                                 / np.maximum(np.abs(np.gradient(pB, r)), 1e-9))))


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
    laws = ["Γ", "α1", "Γ"]
    print("контрольная точка: a1=%.2f" % x[4])
    print("\n== выход РК ==")
    for i, (s_, law) in enumerate(zip(st, laws), 1):
        d = rotor_exit(s_, x[0], law, R, φ)
        print(f"ст.{i} ({law}): расход {d['eps_G']:+.3%} (G_л), равновесие {d['eq_rel']:.2%}, "
              f"T2(ср) Δ={d['T2_mid_err']:+.2e}, w2>c2a {d['w2_ok']}")
        print(f"   p2 вт/ср/пер: {d['p_hub']/1e3:.1f}/{s_['p2']/1e3:.1f}/{d['p_tip']/1e3:.1f} кПа; "
              f"T2 {d['T_hub']:.0f}/{s_['T2']:.0f}/{d['T_tip']:.0f} K; "
              f"α2 вт/пер {d['a2_hub']:.1f}/{d['a2_tip']:.1f}°")
    print("\n== BC: якорь на среднем (A) против якоря на втулке (B) ==")
    for i, (s_, law) in enumerate(zip(st, laws), 1):
        d = bc_table(s_, x[0], law, R, φ)
        print(f"ст.{i} ({law}):")
        print(f"   p вт/ср/пер: A {d['p_hubA']/1e3:.1f}/{d['p_midA']/1e3:.1f}/{d['p_tipA']/1e3:.1f} кПа; "
              f"B {d['p_hubB']/1e3:.1f}/{d['p_midB']/1e3:.1f}/{d['p_tipB']/1e3:.1f}")
        print(f"   ρ_г вт/ср/пер: A {d['rho_hubA']:.3f}/{d['rho_midA']:.3f}/{d['rho_tipA']:.3f}; "
              f"B {d['rho_hubB']:.3f}/{d['rho_midB']:.3f}/{d['rho_tipB']:.3f}")
        print(f"   ρ_корня: A {d['rho_reactA']:.4f}; B {d['rho_reactB']:.4f}; "
              f"невязка равновесия A {d['eqA']:.2%}, B {d['eqB']:.2%}")


if __name__ == "__main__":
    main()
