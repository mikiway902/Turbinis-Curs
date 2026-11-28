"""6Б-4.5: разложение неопределённости радиальной модели.

Считает на контрольной точке:
  M0_book_exact          — (71) буквально, T изоэнтропическая;
  M0_radial_isentropic   — интеграл dlnp с T_ад (самосогласованность);
  M1_energy_T            — интеграл с фактической T (энергия);
  Δs по состояниям       — cp·ln(T2/T1) − R·ln(p2/p1) на втулке;
  Δs по h/T              — сравнение с приближением s = h_c/T;
  чувствительность к BC — якорь p на втулке вместо среднего.

    python radial_uncertainty.py
"""
from __future__ import annotations

import math

import numpy as np

import optimize_extended as oe
import scan5d as s5
import radial_check as rc
import radial_eq_losses as rel
import optimizer_core as oc

RESULTS = oe.RESULTS


def fields(st, n, law, R, r):
    r_c = st["D_c"] / 2
    if law == "Γ":
        cu = st["c1u"] * r_c / r
        ca = np.full_like(r, st["c1a"])
    else:
        q = (r_c / r) ** (math.cos(math.radians(st["α1"])) ** 2)
        cu, ca = st["c1u"] * q, st["c1a"] * q
    c1 = np.hypot(cu, ca)
    return cu, ca, c1, st["T0"] - c1 ** 2 / (2 * st["cp"])


def book_p(st, law, R, φ, r):
    r_c = st["D_c"] / 2
    if law == "Γ":
        k = st["k"]
        M1u = st["c1u"] / math.sqrt(k * R * st["T1_ад"])
        return st["p1"] * (1 + (k - 1) / 2 * M1u ** 2 * (1 - (r_c / r) ** 2)) ** (k / (k - 1))
    q = (r_c / r) ** (math.cos(math.radians(st["α1"])) ** 2)
    c1 = st["c1"] * q
    return st["p0"] * (1 - (c1 / φ) ** 2 / (2 * st["cp"] * st["T0"])) ** (st["k"] / (st["k"] - 1))


def integrate(dlnp, r, anchor_idx, anchor_val):
    integ = np.concatenate(([0.0], np.cumsum(0.5 * (dlnp[1:] + dlnp[:-1]) * np.diff(r))))
    return anchor_val * np.exp(integ - integ[anchor_idx])


def analysis(st, n, law, R, φ, npts=801):
    r_c = st["D_c"] / 2
    r = np.linspace(r_c - 0.5 * st["l_c"], r_c + 0.5 * st["l_c"], npts)
    mid = npts // 2
    cu, ca, c1, T_en = fields(st, n, law, R, r)
    p_book = book_p(st, law, R, φ, r)
    T_ad = st["T0"] * (p_book / st["p0"]) ** ((st["k"] - 1) / st["k"])
    # самосогласованность (71): интеграл с T_ад должен дать p_book
    dlnp_ad = cu ** 2 / (R * T_ad * r)
    p_isen = integrate(dlnp_ad, r, mid, st["p1"])
    dlnp_en = cu ** 2 / (R * T_en * r)
    p_en = integrate(dlnp_en, r, mid, st["p1"])
    # якорь на втулке: p_book(hub) как BC, интегрируем наружу
    p_bc_hub = integrate(dlnp_en, r, 0, p_book[0])
    # энтропия по состояниям (поле M1) vs приближение h/T на втулке
    p_m1 = rel.solve(st, n, law, R, φ, npts=npts, closure="M1")["p_loss"]
    ds_states = st["cp"] * math.log(T_en[0] / T_en[mid]) - R * math.log(p_m1[0] / p_m1[mid])
    hc = (1 / φ ** 2 - 1) * c1 ** 2 / 2
    ds_hT = hc[0] / T_en[0] - hc[mid] / T_en[mid]
    k = st["k"]
    def react(p_hub):
        T1ad = st["T0"] * (p_hub / st["p0"]) ** ((k - 1) / k)
        return st["cp"] * T1ad / st["H0"] * (1 - (st["p2"] / p_hub) ** ((k - 1) / k))
    return dict(
        p_hub_book=float(p_book[0]), p_hub_isen=float(p_isen[0]),
        p_hub_energy=float(p_en[0]), p_bc_mid=float(p_bc_hub[mid]),
        rho_book=react(float(p_book[0])), rho_isen=react(float(p_isen[0])),
        rho_energy=react(float(p_en[0])),
        ds_states=ds_states, ds_hT=ds_hT,
    )


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
    print("контрольная точка: n=90k, D=62, w2=0.8125, w1=0.70, k2=1.04375, k3=1.25, a1=%.2f" % x[4])
    for i, (s_, law) in enumerate(zip(st, laws), 1):
        d = analysis(s_, x[0], law, R, φ)
        print(f"\nст.{i} ({law}):")
        print(f"  p_hub: книга(71) {d['p_hub_book']/1e3:.2f} кПа; "
              f"интеграл с T_ад {d['p_hub_isen']/1e3:.2f} (Δ={d['p_hub_isen']/d['p_hub_book']-1:+.3%}); "
              f"интеграл с T(энергия) {d['p_hub_energy']/1e3:.2f} (Δ={d['p_hub_energy']/d['p_hub_book']-1:+.3%})")
        print(f"  ρ_корня: книга {d['rho_book']:.4f}; T_ад {d['rho_isen']:.4f}; "
              f"T(энергия) {d['rho_energy']:.4f}")
        print(f"  BC: интеграл от втулки даёт на среднем сечении p = {d['p_bc_mid']/1e3:.2f} кПа "
              f"против p1 = {s_['p1']/1e3:.2f} (Δ={d['p_bc_mid']/s_['p1']-1:+.3%})")
        print(f"  Δs втулка: через состояния {d['ds_states']:+.4f} Дж/(кг·К); "
              f"через h/T {d['ds_hT']:+.4f} (отношение {d['ds_hT']/d['ds_states'] if abs(d['ds_states'])>1e-9 else float('nan'):.3f})")


if __name__ == "__main__":
    main()
