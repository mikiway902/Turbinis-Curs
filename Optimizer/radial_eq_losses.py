"""6Б-4.5: радиальное равновесие с потерями — прототип.

Сравнивает приближение книги (изоэнтропические законы профилирования)
с решением уравнения

    (1/ρ) dp/dr = c_u²/r − T·ds/dr

где энтропия берётся из действующей модели потерь СА:
h_c(r) = (φ⁻²−1)·c1²/2,  s(r) = s_0 + h_c(r)/T(r).

Граничное условие — p(r_ср) = p1 из среднесечного расчёта ступени.
Считается расход, реактивность по (73) и углы.

    python radial_eq_losses.py
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


def fields(st, n, law, R, φ, r):
    r_c, r_l = st["D_c"] / 2, st["D_л"] / 2
    u = math.pi * n * r / 30
    if law == "Γ":
        cu = st["c1u"] * r_c / r
        ca = np.full_like(r, st["c1a"])
    else:
        q = (r_c / r) ** (math.cos(math.radians(st["α1"])) ** 2)
        cu, ca = st["c1u"] * q, st["c1a"] * q
    c1 = np.hypot(cu, ca)
    T = st["T0"] - c1 ** 2 / (2 * st["cp"])
    return u, cu, ca, c1, T


def book_p(st, law, R, φ, r):
    r_c = st["D_c"] / 2
    if law == "Γ":
        k, M1u = st["k"], st["c1u"] / math.sqrt(st["k"] * R * st["T1_ад"])
        return st["p1"] * (1 + (k - 1) / 2 * M1u ** 2 * (1 - (r_c / r) ** 2)) ** (k / (k - 1))
    q = (r_c / r) ** (math.cos(math.radians(st["α1"])) ** 2)
    c1 = st["c1"] * q
    k, p0, cp, T0 = st["k"], st["p0"], st["cp"], st["T0"]
    return p0 * (1 - (c1 / φ) ** 2 / (2 * cp * T0)) ** (k / (k - 1))


def solve(st, n, law, R, φ, npts=401):
    r_c = st["D_c"] / 2
    r = np.linspace(r_c - 0.5 * st["l_c"], r_c + 0.5 * st["l_c"], npts)
    u, cu, ca, c1, T = fields(st, n, law, R, φ, r)
    # энтропия: h_c/T, h_c — потери СА по действующей модели
    hc = (1 / φ ** 2 - 1) * c1 ** 2 / 2
    s = hc / T
    dsdr = np.gradient(s, r)
    dlnp = cu ** 2 / (R * T * r) - dsdr / R
    mid = npts // 2
    integ = np.concatenate(([0.0], np.cumsum(0.5 * (dlnp[1:] + dlnp[:-1]) * np.diff(r))))
    p_loss = st["p1"] * np.exp(integ - integ[mid])
    rho_loss = p_loss / (R * T)
    p_book = book_p(st, law, R, φ, r)
    rho_book = p_book / (R * T)
    G_loss = float(np.trapezoid(rho_loss * ca * 2 * math.pi * r, r))
    G_book = float(np.trapezoid(rho_book * ca * 2 * math.pi * r, r))
    # реактивность по (73) на втулке для обеих моделей
    def rho_react(p1_hub):
        T1ad = st["T0"] * (p1_hub / st["p0"]) ** ((st["k"] - 1) / st["k"])
        return st["cp"] * T1ad / st["H0"] * (1 - (st["p2"] / p1_hub) ** ((st["k"] - 1) / st["k"]))
    return dict(
        r=r, p_loss=p_loss, p_book=p_book, rho_loss=rho_loss, rho_book=rho_book,
        G_loss=G_loss, G_book=G_book,
        p_hub_loss=float(p_loss[0]), p_hub_book=float(p_book[0]),
        rho_react_loss=rho_react(float(p_loss[0])), rho_react_book=rho_react(float(p_book[0])),
        T_hub=float(T[0]), c1a_hub=float(ca[0]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--law1", default="Γ", choices=["Γ", "α1"])
    args = ap.parse_args()
    m = oc.load_model()
    ns, _ = oe.build_ns(m, 0.995, 0.995 ** 2 * oe.SIGMA_LIM)
    R, φ = ns["R"], ns["φ"]
    x = s5.base_point()
    x[0], x[1], x[3], x[2], x[8], x[9] = 90000, 0.062, 0.8125, 0.70, 1.04375, 1.25
    x[4] = 11.0
    for _ in range(60):
        # eval_candidate подбирает расход под N=110 кВт; без него G не тот
        oe.eval_candidate(x, ns, "B87_relaxed", 0.995)
        _, st = ns["распределить_π_const_hub"](x[0], x[1], ns["π_т"] ** (1 / ns["z"]))
        if rc.profile(st[0], x[0], args.law1, R, φ)["корень"]["a1"] >= 11.15:
            break
        x[4] = min(20.0, x[4] + 0.05)
    oe.eval_candidate(x, ns, "B87_relaxed", 0.995)
    _, st = ns["распределить_π_const_hub"](x[0], x[1], ns["π_т"] ** (1 / ns["z"]))
    laws = [args.law1, "α1", "Γ"]
    rows = []
    for i, (s_, law) in enumerate(zip(st, laws), 1):
        d = solve(s_, x[0], law, R, φ)
        rows.append(dict(stage=i, law=law,
                         G_book=d["G_book"], G_loss=d["G_loss"], G_stage=s_["G_c"],
                         eps_book=d["G_book"] / s_["G_c"] - 1,
                         eps_loss=d["G_loss"] / s_["G_c"] - 1,
                         p_hub_book=d["p_hub_book"], p_hub_loss=d["p_hub_loss"],
                         dp_hub=(d["p_hub_loss"] / d["p_hub_book"] - 1),
                         rho_react_book=d["rho_react_book"], rho_react_loss=d["rho_react_loss"],
                         drho=d["rho_react_loss"] - d["rho_react_book"],
                         T_hub=d["T_hub"], c1a_hub=d["c1a_hub"]))
    with (RESULTS / "stage6b45_losses.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for d in rows:
        print(f"ст.{d['stage']} ({d['law']}): "
              f"p_hub loss/book = {d['dp_hub']:+.3%}; "
              f"ρ_корня {d['rho_react_book']:.4f} → {d['rho_react_loss']:.4f} (Δ={d['drho']:+.4f}); "
              f"G {d['eps_book']:+.2%} → {d['eps_loss']:+.2%}")


if __name__ == "__main__":
    main()
