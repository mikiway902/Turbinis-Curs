"""6Б-4: контрольный радиальный расчёт на точках с α2_1 ∈ 70…90.

Считает профили по радиусу (корень/средний/периферия) двумя законами:
Γ = const — c1u·r = const, c1a = const, p1 по (71);
α1 = const — c1u·r^{cos²α1} = const, p1 по c1(r),
ρ по (73). Проверяет α1_корня ≥ 11°, λ_c1 ≤ 1.05, w2 > c2a и ρ_корня.

    python radial_check.py
"""
from __future__ import annotations

import csv
import math

import numpy as np

import optimize_extended as oe
import scan5d as s5
import optimizer_core as oc

RESULTS = oe.RESULTS
SECTIONS = (("корень", -0.5), ("средний", 0.0), ("периферия", 0.5))


def profile(st, n, закон, R, φ):
    k, cp, T0, p0, H0, p2 = st["k"], st["cp"], st["T0"], st["p0"], st["H0"], st["p2"]
    r_c, r_l = st["D_c"] / 2, st["D_л"] / 2
    out = {}
    for name, f in SECTIONS:
        r1 = r_c + f * st["l_c"]
        r2 = r_l + f * st["l_л"]
        u1, u2 = math.pi * n * r1 / 30, math.pi * n * r2 / 30
        if закон == "Γ":
            c1u, c1a = st["c1u"] * r_c / r1, st["c1a"]
            M1u = st["c1u"] / math.sqrt(k * R * st["T1_ад"])
            p1 = st["p1"] * (1 + (k - 1) / 2 * M1u ** 2 * (1 - (r_c / r1) ** 2)) ** (k / (k - 1))  # (71)
            c2u, c2a = st["c2u"] * r_l / r2, st["c2a"]
        else:
            q = (r_c / r1) ** (math.cos(math.radians(st["α1"])) ** 2)
            c1u, c1a = st["c1u"] * q, st["c1a"] * q
            c1 = math.hypot(c1u, c1a)
            p1 = p0 * (1 - (c1 / φ) ** 2 / (2 * cp * T0)) ** (k / (k - 1))
            c2u, c2a = (st["L_u"] - c1u * u1) / u2, st["c2a"]
        T1ad = T0 * (p1 / p0) ** ((k - 1) / k)
        rho = cp * T1ad / H0 * (1 - (p2 / p1) ** ((k - 1) / k))  # (73)
        c1 = math.hypot(c1u, c1a)
        w2 = math.hypot(c2a, c2u + u2)
        out[name] = dict(
            r=r1, rho=rho, p1=p1, c1u=c1u, c1a=c1a, c1=c1, u1=u1, u2=u2,
            w1=math.hypot(c1a, c1u - u1), w2=w2, c2a=c2a, c2u=c2u,
            a1=math.degrees(math.atan2(c1a, c1u)),
            b1=math.degrees(math.atan2(c1a, c1u - u1)),
            b2=math.degrees(math.atan2(c2a, c2u + u2)),
            lam=c1 / math.sqrt(2 * k / (k + 1) * R * T0),
        )
    return out


def main() -> None:
    m = oc.load_model()
    ns, _ = oe.build_ns(m, 0.995, 0.995 ** 2 * oe.SIGMA_LIM)
    R, φ = ns["R"], ns["φ"]
    rows = [r for r in s5.load() if r["status"] == "ok" and 70 <= r["a2_1"] <= 90]
    out = []
    for b in rows:
        x = s5.base_point()
        x[0], x[1], x[3], x[2], x[8] = b["n"], b["D_vt"], b["w2"], b["w1"], b["k2"]
        r = oe.eval_candidate(x, ns, "B87_relaxed", 0.995)
        _, st = ns["распределить_π_const_hub"](x[0], x[1], ns["π_т"] ** (1 / ns["z"]))
        for закон in ("Γ", "α1"):
            p = profile(st[0], b["n"], закон, R, φ)
            h, mid, tip = p["корень"], p["средний"], p["периферия"]
            out.append(dict(
                n=b["n"], D=b["D_vt"], w2=b["w2"], w1=b["w1"], k2=b["k2"], law=закон,
                a2_1=r["a2"][0], a2_2=r["a2"][1], a2_3=r["a2"][2],
                eta=r["eta_t"], sigma=r["sigma"], qn=r["qn"], hard=r["hard"],
                rho_hub=h["rho"], rho_mid=mid["rho"], rho_tip=tip["rho"],
                a1_hub=h["a1"], a1_mid=mid["a1"], a1_tip=tip["a1"],
                lam_mid=mid["lam"], lam_hub=h["lam"], lam_tip=tip["lam"],
                w2_hub=h["w2"], c2a_hub=h["c2a"],
                w2_mid=mid["w2"], c2a_mid=mid["c2a"],
                w2_tip=tip["w2"], c2a_tip=tip["c2a"],
                rho_ok=bool(h["rho"] >= 0.05),
                a1_ok=bool(h["a1"] >= 11.0),
                lam_ok=bool(mid["lam"] <= 1.05),
                w2_ok=bool(h["w2"] > h["c2a"] and mid["w2"] > mid["c2a"]
                           and tip["w2"] > tip["c2a"]),
            ))
    with (RESULTS / "stage6b_radial.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for закон in ("Γ", "α1"):
        sub = [r for r in out if r["law"] == закон]
        print(f"== закон {закон}: {len(sub)} точек")
        print("   rho_hub: %.4f…%.4f  (>=0.05: %d)" % (
            min(r["rho_hub"] for r in sub), max(r["rho_hub"] for r in sub),
            sum(r["rho_ok"] for r in sub)))
        print("   alpha1_hub: %.1f…%.1f°  (>=11: %d)" % (
            min(r["a1_hub"] for r in sub), max(r["a1_hub"] for r in sub),
            sum(r["a1_ok"] for r in sub)))
        print("   lambda_c1 mid: %.3f…%.3f  (<=1.05: %d)" % (
            min(r["lam_mid"] for r in sub), max(r["lam_mid"] for r in sub),
            sum(r["lam_ok"] for r in sub)))
        print("   w2>c2a во всех сечениях: %d" % sum(r["w2_ok"] for r in sub))
        print("   всё вместе: %d" % sum(r["rho_ok"] and r["a1_ok"] and r["lam_ok"]
                                        and r["w2_ok"] for r in sub))
    both = [r for r in out if r["law"] == "Γ" and r["rho_ok"] and r["a1_ok"]
            and r["lam_ok"] and r["w2_ok"]]
    for r in sorted(both, key=lambda r: -r["eta"])[:5]:
        print("  Γ ok: n=%.0fk D=%.0f a2=%.1f/%.1f/%.1f rho=%.4f a1h=%.1f lam=%.3f eta=%.4f sig=%.0f" % (
            r["n"] / 1e3, r["D"] * 1e3, r["a2_1"], r["a2_2"], r["a2_3"],
            r["rho_hub"], r["a1_hub"], r["lam_mid"], r["eta"], r["sigma"]))


if __name__ == "__main__":
    main()
