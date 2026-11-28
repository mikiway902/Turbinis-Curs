"""Независимые проверки физических инвариантов (без изменения модели).

1. η_u = L_u/H0  ==  1 − ξ_c' − ξ_л − ξ_в  (п. 102 и 111).
2. ξ_з учитывается отдельно: η_т1 = η_u − ξ_з, η_т = η_т1 − ξ_тв.
3. q_n: n_max = √(σ_л/(k_n A_лa)), q_n = n/n_max, σ_р = k_n n² A_лa.
4. Корневая реактивность: упрощённая формула α1=const
   ρ_к = 1 − (1−ρ_ср)(D_ср/D_вт)^(2cos²α1) — только для закона α1=const;
   для Γ=const она не обязана совпадать (проверяется отдельно).

Запуск в контейнере:
    python Turbinis-Calc/check_invariants.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Optimizer"))
import optimizer_core as oc  # noqa: E402

CFG = dict(mode="A", phi=0.97, psi=0.96, pi=11.0, T_g=1250.0, N=110000.0,
           lc1_target=0.020, eta_target=0.90, seed=1)

m = oc.load_model()
r = oc.evaluate(oc.warm_start(CFG), m, CFG, full=True)
sig_allow = r["p"]["sigma"]
k_n = 6.8e-6
print(f"σ_л = {sig_allow:.1f} МПа; n = {r['n_t']:.0f} об/мин; "
      f"l_с1 = {r['lc1']*1e3:.2f} мм")
print("ступень | η_u(L_u/H0) | 1−ξc'−ξл−ξв | η_т1 | η_u−ξз | q_n")
for i, s in enumerate(r["stages"], 1):
    e_loss = 1.0 - s["ξ_c_"] - s["ξ_л"] - s["ξ_в"]
    q = math.sqrt(max(s["σ_р"], 0.0) / sig_allow)
    print(f"{i} | {s['η_u']:.6f} | {e_loss:.6f} | {s['η_т1']:.6f} | "
          f"{s['η_u'] - s['ξ_з']:.6f} | {q:.5f}")
    assert abs(s["η_u"] - e_loss) < 1e-6, "η_u != 1-ξc'-ξл-ξв"
    assert abs(s["η_т1"] - (s["η_u"] - s["ξ_з"])) < 1e-6, "η_т1 != η_u-ξз"
    assert abs(s["η_т"] - (s["η_т1"] - s["ξ_тв"])) < 1e-6, "η_т != η_т1-ξтв"
    # σ_р = k_n n² A_лa
    sigma_calc = k_n * r["n_t"] ** 2 * s["A_лa"]
    assert abs(sigma_calc - s["σ_р"]) / max(s["σ_р"], 1e-9) < 1e-9, "σ_р formula"

print("\nКорневая реактивность: упрощённая α1=const vs полный ρ_корня")
for i, s in enumerate(r["stages"], 1):
    Dm = s["D_c"]
    Dh = Dm - s["l_c"]
    a = math.radians(s["α1"])
    rho_simple = 1.0 - (1.0 - s["ρ"]) * (Dm / Dh) ** (2.0 * math.cos(a) ** 2)
    print(f"ст.{i}: α1={s['α1']:.2f}°, ρ_ср={s['ρ']:.3f}, D_ср/D_вт={Dm/Dh:.3f}, "
          f"ρ_к(упрощ.)={rho_simple:.4f}, ρ_к(полн.)={r['rho_root'][i-1]:.4f}")
    if i <= 2:  # в проекте закон_ст = ['α1','α1','Γ']; ст.3 — Γ=const
        assert abs(rho_simple - r["rho_root"][i - 1]) < 0.05, \
            "упрощённая формула α1=const расходится с полным расчётом >0.05"

q_last = math.sqrt(r["sigma_max"] / sig_allow)
print(f"\nq_n (по σ_max) = {q_last:.5f}; σ_max/σ_л = {r['sigma_max']/sig_allow:.5f}")
assert abs(q_last - 0.995) < 0.01, "q_n не соответствует проектному 0.995"

print("\nВсе проверки пройдены.")
