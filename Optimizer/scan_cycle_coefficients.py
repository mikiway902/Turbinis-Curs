"""Скан коэффициентов цикла при π*к = 11, T*г = 1250 К, N = 110 кВт.

Отвечает на вопрос: можно ли, ухудшив КПД/потери цикла, получить l_с1 >= 20 мм
при сохранении π*к = 11 и N = 110 кВт.

Внешний параметр — η_к (политропный КПД компрессора). Остальные коэффициенты
берутся по нижним (уже частично внекнижным) границам, чтобы дать длине максимум:
    η_т = 0.87, η_м = 0.96, η_р = 0.93,
    σ_вх = 0.95, σ_г = 0.93, π_с = 1.10.
Для каждого η_к выполняется локальная доводка геометрии (strict α2, l_с1).

Запуск:
    python scan_cycle_coefficients.py
    python scan_cycle_coefficients.py --full-range
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import optimizer_core as oc

ПАПКА = Path(__file__).resolve().parent
РЕЗУЛЬТАТЫ = ПАПКА / "results"
РЕЗУЛЬТАТЫ.mkdir(exist_ok=True)

НИЖНИЕ = dict(η_т_п_полн=0.87, η_м_тур=0.96, η_р=0.93,
              σ_вх=0.95, σ_г=0.93, π_с=1.10)


def оценить(m, eta_k, eta_t=0.87):
    C = m.cycle
    C["η_к_п_полн"] = float(eta_k)
    C["η_т_п_полн"] = float(eta_t)
    C["η_м_тур"] = НИЖНИЕ["η_м_тур"]
    C["η_р"] = НИЖНИЕ["η_р"]
    C["σ_вх"] = НИЖНИЕ["σ_вх"]
    C["σ_г"] = НИЖНИЕ["σ_г"]
    C["π_с"] = НИЖНИЕ["π_с"]
    m.base["η_м"] = НИЖНИЕ["η_м_тур"]
    m.base["η_р"] = НИЖНИЕ["η_р"]

    cfg = {"mode": "A", "warm": "book",            "vary_kappa": True, "vary_sigma": True, "phi": 0.965, "psi": 0.95,
           "pi": 11.0, "T_g": 1250.0, "N": 110000.0, "lc1_target": 0.020,
           "eta_target": 0.90, "seed": 1}
    x0 = oc.warm_start(cfg)
    r0 = oc.evaluate(x0, m, cfg, full=True)
    x, sc, _ = oc.run_local(m, cfg, x0=x0, steps=1, seed=1, full=False)
    r = oc.evaluate(x, m, cfg, full=True)
    ch = oc.validate(r, cfg, full=True)
    return dict(eta_k=eta_k, eta_t=eta_t, G=r["cycle"]["G"], lc1=r["lc1"] * 1e3,
                a2=r["a2"], pi=r["pi"], eta_e=r["cycle"]["eta_e"], C_e=r["cycle"]["C_e"],
                lc1_book=r0["lc1"] * 1e3, strict=oc.is_strict(ch),
                fail=[c["name"] for c in ch if not c["ok"] and c["kind"] in ("hard", "project")])


def main():
    full = "--full-range" in sys.argv
    m = oc.load_model()
    etas = np.round(np.arange(0.85, 0.689, -0.01), 2) if full else np.round(np.arange(0.85, 0.699, -0.02), 2)
    rows = []
    print("η_к    G, кг/с   l_с1, мм   α2 (1/2/3)          η_e      C_e, кг/(кВт·ч)  статус")
    for ek in etas:
        r = оценить(m, float(ek))
        rows.append(r)
        print(f"{r['eta_k']:.2f}  {r['G']:7.3f}  {r['lc1']:7.2f}   "
              f"{'/'.join(f'{v:.0f}' for v in r['a2']):18s} {r['eta_e']:.4f}  {r['C_e']:10.3f}      "
              f"{'OK' if r['strict'] else 'нет'}")
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "fail"} for r in rows])
    df.to_csv(РЕЗУЛЬТАТЫ / "cycle_coeff_scan.csv", index=False)
    print("Сохранено:", РЕЗУЛЬТАТЫ / "cycle_coeff_scan.csv")
    ok = [r for r in rows if r["strict"]]
    if ok:
        best = min(ok, key=lambda r: r["eta_k"])  # минимальное ухудшение η_к
        print(f"\nПервый строго допустимый: η_к = {best['eta_k']:.2f}, "
              f"l_с1 = {best['lc1']:.2f} мм, η_e = {best['eta_e']:.4f}, "
              f"C_e = {best['C_e']:.3f} кг/(кВт·ч)")
        print("Все проверки:", "пройдены" if not best["fail"] else best["fail"])
    else:
        print("\nВ этом диапазоне строго допустимых нет.")


if __name__ == "__main__":
    main()
