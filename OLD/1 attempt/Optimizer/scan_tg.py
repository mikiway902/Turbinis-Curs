"""Скан T*г при π*к = 11 и N = 110 кВт.

Для каждого T*г берётся книжная геометрия (D1=0.915, D2=0.97, lD=0.26,
α1=11.2/13.8/19.5°, κ=1.12/1.04/1.02, σ=285 МПа) и выполняется локальная
доводка геометрии. Показывается, что даёт изменение T*г по длине, α2, КПД
и экономичности. Все значения — фактический расчёт существующей модели.

Запуск:
    python scan_tg.py
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


def оценить(m, tg):
    cfg = {"mode": "A", "warm": "book",            "vary_kappa": True, "vary_sigma": True, "phi": 0.965, "psi": 0.95,
           "pi": 11.0, "T_g": float(tg), "N": 110000.0, "lc1_target": 0.020,
           "eta_target": 0.90, "seed": 1}
    x0 = oc.warm_start(cfg)
    r0 = oc.evaluate(x0, m, cfg, full=True)
    x, sc, _ = oc.run_local(m, cfg, x0=x0, steps=1, seed=1, full=False)
    r = oc.evaluate(x, m, cfg, full=True)
    ch = oc.validate(r, cfg, full=True)
    return dict(Tg=tg, G=r["cycle"]["G"], lc1=r["lc1"] * 1e3,
                a2=r["a2"], pi=r["pi"], eta=r["eta"],
                eta_e=r["cycle"]["eta_e"], C_e=r["cycle"]["C_e"],
                Tw1=max(s["T_w1"] for s in r["stages"]),
                охл=r["cycle"]["охлаждение"], strict=oc.is_strict(ch),
                fail=", ".join(c["name"] for c in ch if not c["ok"] and c["kind"] in ("hard", "project")))


def main():
    m = oc.load_model()
    rows = []
    print("T*г, К  G, кг/с  l_с1, мм  α2 (1/2/3)          η*т     η_e     C_e    T_w1,max  охл  статус")
    for tg in (1100, 1150, 1200, 1225, 1250, 1275, 1300, 1350, 1400, 1450):
        r = оценить(m, tg)
        rows.append(r)
        print(f"{tg:6d}  {r['G']:7.3f}  {r['lc1']:8.2f}  "
              f"{'/'.join(f'{v:.0f}' for v in r['a2']):18s} {r['eta']:.4f}  {r['eta_e']:.4f}  "
              f"{r['C_e']:.3f}  {r['Tw1']:8.0f}  {'да' if r['охл'] else 'нет':3s}  "
              f"{'OK' if r['strict'] else 'нет'}")
    pd.DataFrame([{k: v for k, v in r.items()} for r in rows]).to_csv(
        РЕЗУЛЬТАТЫ / "tg_scan.csv", index=False)
    print("Сохранено:", РЕЗУЛЬТАТЫ / "tg_scan.csv")
    ok = [r for r in rows if r["strict"]]
    if ok:
        b = min(ok, key=lambda r: abs(r["Tg"] - 1250.0))
        print(f"\nБлижайший к 1250 К строго допустимый: T*г = {b['Tg']} К, "
              f"l_с1 = {b['lc1']:.2f} мм, η_e = {b['eta_e']:.4f}, C_e = {b['C_e']:.3f}")
    else:
        print("\nСтрого допустимых во всём диапазоне нет.")


if __name__ == "__main__":
    main()
