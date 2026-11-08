"""Обратная оптимизация исходных параметров цикла.

Внешний уровень: π*к, T*г, N.
Внутренний уровень: геометрия и углы турбины (локальная доводка существующей модели).

Скрипт использует optimizer_core.py и тот же блокнот Main_Turbinis_Calc_Book.ipynb.
Никакие числа не подставляются вручную: каждое решение получено вызовом модели
и проверено validate().

Запуск (из папки Optimizer):
    python optimize_initial_conditions.py --mode 1 --quick
    python optimize_initial_conditions.py --mode 2
    python optimize_initial_conditions.py --mode 3 --inner-steps 2

Режимы:
    1 - минимальное отклонение цикла от исходного (π*к=11, T*г=1250, N=110 кВт);
    2 - максимальный КПД при выполнении всех ограничений;
    3 - максимальная технологичность (l_с1, умеренные u и σ, ровные длины).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import optimizer_core as oc

ПАПКА = Path(__file__).resolve().parent
РЕЗУЛЬТАТЫ = ПАПКА / "results"
РЕЗУЛЬТАТЫ.mkdir(exist_ok=True)

БАЗОВЫЙ_ЦИКЛ = dict(pi=11.0, T_g=1250.0, N=110000.0)
БАЗОВЫЕ_ДЕЛЬТЫ = dict(pi=5.0, T_g=150.0, N=350000.0)


def базовый_cfg(pi, tg, n, quick=True, inner_steps=1):
    return {
        "mode": "A", "warm": "book",
                "vary_kappa": True, "vary_sigma": True, "vary_tg": False,
        "phi": 0.965, "psi": 0.95,
        "pi": float(pi), "T_g": float(tg), "N": float(n),
        "lc1_target": 0.020, "eta_target": 0.90,
        "seed": 1, "quick": quick, "inner_steps": inner_steps,
    }


def внутренняя_доводка(m, cfg, steps=1, full=True):
    xw = oc.warm_start(cfg)
    x, sc, arch = oc.run_local(m, cfg, x0=xw, steps=steps, seed=cfg.get("seed", 1), full=False)
    res = oc.evaluate(x, m, cfg, full=full)
    chk = oc.validate(res, cfg, full=full)
    return x, res, chk, sc


def внешние_точки(quick=True):
    """Инженерные начальные точки: сетка вокруг границы l_с1=20 мм и N≈450...480 кВт."""
    if quick:
        return [
            (11.0, 1250.0, 462000.0),  # оценка по пределу l_с1 ~ sqrt(N)
            (11.0, 1250.0, 450000.0),
            (11.0, 1250.0, 480000.0),
            (10.0, 1250.0, 462000.0),
            (12.0, 1250.0, 462000.0),
            (11.0, 1300.0, 462000.0),
        ]
    pts = []
    for pi in (9.0, 10.0, 11.0, 12.0, 13.0):
        for tg in (1200.0, 1250.0, 1300.0):
            for n in (430000.0, 450000.0, 462000.0, 480000.0):
                pts.append((pi, tg, n))
    return pts


def отклонение(pi, tg, n):
    return (((pi - БАЗОВЫЙ_ЦИКЛ["pi"]) / БАЗОВЫЕ_ДЕЛЬТЫ["pi"]) ** 2
            + ((tg - БАЗОВЫЙ_ЦИКЛ["T_g"]) / БАЗОВЫЕ_ДЕЛЬТЫ["T_g"]) ** 2
            + ((n - БАЗОВЫЙ_ЦИКЛ["N"]) / БАЗОВЫЕ_ДЕЛЬТЫ["N"]) ** 2)


def выбрать(rows, mode):
    допустимые = [r for r in rows if r["strict"]]
    if not допустимые:
        return None
    if mode == 1:
        return min(допустимые, key=lambda r: (r["deviation"], -r["lc1"]))
    if mode == 2:
        return max(допустимые, key=lambda r: (r["eta"], -r["deviation"]))
    # mode 3: технологичность: длина, умеренные u и σ, ровные длины
    def tech(r):
        return (r["lc1"] / 0.020, -r["u_t"] / 400.0, -r["sigma_max"] / 400.0)
    return max(допустимые, key=lambda r: tech(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--quick", action="store_true", help="6 внешних точек вместо 60")
    ap.add_argument("--inner-steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    m = oc.load_model()
    rows = []
    print("Внешних точек:", len(внешние_точки(args.quick)))
    for pi, tg, n in внешние_точки(args.quick):
        cfg = базовый_cfg(pi, tg, n, quick=args.quick, inner_steps=args.inner_steps)
        cfg["seed"] = args.seed
        try:
            x, res, chk, sc = внутренняя_доводка(m, cfg, steps=args.inner_steps, full=True)
        except Exception as e:
            print(f"  π={pi} T*г={tg:.0f} N={n/1e3:.0f}: ошибка {type(e).__name__}: {e}")
            continue
        strict = oc.is_strict(chk)
        rows.append(dict(
            pi=pi, Tg=tg, N=n / 1e3, lc1=res["lc1"] * 1e3, eta=res["eta"],
            a2=res["a2"], pi_st=res["pi"], u_t=res["u_t"], sigma_max=res["sigma_max"],
            D_t=res["D_t"] * 1e3, G=res["cycle"]["G"],
            strict=strict, score=sc, deviation=отклонение(pi, tg, n),
            x=x.tolist(), res=res, checks=chk,
        ))
        print(f"  π={pi:.0f} T*г={tg:.0f} N={n/1e3:.0f}: l_с1={res['lc1']*1e3:.2f} мм, "
              f"α2={[round(v,1) for v in res['a2']]}, η*т={res['eta']:.4f}, "
              f"{'ДОПУСТИМ' if strict else 'недопустим'}")

    df = pd.DataFrame([{k: r[k] for k in ("pi", "Tg", "N", "lc1", "eta", "u_t", "sigma_max",
                                          "D_t", "G", "strict", "score", "deviation")} for r in rows])
    if not df.empty:
        df.to_csv(РЕЗУЛЬТАТЫ / "initial_conditions_scan.csv", index=False)

    best = выбрать(rows, args.mode)
    print("\n=== РЕЗУЛЬТАТ ===")
    if best is None:
        print("Строго допустимых вариантов не найдено.")
        if rows:
            b = min(rows, key=lambda r: r["score"])
            print("Лучший по штрафу:", {k: b[k] for k in ("pi", "Tg", "N", "lc1", "eta")})
            ок = [c["name"] for c in b["checks"] if not c["ok"] and c["kind"] in ("hard", "project")]
            print("Нарушено:", ок)
        return

    print(f"Режим {args.mode}: π*к={best['pi']:.2f}, T*г={best['Tg']:.0f} K, "
          f"N={best['N']:.0f} кВт, G={best['G']:.3f} кг/с")
    print(f"l_с1={best['lc1']:.2f} мм, α2={[round(v,1) for v in best['a2']]}, "
          f"π={[round(v,2) for v in best['pi_st']]}, η*т={best['eta']:.4f}, "
          f"σ_max={best['sigma_max']:.0f} МПа, u_т={best['u_t']:.0f} м/с")
    print("Отклонение от исходного цикла:", round(best["deviation"], 5))
    print("Статус:", "строго допустим" if best["strict"] else "недопустим")

    данные = {
        "mode": args.mode,
        "baseline_cycle": БАЗОВЫЙ_ЦИКЛ,
        "optimized_cycle": {"pi": best["pi"], "T_g": best["Tg"], "N": best["N"] * 1e3},
        "x": best["x"],
        "result": {k: best["res"][k] for k in
                   ("lc1", "a2", "pi", "N_val", "eta", "sigma_max", "D_t", "n_t", "gamma_max",
                    "rho_root", "lambda_c1")},
        "checks": best["checks"],
    }
    oc.save_json(РЕЗУЛЬТАТЫ / f"initial_conditions_mode{args.mode}.json", данные)
    oc.save_json(РЕЗУЛЬТАТЫ / "initial_conditions.json", данные)
    pd.DataFrame(best["checks"]).to_csv(РЕЗУЛЬТАТЫ / f"initial_conditions_mode{args.mode}_checks.csv", index=False)
    pd.DataFrame(best["checks"]).to_csv(РЕЗУЛЬТАТЫ / "initial_conditions_checks.csv", index=False)
    print("Сохранено:", РЕЗУЛЬТАТЫ / f"initial_conditions_mode{args.mode}.json")


if __name__ == "__main__":
    main()
