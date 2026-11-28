"""Проверка полноты фазы 2: допустимые точки фазы 1 реально попадают в её старт.

Оценщик подменяется простой функцией от x, поэтому тест не зависит от модели.
"""
import numpy as np

import optimize_extended as oe

CALLS = {"p1": 0, "p2": 0}
FAKE_FEASIBLE = []  # x, признанные допустимыми в фазе 1


def fake_eval(x, ns, scenario, q_n_limit, soft=False):
    x = np.asarray(x, dtype=float)
    # допустимо: частота в ±5 % от WARM; hard — нормированный выход за эту полосу
    d = max(0.0, abs(x[0] - oe.WARM[0]) - 0.05 * oe.WARM[0]) / (0.05 * oe.WARM[0])
    feas = d <= 0.0
    r = dict(score=d, score1=d, score2=float(x[0]), feasible=feas, hard=d,
             hard_max=d, violation_count=0 if feas else 1, status="ok", x=x,
             n=x[0], D_hub=x[1], w1=x[2], w2=x[3], a1=tuple(x[4:7]), k=tuple(x[7:10]),
             lc1=x[1], a2=[float(v) for v in x[4:7]], sigma=float(x[0]), u=0.0,
             lD=float(x[1]), qn=0.0, N=110e3, eta_t=0.91, eta_ok=True, eta_below=0.0,
             uc0=[], uc0_opt=[], rho=[], rho_root=[], cont=0.0, energy=0.0, p2z=0.0,
             violations=[] if feas else ["fake:1"], err="")
    return r


def main() -> None:
    oe.eval_candidate = fake_eval
    oe.build_ns = lambda m, q_n_limit=0.995, sigma_lim=oe.SIGMA_LIM: ({}, None)
    # маленький бюджет: 60 особей достаточно, чтобы старт фазы 2 попал в оценку
    rows = oe.run_scenario(None, "A87_strict", seeds=2, maxiter=6, popsize=6, q_n_limit=0.995)
    ph1 = [r for r in rows if r["phase"] == 1 and r["feasible"]]
    ph2 = [r for r in rows if r["phase"] == 2]
    assert ph1, "фаза 1 не нашла допустимых — тест не показателен"
    assert ph2, "фаза 2 не запустилась при допустимых в фазе 1"
    assert any(r["feasible"] for r in ph2), "старт фазы 2 не содержит допустимых точек"
    # фаза 2 оптимизирует score2 (в фейке = x[0]), т.е. максимум l_c1
    best2 = min((r for r in ph2 if r["feasible"]), key=lambda r: r["score2"])
    best1 = min(ph1, key=lambda r: r["score2"])
    print(f"phase1 feasible={len(ph1)}, phase2 evals={len(ph2)}, "
          f"best1 x0={best1['x'][0]:.0f}, best2 x0={best2['x'][0]:.0f}")


if __name__ == "__main__":
    main()
