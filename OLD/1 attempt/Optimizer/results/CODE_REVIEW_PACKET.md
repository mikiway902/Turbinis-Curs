# Пакет для проверки optimize_extended.py

## Файлы

- `Optimizer/optimize_extended.py` — полный код (в zip).
- `Optimizer/optimizer_core.py` — модель, `evaluate`, `validate`; const-hub путь
  использует `m.code` и функции из блокнота, не `evaluate`.
- `Turbinis-Calc/constraints.py` — рис. 18/19, `DIGITIZATION`, PCHIP.
- `Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb`, ячейка `const-hub-scan` —
  `ступень_const_hub`, `турбина_const_hub`, `распределить_π_const_hub`.
- `Optimizer/results/extended_A87_strict_q0.995.csv`,
  `extended_B87_relaxed_q0.995.csv`.

## q_n / σ_л: где используются

`grep` по коду блокнота и `optimizer_core.py`:

- `q_n` и `σ_л` — в старом пути `решить` (ячейка 16) и в проверках блокнота;
- `ступень_const_hub` / `турбина_const_hub` их не читают;
- `optimizer_core.py`: `ns["q_n"]` (271), `σ_л` в `ns.update` (293),
  проверка `n < n_max (q_n)` (456–459) — это старый `evaluate`.

`build_ns` ставит `ns["σ_л"] = q_n²·300` (литерал 290 убран), т.е. он
совпадает с пределом residual: 297.0075 МПа при `q_n=0.995`.
Тест: один кандидат при `q_n=0.995, σ_л=290` и `q_n=1.0, σ_л=300` даёт
одинаковые `l_c1, α2, σ, l/D, N, η_t*`; меняется только предел residual.

## q_n по ступеням

```
n_max = min_i sqrt(300/(k_n A_лa,i))
q_n = n/n_max = max_i sqrt(σ_i/300)
```
В `eval_candidate`:
- `sig_all` по всем ступеням;
- `n_max_all` по всем `A_лa`;
- `sig_cons = max |σ_i − k_n n² A_лa,i| / σ_i` (порог 1e-9);
- `qn_sigma = max sqrt(σ_i/300)`; расхождение >1e-3 — нарушение.

## η_t* и фазы

`eta_t = Σ(G_c,i/G_c1 · L_т,i) / H_т*`, `H_т*` по `T*г,p*г` и `T*,p*`
за последней ступенью. `η_т_полн_цикл` не используется.
`score1 = hard`; `score2 = 1e6·hard + 1e6·η_below + (1−l_c1/0.020)` — только `l_c1`.
`u/c0` и `ρ_ср` — диагностика; `--soft` добавляет их отклонения как предпочтения.
Фаза 2 запускается только при наличии `feasible` в фазе 1.

## NaN и независимость

- `crit` — все критические величины; `np.isfinite` → иначе `status="numerical"`,
  `hard=1e9`, `feasible=False`.
- `G_base` фиксируется в `build_ns`; каждый кандидат стартует от него.
- Масштабирование G идёт до `|N/110 кВт − 1| < 1e-12` (до 20 шагов).
- Тест X→Y→X→X(fresh): `l_c1, α2, σ, l/D, q_n, N, η_t*` совпадают до 1e-8;
  повторная оценка `x` из CSV (`%.17g`) даёт `|Δ| = 0`.
- Фаза 2 стартует из **всех** допустимых точек фазы 1 (до размера популяции);
  `test_phase2.py` проверяет, что фаза 2 запускается и содержит допустимые.

## CSV

Колонки: `scenario, seed, phase, status, score1, score2, feasible, hard,
hard_max, violation_count, n, D_hub, w1, w2, a1, k, lc1, a2, sigma, u, lD,
qn, N, eta_t, eta_ok, eta_below, uc0, uc0_opt, rho, rho_root, cont, energy,
p2z, violations, err, x`.

Лучшие строки:

```
A87_strict  phase1 hard=0.289 count=10 l=11.41 alpha2=52.35/49.76/59.06
            sigma=307.9 lD=0.3174 qn=1.0132 eta_t=0.9016
B87_relaxed phase1 hard=0.139 count=5  l=12.54 alpha2=38.67/92.26/68.90
            sigma=269.2 lD=0.2681 qn=0.9473 eta_t=0.9062 rho=0.0442/0.0447
```

Контроль WARM (n=84k,D=65,w=.75/.80,α1_1=11):
`alpha2=78.4/73.8/80.8` (в книге), `eta_t=0.9131`,
`l=11.48`, `sigma=315.9`, `q=1.0262`, `rho_root=-0.0471/0.0040`, `hard=1.942`.
