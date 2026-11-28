#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сверка: Mathcad «С_учетом_охлаждения_многоступенч.xmcd» (этапы 1-5, 8)
        vs Turbinis-Cycle.ipynb (ячейки 2-8)
Оба считаются заново на Python; проверка - по сохранённым в файлах числам.
"""
import math

# ============================================================
# 1. MATHCAD-МОДЕЛЬ  (цикл: этапы 3-5, 8)
# ============================================================
def mc_cp_true_air(T):
    """C_p(T) - истинная теплоёмкость воздуха, Дж/(кг*К)"""
    if T < 750:
        return (0.2407 + 0.0193 * (2.5e-3 * T - 0.875)
                + 2.0e-3 * (2.5e-5 * T**2 - 0.0275 * T + 6.5625)) * 4187.0
    return (0.26 + 0.032 * (1.176e-3 * T - 0.88235)
            + (-0.374e-2) * (5.5556e-6 * T**2 - 1.3056e-2 * T + 6.67)) * 4187.0


def mc_cpm_air(T):
    """C_pm.в(T) - средняя теплоёмкость воздуха"""
    if T < 700:
        return (1.2e-5 * (T - 70.0) + 0.236) * 4187.0
    return (2.2e-5 * (T + 450.0) + 0.218) * 4187.0


def mc_cp_true_gas(alpha, T):
    """C_p.г(α,T) - истинная теплоёмкость продуктов сгорания"""
    if T < 750:
        return (0.0174 / alpha + 0.2407
                + (0.0093 / alpha + 0.0193) * (2.5e-3 * T - 0.875)
                + (2.0e-3 - 1.056e-3 / (alpha - 0.2))
                * (2.5e-5 * T**2 - 0.0275 * T + 6.5625)) * 4187.0
    return (0.0267 / alpha + 0.26
            + (0.0133 / alpha + 0.032) * (1.176e-3 * T - 0.88235)
            + (-1) * (0.94e-2 / (alpha**2 + 10) + 0.374e-2)
            * (5.5556e-6 * T**2 - 1.3056e-2 * T + 6.67)) * 4187.0


def mc_cpm_gas(alpha, T):
    """C_pm(α,T) - средняя теплоёмкость продуктов сгорания"""
    if T < 700:
        return ((2.25 + 1.2 * alpha) / (alpha * 1e5) * (T - 70.0) + 0.236) * 4187.0
    return ((1.25 + 2.2 * alpha) / (alpha * 1e5) * (T + 450.0) + 0.218) * 4187.0


MC = dict(
    N=750000.0, T_sm=1440.0, n_t=30000.0,
    g_ohl=0.05, g_ut=0.007,
    sig_vh=0.98, eta_k_pol=0.83, eta_g=0.995, sig_g=0.97,
    eta_t=0.9, eta_st=0.9, eta_m=0.995, eta_r=0.99, sig_vyh=0.995,
    Qn=43000e3, l0=14.61, T_a=288.0, p_a=101300.0, k=1.4, R=287.4,
    g_ohl_used=0.0, g_voz=0.0, pi_s=None,   # g_ohl_used=0 -> как в файле (без охлаждения)
)


def mathcad_cycle(pi_k, T_sm=1440.0, N=750000.0, **kw):
    """Цикл по Mathcad-файлу. Возвращает словарь параметров."""
    P = dict(MC)
    P.update(kw)
    T_a, p_a, R = P['T_a'], P['p_a'], P['R']
    eta_k_pol, eta_g, sig_g = P['eta_k_pol'], P['eta_g'], P['sig_g']
    sig_vh = P['sig_vh']
    g_ut, l0, Qn = P['g_ut'], P['l0'], P['Qn']

    # ---- этап 3: компрессор (3 приближения, k по C_p истинной) ----
    k = P['k']
    eta_k = (pi_k**((k - 1) / k) - 1) / (pi_k**((k - 1) / (k * eta_k_pol)) - 1)
    for _ in range(3):
        T_k = T_a * (1 + (pi_k**((k - 1) / k) - 1) / eta_k)
        cp_v = mc_cp_true_air(0.5 * (T_k + T_a))
        k = cp_v / (cp_v - R)
        eta_k = (pi_k**((k - 1) / k) - 1) / (pi_k**((k - 1) / (k * eta_k_pol)) - 1)
    T_k = T_a * (1 + (pi_k**((k - 1) / k) - 1) / eta_k)
    cp_v = mc_cp_true_air(0.5 * (T_k + T_a))
    p_k = sig_vh * p_a * pi_k
    L_k = cp_v * (T_k - T_a)

    # ---- этап 4: камера сгорания (3 приближения f) ----
    f = 0.025
    alpha_sm = 1.0 / (f * l0)
    for _ in range(3):
        h_sm = mc_cpm_gas(alpha_sm, T_sm) * T_sm - mc_cpm_gas(alpha_sm, 288) * 288
        h_vk = mc_cpm_air(T_k) * T_k - mc_cpm_air(288) * 288
        f = (1 - g_ut) * (h_sm - h_vk) / (Qn * eta_g - h_sm)
        alpha_sm = (1 - g_ut) / (f * l0)
        h_sm = mc_cpm_gas(alpha_sm, T_sm) * T_sm - mc_cpm_gas(alpha_sm, 288) * 288
    g_m = f / (1 - g_ut)
    a_S = alpha_sm
    k_g = mc_cp_true_gas(a_S, T_sm) / (mc_cp_true_gas(a_S, T_sm) - R)

    # ---- охлаждение (в файле этого блока нет: g_охл объявлен и не используется) ----
    g_ohl, g_voz = P['g_ohl_used'], P['g_voz']
    g_core = (1 + f) * (1 - g_ut - g_ohl)     # расход через ТГГ / расход воздуха
    g_sum = g_core + g_voz                    # расход через СТ / расход воздуха

    # ---- этап 4-5: ТГГ (3 приближения, как в xmcd) ----
    L_t_gg = L_k / g_core / P['eta_m']
    p_g = sig_g * p_k
    cp_g_sr = mc_cp_true_gas(a_S, T_sm)
    k_g_sr = cp_g_sr / (cp_g_sr - R)
    p_tv = p_g
    T_t_gg = T_sm
    for _ in range(3):
        p_tv = p_g * (1 - L_t_gg / (cp_g_sr * T_sm * P['eta_t'])) ** (k_g_sr / (k_g_sr - 1))
        T_t_gg = T_sm * (1 - (1 - (p_tv / p_g) ** ((k_g_sr - 1) / k_g_sr)) * P['eta_t'])
        T_sr = 0.5 * (T_t_gg + T_sm)
        cp_g_sr = mc_cp_true_gas(a_S, T_sr)
        k_g_sr = cp_g_sr / (cp_g_sr - R)
        T_t_gg = T_sm - L_t_gg / cp_g_sr
    T_sr = 0.5 * (T_t_gg + T_sm)
    cp_g_sr = mc_cp_true_gas(a_S, T_sr)
    k_g_sr = cp_g_sr / (cp_g_sr - R)
    pi_t_gg = p_g / p_tv

    # ---- этап 5: свободная турбина ----
    p_t_s = P['pi_s'] * p_a if P['pi_s'] else p_a / P['sig_vyh']
    pi_t_c = p_tv / p_t_s
    eta_l = P.get('eta_l', 0.91)
    c_t = P.get('c_t', 210.0)
    for _ in range(3):
        L_t_s = (cp_g_sr * T_t_gg * (1 - (p_t_s / p_tv) ** ((k_g_sr - 1) / k_g_sr))
                 * eta_l - c_t**2 / 2)
        T_t_s = T_t_gg * (1 - (1 - (p_t_s / p_tv) ** ((k_g_sr - 1) / k_g_sr)) * eta_l)
        T_sr = 0.5 * (T_t_gg + T_t_s)
        cp_g_sr = mc_cp_true_gas(a_S, T_sr)
        k_g_sr = cp_g_sr / (cp_g_sr - R)
    L_t_s = (cp_g_sr * T_t_gg * (1 - (p_t_s / p_tv) ** ((k_g_sr - 1) / k_g_sr))
             * eta_l - c_t**2 / 2)
    T_t_s = T_t_gg * (1 - (1 - (p_t_s / p_tv) ** ((k_g_sr - 1) / k_g_sr)) * eta_l)

    L_val = L_t_s * P['eta_r'] * P['eta_m']
    if P['g_ohl_used'] or P['g_voz']:
        G_v = N / (L_val * g_sum)             # как в Python: на 1 кг воздуха
    else:
        G_v = N / (L_val * ((1 - g_ut) + f))
    G_t = f * G_v
    C_e = G_t * 3600 * 1000 / N
    eta_e = 3600 * 1000 / (C_e * Qn)

    return dict(pi_k=pi_k, T_k=T_k, k_sz=k, cp_v=cp_v, L_k=L_k, p_k=p_k,
                f=f, g_m=g_m, alpha=a_S, k_g=k_g, L_t_gg=L_t_gg, p_g=p_g,
                p_tv=p_tv, pi_t_gg=pi_t_gg, T_t_gg=T_t_gg, pi_t_c=pi_t_c,
                L_t_s=L_t_s, T_t_s=T_t_s, cp_g_sr=cp_g_sr, k_g_sr=k_g_sr,
                L_val=L_val, G_v=G_v, G_t=G_t, C_e=C_e, eta_e=eta_e,
                G_gas=G_v * g_sum, g_sum=g_sum, g_core=g_core, g_ohl=g_ohl,
                N_ud=L_val * g_sum)


# ============================================================
# 2. PYTHON-МОДЕЛЬ (Turbinis-Cycle.ipynb, ячейки 2-5)
# ============================================================
PY = dict(
    T_a=288.0, p_a=0.1013e6, R=287.4, k_v=1.4, T_g=1250.0,
    sig_vh=0.985, eta_k_pol=0.86, eta_t_pol=0.905, eta_g=0.98, sig_g=0.965,
    eta_m=0.9925, eta_r=0.985, sig_vyh=0.995, Qn=43150.0, l0=14.61,
    g_ut=0.0075, pi_s=1.04, M_t=0.2031, phi_c=0.95, beta_v=0.015,
    T_0=288.0,
)


CP_MODE = 'mean'   # 'mean' - как в ноутбуке, 'true' - как в Mathcad


def cp_py(T, alpha=math.inf):
    T = float(T)
    if CP_MODE == 'true':
        return mc_cp_true_air(T) if alpha == math.inf else mc_cp_true_gas(alpha, T)
    if T < 700.0:
        return ((1.2 + 2.25 / alpha) / 1e5 * (T - 70.0) + 0.236) * 4187.0
    return ((2.2 + 1.25 / alpha) / 1e5 * (T + 450.0) + 0.218) * 4187.0


def k_py(T, alpha=math.inf):
    cp = cp_py(T, alpha)
    return cp / (cp - 287.4)


def calc_alpha_poly(T_k, T_g):
    x, y = T_k / 1000.0, T_g / 1000.0
    return 1.0 / (-0.004149 + 0.367730 * y - 0.011584 * y**2 + 0.027054 * y**3
                  - 0.398434 * x + 0.031502 * x * y - 0.022030 * x * y**2
                  + 0.062033 * x**2 - 0.003988 * x**2 * y - 0.049657 * x**3)


def python_cycle(pi_k, T_g=None, N=110e3, **kw):
    P = dict(PY)
    P.update(kw)
    if T_g is None:
        T_g = P['T_g']
    else:
        P['T_g'] = T_g
    R, eta_k_pol, eta_t_pol = P['R'], P['eta_k_pol'], P['eta_t_pol']
    T_a, p_a = P['T_a'], P['p_a']
    T_a_poln, pi_v = T_a, 1.0                     # стенд
    cp_v = P['k_v'] * R / (P['k_v'] - 1)

    # --- компрессор ---
    T_k = T_a_poln * pi_k ** ((P['k_v'] - 1) / (P['k_v'] * eta_k_pol))
    for _ in range(100):
        kk = k_py(0.5 * (T_a_poln + T_k))
        T_new = T_a_poln * pi_k ** ((kk - 1) / (kk * eta_k_pol))
        if abs(T_new - T_k) < 1e-6:
            T_k = T_new
            break
        T_k = T_new
    k_sz = k_py(0.5 * (T_a_poln + T_k))
    L_k = R * k_sz / (k_sz - 1) * (T_k - T_a_poln)
    alpha_pol = calc_alpha_poly(T_k, T_g)

    # --- g_m (формула стр. 20 методички) ---
    T_0, l0, Qn = P['T_0'], P['l0'], P['Qn'] * 1e3
    числ = cp_py(T_g) * T_g - cp_py(T_k) * T_k
    знам = (Qn * P['eta_g']
            - (1 + l0) * (cp_py(T_g, 1.0) * T_g - cp_py(T_0, 1.0) * T_0)
            + l0 * (cp_py(T_g) * T_g - cp_py(T_0) * T_0))
    g_m = числ / знам
    alpha_g = 1.0 / (g_m * l0)

    # --- охлаждение ---
    g_ohl = kw.get('g_ohl', 0.05 if T_g > 1250 else 0.015)
    g_voz = 0.6 * g_ohl
    g_t_otn = (1 + g_m) * (1 - P['g_ut'] - g_ohl)

    # --- турбина (итерация pi_t) ---
    pi_t = P['sig_vh'] * P['sig_g'] * pi_k * pi_v / P['pi_s']
    for _ in range(100):
        T_sr = 0.5 * T_g * (1 + pi_t ** -0.25)
        cp_t = cp_py(T_sr, alpha_pol)
        k_t = cp_t / (cp_t - R)
        pi_s_poln = P['pi_s'] * (1 + (k_t - 1) / 2 * P['M_t']**2) ** (k_t / (k_t - 1))
        pi_new = P['sig_vh'] * P['sig_g'] * pi_k * pi_v / pi_s_poln
        if abs(pi_new - pi_t) < 1e-6:
            pi_t = pi_new
            break
        pi_t = pi_new
    eta_t = (1 - pi_t ** ((1 - k_t) * eta_t_pol / k_t)) / (1 - pi_t ** ((1 - k_t) / k_t))
    if 'eta_t_override' in kw:
        eta_t = kw['eta_t_override']
    L_t = cp_t * T_g * (1 - pi_t ** ((1 - k_t) / k_t)) * eta_t
    T_t = T_g - L_t / cp_t

    # --- сопло ---
    T_sr_s = 0.5 * T_t * (1 + pi_s_poln ** -0.25)
    cp_s = cp_py(T_sr_s, alpha_pol)
    k_s = cp_s / (cp_s - R)
    H_s = cp_s * T_t * (1 - pi_s_poln ** ((1 - k_s) / k_s))
    c_s = P['phi_c'] * math.sqrt(2 * H_s)

    g_sum = g_t_otn + g_voz
    N_ud = (L_t * g_sum * P['eta_m'] - L_k) * P['eta_r']
    R_ud = g_sum * (c_s - 0.0)
    N_e_ud = N_ud + R_ud / P['beta_v']
    C_e = g_m * (1 - P['g_ut'] - g_ohl) * 3600 / (N_ud / 1e3)
    eta_e = 3600 / (C_e * P['Qn'])
    G_v = N / N_ud

    return dict(pi_k=pi_k, T_k=T_k, k_sz=k_sz, L_k=L_k, g_m=g_m, alpha=alpha_g,
                alpha_pol=alpha_pol, g_ohl=g_ohl, g_t=g_t_otn, pi_t=pi_t,
                eta_t=eta_t, cp_t=cp_t, k_t=k_t, L_t=L_t, T_t=T_t, pi_s_poln=pi_s_poln,
                c_s=c_s, N_ud=N_ud, N_e_ud=N_e_ud, R_ud=R_ud, C_e=C_e, eta_e=eta_e,
                G_v=G_v)


def main():
    print("=" * 78)
    print("ПРОВЕРКА 1: Mathcad-модель против сохранённых чисел xmcd (π=7..11)")
    print("=" * 78)
    ref = {7: dict(T_k=558.42368, L_k=275181.17, f=0.0251518, alpha=2.70228,
                   pi_t_gg=2.221476, T_t_gg=1222.89344, pi_t_c=2.98042,
                   L_t_s=285603.65, T_t_s=968.66152, L_val=281333.87,
                   G_v=2.6183440, C_e=0.3161090, eta_e=0.2648483),
           10: dict(T_k=627.87905, L_k=347702.19, f=0.02332905, alpha=2.91341,
                    pi_t_gg=2.8247523, T_t_gg=1163.26648, pi_t_c=3.3484245,
                    L_t_s=296930.93, T_t_s=895.88967, L_val=292491.81,
                    G_v=2.5229765, C_e=0.2825215, eta_e=0.2963347),
           11: dict(T_k=647.63074, L_k=368496.47, f=0.02280854, alpha=2.97990,
                    pi_t_gg=3.0348327, T_t_gg=1145.97082, pi_t_c=3.4283000,
                    L_t_s=297271.83, T_t_s=877.22487, L_val=292827.62,
                    G_v=2.5213746, C_e=0.2760425, eta_e=0.3032899)}
    print(f"{'π':>3} {'величина':>12} {'Mathcad':>12} {'Python-порт':>12} {'Δ, %':>8}")
    for pi, r in ref.items():
        d = mathcad_cycle(pi)
        for key, val in r.items():
            got = d[key]
            print(f"{pi:>3} {key:>12} {val:>12.6g} {got:>12.6g} {100*(got-val)/val:>8.2f}")
    print()
    print("=" * 78)
    print("ПРОВЕРКА 2: Python-порт против выходов ноутбука (π=11, T*г=1250)")
    print("=" * 78)
    d = python_cycle(11.0)
    print(f"  π*т    = {d['pi_t']:.4f}   (ноутбук 9.77)")
    print(f"  T*т    = {d['T_t']:.2f} К  (ноутбук 715.6)")
    print(f"  L_т    = {d['L_t']/1e3:.1f} кДж/кг (ноутбук 568.1)")
    print(f"  η*т    = {d['eta_t']:.4f}   (ноутбук ~0.929)")
    print(f"  N_в.уд = {d['N_ud']/1e3:.1f} кВт·с/кг (ноутбук 210.0)")
    print(f"  N_e.уд = {d['N_e_ud']/1e3:.1f} кВт·с/кг (ноутбук 220.6)")
    print(f"  G_в    = {d['G_v']:.4f} кг/с (ноутбук 0.524)")
    print(f"  C_e    = {d['C_e']:.4f} кг/(кВт·ч) (ноутбук 0.291)")
    print(f"  η_e    = {d['eta_e']:.4f}  (ноутбук 0.286)")
    print(f"  g_m    = {d['g_m']:.5f}   α = {d['alpha']:.3f}")
    # сетка π_к: ноутбук даёт min G_в при 8.29, min C_e при 16.86
    grid = [4 + (25 - 4) * i / 49 for i in range(50)]
    res = [python_cycle(pi) for pi in grid]
    iG = min(range(50), key=lambda i: res[i]['G_v'])
    iC = min(range(50), key=lambda i: res[i]['C_e'])
    print(f"  min G_в: π*к = {grid[iG]:.2f}, G_в = {res[iG]['G_v']:.3f}  (ноутбук 8.29 / 0.513)")
    print(f"  min C_e: π*к = {grid[iC]:.2f}, C_e = {res[iC]['C_e']:.3f}  (ноутбук 16.86 / 0.281)")


if __name__ == '__main__':
    main()
