#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Что будет, если в цикле считать c_p так же, как в Turbinis-Calc (истинная c_p
из той же корреляции: c_p = c_pm + (T-273.15) dc_pm/dT, энтальпия h = c_pm*(T-273.15))."""
import math
import compare as C

R = 287.4


def cpm(T, a=math.inf):
    return C.cp_py(T, a)


def cp_true(T, a=math.inf):
    dT = 1.0
    m = cpm(T, a)
    return m + (T - 273.15) * (cpm(T + dT, a) - cpm(T - dT, a)) / (2 * dT)


def h(T, a=math.inf):
    """энтальпия по средней c_p: h = c_pm(T)*(T-273.15)"""
    return cpm(T, a) * (T - 273.15)


def cycle_fixed(pi_k, T_g=1250.0, N=110e3, eta_k_pol=0.86, eta_t_pol=0.905,
                g_ut=0.0075, g_ohl=None, sig_vh=0.985, sig_g=0.965, eta_g=0.98,
                eta_m=0.9925, eta_r=0.985, Qn=43150.0, pi_s=1.04, M_t=0.2031,
                l0=14.61, T_a=288.0, p_a=0.1013e6, T_0=288.0):
    if g_ohl is None:
        g_ohl = 0.05 if T_g > 1250 else 0.015
    g_voz = 0.6 * g_ohl
    # --- компрессор ---
    T_k = T_a * pi_k ** ((1.4 - 1) / (1.4 * eta_k_pol))
    for _ in range(100):
        Tm = 0.5 * (T_a + T_k)
        k = cp_true(Tm) / (cp_true(Tm) - R)
        T_new = T_a * pi_k ** ((k - 1) / (k * eta_k_pol))
        if abs(T_new - T_k) < 1e-7:
            T_k = T_new
            break
        T_k = T_new
    Tm = 0.5 * (T_a + T_k)
    k_sz = cp_true(Tm) / (cp_true(Tm) - R)
    L_k = h(T_k) - h(T_a)                      # точная разность энтальпий
    alpha_pol = C.calc_alpha_poly(T_k, T_g)

    # --- камера сгорания (h через c_pm, как в книге) ---
    числ = h(T_g) - h(T_k)
    знам = (Qn * 1e3 * eta_g - (1 + l0) * (h(T_g, 1.0) - h(T_0, 1.0))
            + l0 * (h(T_g) - h(T_0)))
    g_m = числ / знам
    g_t = (1 + g_m) * (1 - g_ut - g_ohl)

    # --- турбина ---
    pi_t = sig_vh * sig_g * pi_k / pi_s
    for _ in range(100):
        T_sr = 0.5 * T_g * (1 + pi_t ** -0.25)
        cp_t = cp_true(T_sr, alpha_pol)
        k_t = cp_t / (cp_t - R)
        pi_s_poln = pi_s * (1 + (k_t - 1) / 2 * M_t**2) ** (k_t / (k_t - 1))
        pi_new = sig_vh * sig_g * pi_k / pi_s_poln
        if abs(pi_new - pi_t) < 1e-6:
            pi_t = pi_new
            break
        pi_t = pi_new
    eta_t = (1 - pi_t ** ((1 - k_t) * eta_t_pol / k_t)) / (1 - pi_t ** ((1 - k_t) / k_t))
    T_is = T_g * pi_t ** ((1 - k_t) / k_t)
    L_t = cp_t * (T_g - T_is) * eta_t
    T_t = T_g - L_t / cp_t

    g_sum = g_t + g_voz
    N_ud = (L_t * g_sum * eta_m - L_k) * eta_r
    C_e = g_m * (1 - g_ut - g_ohl) * 3600 / (N_ud / 1e3)
    return dict(T_k=T_k, L_k=L_k, g_m=g_m, alpha=1 / (g_m * l0), alpha_pol=alpha_pol,
                pi_t=pi_t, cp_t=cp_t, k_t=k_t, eta_t=eta_t, L_t=L_t, T_t=T_t,
                N_ud=N_ud, C_e=C_e, eta_e=3600 / (C_e * Qn),
                G_v=N / N_ud, g_ohl=g_ohl)


if __name__ == '__main__':
    print('ТОЧКА PYTHON (1250 К, π=11, 110 кВт):')
    cur = C.python_cycle(11.0, T_g=1250.0, N=110e3)
    fix = cycle_fixed(11.0)
    mc = C.mathcad_cycle(11.0, T_sm=1250.0, N=110e3)
    for name, d in (('как в ноутбуке', cur), ('c_p исправлена', fix),
                    ('Mathcad-модель', dict(T_k=mc['T_k'], L_k=mc['L_k'], g_m=mc['g_m'],
                                            cp_t=mc['cp_g_sr'], k_t=mc['k_g_sr'],
                                            L_t=mc['L_t_gg'] + mc['L_t_s'], T_t=mc['T_t_s'],
                                            N_ud=mc['L_val'], C_e=mc['C_e'], eta_e=mc['eta_e'],
                                            G_v=mc['G_v']))):
        print(f"  {name:18s} T*к={d['T_k']:6.1f} L_к={d['L_k']/1e3:6.1f} g_m={d['g_m']:.5f} "
              f"c_pт={d['cp_t']:5.0f} k_т={d['k_t']:.4f} L_т={d['L_t']/1e3:6.1f} "
              f"T*т={d['T_t']:6.1f} N_уд={d['N_ud']/1e3:5.1f} G_в={d['G_v']:.4f} "
              f"C_e={d['C_e']:.4f} η_e={d['eta_e']:.4f}")
    print()
    print('ТОЧКА MATHCAD (1440 К, π=10, 750 кВт):')
    cur = C.python_cycle(10.0, T_g=1440.0, N=750e3)
    fix = cycle_fixed(10.0, T_g=1440.0, N=750e3)
    mc = C.mathcad_cycle(10.0)
    for name, d in (('как в ноутбуке', cur), ('c_p исправлена', fix),
                    ('Mathcad-модель', dict(T_k=mc['T_k'], L_k=mc['L_k'], g_m=mc['g_m'],
                                            cp_t=mc['cp_g_sr'], k_t=mc['k_g_sr'],
                                            L_t=mc['L_t_gg'] + mc['L_t_s'], T_t=mc['T_t_s'],
                                            N_ud=mc['L_val'], C_e=mc['C_e'], eta_e=mc['eta_e'],
                                            G_v=mc['G_v']))):
        print(f"  {name:18s} T*к={d['T_k']:6.1f} L_к={d['L_k']/1e3:6.1f} g_m={d['g_m']:.5f} "
              f"c_pт={d['cp_t']:5.0f} k_т={d['k_t']:.4f} L_т={d['L_t']/1e3:6.1f} "
              f"T*т={d['T_t']:6.1f} N_уд={d['N_ud']/1e3:5.1f} G_в={d['G_v']:.4f} "
              f"C_e={d['C_e']:.4f} η_e={d['eta_e']:.4f}")
    print()
    print('π*к-сканирование с исправленной c_p (1250 К):')
    for p in (8, 9, 10, 11, 12, 14, 16, 18, 20):
        d = cycle_fixed(float(p))
        print(f'  π={p:>3}  C_e={d["C_e"]:.4f}  η_e={d["eta_e"]:.4f}')
