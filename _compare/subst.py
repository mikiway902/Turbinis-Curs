#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Порт Mathcad-файла на МОЁМ режиме (1250 К, 110 кВт, π=11, стенд) с МОИМ условием
охлаждения (g_охл = 1.5 % при T*г <= 1250, возврат 60 %) и сравнение с моим циклом."""
import math
import compare as C

MY = dict(T=1250.0, N=110e3, pi=11.0)
COOL = dict(g_ohl_used=0.015, g_voz=0.6 * 0.015)
MY_ALL = dict(eta_k_pol=0.86, sig_vh=0.985, sig_g=0.965, pi_s=1.04, eta_g=0.98,
              eta_m=0.9925, eta_r=0.985, Qn=43150e3, g_ut=0.0075, eta_t=0.929,
              c_t=108.0)
NA = float('nan')


def mcm(pi, **kw):
    return C.mathcad_cycle(pi, **kw)


def norm_mc(d):
    return dict(T_k=d['T_k'], L_k=d['L_k'] / 1e3, g_m=d['g_m'], alpha=d['alpha'],
                pi_t_sum=d['pi_t_gg'] * d['pi_t_c'], pi_t_gg=d['pi_t_gg'],
                pi_t_c=d['pi_t_c'], L_t=(d['L_t_gg'] + d['L_t_s']) / 1e3,
                L_t_gg=d['L_t_gg'] / 1e3, L_t_s=d['L_t_s'] / 1e3,
                T_t_gg=d['T_t_gg'], T_t=d['T_t_s'], cp_g=d['cp_g_sr'], k_g=d['k_g_sr'],
                N_ud=d['N_ud'] / 1e3, G_v=d['G_v'], G_t=d['G_t'],
                C_e=d['C_e'], eta_e=d['eta_e'])


def norm_py(d):
    return dict(T_k=d['T_k'], L_k=d['L_k'] / 1e3, g_m=d['g_m'], alpha=d['alpha'],
                pi_t_sum=d['pi_t'], pi_t_gg=NA, pi_t_c=NA, L_t=d['L_t'] / 1e3,
                L_t_gg=NA, L_t_s=NA, T_t_gg=NA, T_t=d['T_t'], cp_g=d['cp_t'],
                k_g=d['k_t'], N_ud=d['N_ud'] / 1e3, G_v=d['G_v'],
                G_t=d['g_m'] * (1 - 0.0075 - 0.015) * d['G_v'],
                C_e=d['C_e'], eta_e=d['eta_e'])


ROWS = [('T_k', '*к, К', '{:8.1f}'), ('L_k', 'L_к, кДж/кг', '{:8.1f}'),
        ('g_m', 'g_m', '{:8.5f}'), ('alpha', 'α', '{:6.2f}'),
        ('pi_t_sum', 'π*т сумм.', '{:8.2f}'), ('pi_t_gg', '  в т.ч. ТГГ', '{:8.2f}'),
        ('pi_t_c', '  в т.ч. СТ', '{:8.2f}'),
        ('L_t', 'L_т, кДж/кг', '{:8.1f}'), ('L_t_gg', '  в т.ч. ТГГ', '{:8.1f}'),
        ('L_t_s', '  в т.ч. СТ', '{:8.1f}'),
        ('T_t_gg', 'T*т после ТГГ, К', '{:8.1f}'), ('T_t', 'T*т за турбиной, К', '{:8.1f}'),
        ('cp_g', 'c_p газа', '{:8.0f}'), ('k_g', 'k газа', '{:8.4f}'),
        ('N_ud', 'N_в.уд, кДж/кг', '{:8.1f}'), ('G_v', 'G_в, кг/с', '{:8.3f}'),
        ('G_t', 'G_т, кг/с', '{:8.4f}'),
        ('C_e', 'C_e, кг/(кВт·ч)', '{:8.4f}'), ('eta_e', 'η_e', '{:8.4f}')]


def show(cols, width=15):
    hdr = f"{'':24s}" + ''.join(f"{c[0]:>{width}s}" for c in cols)
    print(hdr)
    print('-' * len(hdr))
    for key, name, fmt in ROWS:
        line = f"{name:24s}"
        for _, d in cols:
            v = d[key]
            cell = '—' if isinstance(v, float) and math.isnan(v) else fmt.format(v)
            line += f"{cell:>{width}s}"
        print(line)
    return hdr


def main():
    a0 = norm_mc(mcm(10.0, T_sm=1440.0, N=750e3))
    a1 = norm_mc(mcm(MY['pi'], T_sm=MY['T'], N=MY['N']))
    a2 = norm_mc(mcm(MY['pi'], T_sm=MY['T'], N=MY['N'], **COOL))
    a3 = norm_mc(mcm(MY['pi'], T_sm=MY['T'], N=MY['N'], **COOL, **MY_ALL))
    b = norm_py(C.python_cycle(MY['pi'], T_g=MY['T'], N=MY['N']))

    print('=' * 112)
    print('ПОРТ MATHCAD-ФАЙЛА НА МОЁМ РЕЖИМЕ: 1250 К, 110 кВт, π=11, стенд, моё охлаждение')
    print('=' * 112)
    show([('1. файл\n1440/750/π10', a0),
          ('2. мой режим\nбез охл.', a1),
          ('3. + моё охл.\n1.5%/возвр.60', a2),
          ('4. + все мои\nчисла и c_т', a3),
          ('5. МОЙ ЦИКЛ\nPython', b)])

    print()
    print(f"C_e:  {a0['C_e']:.4f} --> {a1['C_e']:.4f} --> {a2['C_e']:.4f} --> {a3['C_e']:.4f}   | мой цикл {b['C_e']:.4f}")
    print(f"η_e:  {a0['eta_e']:.4f} --> {a1['eta_e']:.4f} --> {a2['eta_e']:.4f} --> {a3['eta_e']:.4f}   | {b['eta_e']:.4f}")
    print(f"\nНа моём режиме с моим охлаждением файл даёт C_e на {100*(a2['C_e']/b['C_e']-1):+.1f} % "
          f"и G_в на {100*(a2['G_v']/b['G_v']-1):+.1f} % к моему циклу;")
    print(f"с моими числами и моим c_т — уже на {100*(a3['C_e']/b['C_e']-1):+.1f} % к моему циклу.")

    base = mcm(MY['pi'], T_sm=MY['T'], N=MY['N'], **COOL)
    print()
    print('=' * 112)
    print('РАЗЛОЖЕНИЕ: от «файл на моём режиме» к «мой цикл» — по одному изменению')
    print('=' * 112)
    print(f"{'что меняю на моё':40s}{'C_e':>9}{'Δ,%':>8}{'η_e':>9}{'G_в':>8}{'T*т':>8}{'N_уд':>8}")
    items = [
        ('η_к.политроп 0.83 -> 0.86', dict(eta_k_pol=0.86)),
        ('c_т 210 -> 108 м/с (мой M_т)', dict(c_t=108.0)),
        ('η*т 0.90 -> 0.929 (из η_тп)', dict(eta_t=0.929)),
        ('π_с 1.005 -> 1.04', dict(pi_s=1.04)),
        ('η_г 0.995 -> 0.98', dict(eta_g=0.98)),
        ('η_м 0.995 -> 0.9925', dict(eta_m=0.9925)),
        ('η_р 0.99 -> 0.985', dict(eta_r=0.985)),
        ('σ_г 0.97 -> 0.965', dict(sig_g=0.965)),
        ('σ_вх 0.98 -> 0.985', dict(sig_vh=0.985)),
        ('Q_н 43000 -> 43150', dict(Qn=43150e3)),
        ('g_ут 0.007 -> 0.0075', dict(g_ut=0.0075)),
    ]
    for name, kw in items:
        d = mcm(MY['pi'], T_sm=MY['T'], N=MY['N'], **COOL, **kw)
        print(f"{name:40s}{d['C_e']:9.4f}{100*(d['C_e']/base['C_e']-1):8.2f}{d['eta_e']:9.4f}"
              f"{d['G_v']:8.3f}{d['T_t_s']:8.1f}{d['N_ud']/1e3:8.1f}")
    print(f"{'все сразу':40s}{a3['C_e']:9.4f}{100*(a3['C_e']/base['C_e']-1):8.2f}{a3['eta_e']:9.4f}"
          f"{a3['G_v']:8.3f}{a3['T_t']:8.1f}{a3['N_ud']:8.1f}")
    print(f"{'→ мой цикл':40s}{b['C_e']:9.4f}{100*(b['C_e']/base['C_e']-1):8.2f}{b['eta_e']:9.4f}"
          f"{b['G_v']:8.3f}{b['T_t']:8.1f}{b['N_ud']:8.1f}")

    print()
    print('=' * 112)
    print('СКАНИРОВАНИЕ π*к НА МОЁМ РЕЖИМЕ (1250 К, 110 кВт, моё охлаждение)')
    print('=' * 112)
    print(f"{'π*к':>5}{'файл: C_e':>12}{'файл: G_в':>12}{'файл: η_e':>11}"
          f"{'   |':>5}{'мой цикл: C_e':>16}{'G_в':>9}{'η_e':>8}")
    for p in (7, 8, 9, 10, 11, 12, 13, 14, 16, 18):
        d = mcm(float(p), T_sm=MY['T'], N=MY['N'], **COOL)
        e = C.python_cycle(float(p), T_g=MY['T'], N=MY['N'])
        print(f"{p:>5}{d['C_e']:12.4f}{d['G_v']:12.3f}{d['eta_e']:11.4f}{'   |':>5}"
              f"{e['C_e']:16.4f}{e['G_v']:9.3f}{e['eta_e']:8.4f}")

    grid = [6 + 0.1 * i for i in range(131)]
    mcf = [mcm(p, T_sm=MY['T'], N=MY['N'], **COOL) for p in grid]
    pyf = [C.python_cycle(p, T_g=MY['T'], N=MY['N']) for p in grid]
    im = min(range(len(grid)), key=lambda i: mcf[i]['C_e'])
    ip = min(range(len(grid)), key=lambda i: pyf[i]['C_e'])
    print(f"\nmin C_e:  порт файла  π*к = {grid[im]:.1f}  (C_e = {mcf[im]['C_e']:.4f})"
          f"      мой цикл  π*к = {grid[ip]:.1f}  (C_e = {pyf[ip]['C_e']:.4f})")

    # отдельно: масштаб эффекта охлаждения в двух структурах
    print()
    print(f"Цена охлаждения (1.5 % + возврат 60 %): в структуре файла "
          f"{100*(a2['C_e']/a1['C_e']-1):+.1f} % к C_e, в моём цикле "
          f"{100*(C.python_cycle(MY['pi'], T_g=MY['T'], N=MY['N'])['C_e']/C.python_cycle(MY['pi'], T_g=MY['T'], N=MY['N'], g_ohl=0.0)['C_e']-1):+.1f} %")


if __name__ == '__main__':
    main()
