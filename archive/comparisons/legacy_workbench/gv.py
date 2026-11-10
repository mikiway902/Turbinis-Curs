#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Подбор коэффициентов цикла так, чтобы вырос расход воздуха G_в при N = 110 кВт.
Считает и цикл, и геометрию из п. 6.1-6.2 блокнота (D_т, l_т, F_т, n_max)."""
import math
import compare as C

R = 287.4
K_n = 6.8e-6
SIG_L = 160.0
L_D = 0.250586
PI_K = 11.0
T_G = 1250.0
N = 110e3
P_A = 0.1013e6

BASE = {}          # как в блокноте сейчас

CAND = dict(eta_k_pol=0.85, eta_t_pol=0.90, eta_m=0.990, eta_r=0.98,
            sig_g=0.96, sig_vh=0.98, pi_s=1.05)

CAND2 = dict(CAND, eta_k_pol=0.83)          # + η_к как в Mathcad-файле
CAND3 = dict(CAND, M_t=0.30)                # + выходная скорость


def geom(d, pi_s, M_t):
    """п. 6.1-6.2 блокнота: T_т(стат), c_т, v_т, F_т, D_т, l_т, n_max, u_т"""
    k = C.k_py(d['T_t'], d['alpha'])
    T = d['T_t'] / (1 + (k - 1) / 2 * M_t**2)
    p = pi_s * P_A
    c = M_t * math.sqrt(k * R * T)
    v = R * T / p
    F = d['G_v'] * v / c
    n_max = math.sqrt(SIG_L / (K_n * F))
    D = math.sqrt(F / (math.pi * L_D))
    return dict(T_stat=T, c_t=c, v_t=v, F=F, n_max=n_max, D=D, l=L_D * D,
                u=math.pi * D * n_max / 60)


def row(name, kw, **gk):
    d = C.python_cycle(PI_K, T_g=T_G, N=N, **kw)
    pi_s = kw.get('pi_s', 1.04)
    M_t = kw.get('M_t', 0.2031)
    g = geom(d, pi_s, M_t)
    print(f"{name:34s}{d['N_ud']/1e3:8.1f}{d['G_v']:9.4f}{100*(d['G_v']/b0['G_v']-1):8.1f}"
          f"{d['C_e']:9.4f}{d['eta_e']:8.4f}{g['D']*1e3:8.0f}{g['l']*1e3:8.1f}"
          f"{g['F']*1e4:8.1f}{g['c_t']:7.0f}{g['n_max']:8.0f}")
    return d, g


b0 = C.python_cycle(PI_K, T_g=T_G, N=N)
print(f"{'вариант':34s}{'N_уд':>8}{'G_в':>9}{'ΔG_в,%':>8}{'C_e':>9}{'η_e':>8}"
      f"{'D_т,мм':>8}{'l_т,мм':>8}{'F_т,см²':>8}{'c_т':>7}{'n_max':>8}")
d0, g0 = row('0. как сейчас в блокноте', BASE)
row('1. консервативный край книги', CAND)
row('2. + η_к = 0.83 (как в Mathcad)', CAND2)
row('3. + M_т = 0.30', CAND3)

print()
print('по отдельности (что даёт каждый коэффициент, п.5)')
print(f"{'изменение':34s}{'N_уд':>8}{'G_в':>9}{'ΔG_в,%':>8}{'C_e':>9}{'η_e':>8}"
      f"{'D_т,мм':>8}{'l_т,мм':>8}{'F_т,см²':>8}{'c_т':>7}{'n_max':>8}")
for n, kw in [('η_к.политроп 0.86 -> 0.85', dict(eta_k_pol=0.85)),
              ('η_т.политроп 0.905 -> 0.90', dict(eta_t_pol=0.90)),
              ('η_м 0.9925 -> 0.990', dict(eta_m=0.990)),
              ('η_р 0.985 -> 0.98', dict(eta_r=0.98)),
              ('σ_г 0.965 -> 0.96', dict(sig_g=0.96)),
              ('σ_вх 0.985 -> 0.98', dict(sig_vh=0.98)),
              ('π_с 1.04 -> 1.05', dict(pi_s=1.05)),
              ('M_т 0.2031 -> 0.30', dict(M_t=0.30))]:
    row(n, kw)

print()
print('чего НЕ достаточно: топливные коэффициенты на G_в не влияют (G_в = N/N_уд)')
for n, kw in [('η_г 0.98 -> 0.97', dict(eta_g=0.97)), ('Q_н 43150 -> 42700', dict(Qn=42700.0))]:
    d = C.python_cycle(PI_K, T_g=T_G, N=N, **kw)
    print(f"  {n:30s} G_в = {d['G_v']:.4f} (Δ {100*(d['G_v']/b0['G_v']-1):+.2f} %), "
          f"C_e = {d['C_e']:.4f} (Δ {100*(d['C_e']/b0['C_e']-1):+.2f} %)")
