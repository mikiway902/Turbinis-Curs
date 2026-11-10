#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Прогон цикла из присланного блокнота TVD_pi6 (без numpy/pandas - заглушка)
и сравнение с Turbinis-Cycle."""
import json
import math
import os
import sys
import types

# --- заглушка numpy: функциям нужны только inf/isinf ---
np = types.ModuleType("numpy")
np.inf = math.inf
np.isinf = math.isinf
np.linspace = lambda a, b, n: [a + (b - a) * i / (n - 1) for i in range(n)]
sys.modules.setdefault("numpy", np)

CELLS = [4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 33]


def load(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TVD_pi6.ipynb')):
    nb = json.load(open(path, encoding='utf-8'))
    ns = {'__name__': 'tvd6', 'math': math, 'np': np}
    for i in CELLS[:10]:
        src = ''.join(nb['cells'][i]['source'])
        src = '\n'.join(l for l in src.splitlines()
                        if not l.startswith(('import ', 'from ', 'pd.set_option')))
        if i == 33:      # в этой ячейке строятся таблицы - не нужна
            continue
        exec(compile(src, f'cell{i}', 'exec'), ns)
    return ns


def run(ns, pi):
    d = ns['cycle'](float(pi))
    return d


def show(label, d):
    print(f"{label:38s}{d['pi_k']:6.2f}{d['G_air']:9.4f}{d['Ce_shaft']:10.4f}"
          f"{100*d['eta_shaft']:8.2f}{d['T3']:8.1f}{d['L_k']:8.1f}{d['pi_gg']:7.2f}"
          f"{d['pi_pt']:7.2f}{d['T4']:8.1f}{d['T5_total']:8.1f}{d['fuel_hour']:9.2f}")


def main():
    ns = load()
    print('=' * 108)
    print('ЦИКЛ ИЗ БЛОКНОТА TVD_pi6 (свои коэффициенты: c_вых=250 м/с, g_охл=0, eta_m=0.99)')
    print('=' * 108)
    print(f"{'':38s}{'π*к':>6}{'G_в':>9}{'C_e':>10}{'η_e,%':>8}{'T*к':>8}{'L_к':>8}"
          f"{'π ТГГ':>7}{'π СТ':>7}{'T*4':>8}{'T*5':>8}{'кг/ч':>9}")
    show('как есть (π*к = 6, их точка)', run(ns, 6.0))
    for pi in (7, 8, 9, 10, 11, 12, 14, 16, 18.8):
        show(f'тот же блокнот при π*к = {pi:g}', run(ns, pi))

    print()
    print('что стоит их π*к = 6 против π*к = 11 (в их же модели): ',
          f"{100*(run(ns, 6.0)['Ce_shaft']/run(ns, 11.0)['Ce_shaft']-1):+.1f} % по C_e")

    # их модель, но с моими установками
    print()
    print('=' * 108)
    print('ИХ МОДЕЛЬ С МОИМИ УСТАНОВКАМИ (c_вых 250 -> 108 м/с, охлаждение 1.5 %, '
          'eta_m 0.99->0.9925, eta_red 0.98->0.98)')
    print('=' * 108)
    for label, patch in [
        ('их настройки', {}),
        ('c_вых 250 -> 108 м/с', dict(c_pt_exit=108.0)),
        ('+ g_охл 1.5 % (g_leak 0.005)', dict(c_pt_exit=108.0, g_leak=0.0075, g_cool=0.015)),
        ('+ eta_m 0.9925', dict(c_pt_exit=108.0, g_leak=0.0075, g_cool=0.015,
                                eta_m_gg=0.9925, eta_m_pt=0.9925)),
    ]:
        old = {k: ns[k] for k in patch}
        ns.update(patch)
        try:
            d = run(ns, 11.0)
            show(label, d)
        finally:
            ns.update(old)

    print()
    print('для справки: их модель при π*к = 6 с c_вых = 108 м/с')
    old = ns['c_pt_exit']
    ns['c_pt_exit'] = 108.0
    show('π*к=6, c_вых=108', run(ns, 6.0))
    ns['c_pt_exit'] = old


if __name__ == '__main__':
    main()
