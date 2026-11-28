#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SVG: C_e(π*к) - две модели, два уровня T*г (две панели)."""
import compare as C

PIS = [6 + 0.2 * i for i in range(71)]
W, H = 900, 430
x0, x1 = 6.0, 20.0
y0, y1 = 0.245, 0.345
PANELS = [
    (60, 'T*г = 1440 К  (точка Mathcad: 750 кВт, π*к = 10)',
     lambda p: C.mathcad_cycle(p, T_sm=1440.0)['C_e'],
     lambda p: C.python_cycle(p, T_g=1440.0, N=750e3)['C_e'],
     [(10.0, C.mathcad_cycle(10.0)['C_e'], 'Mathcad'),
      (11.0, C.python_cycle(11.0, T_g=1440.0, N=750e3)['C_e'], 'Python')]),
    (490, 'T*г = 1250 К  (точка Python: 110 кВт, π*к = 11)',
     lambda p: C.mathcad_cycle(p, T_sm=1250.0)['C_e'],
     lambda p: C.python_cycle(p, T_g=1250.0, N=110e3)['C_e'],
     [(11.0, C.python_cycle(11.0, T_g=1250.0, N=110e3)['C_e'], 'Python')]),
]
PW, PH = 340, 260
TOP, BOT = 95, 65

out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}" font-family="DejaVu Sans, Arial, sans-serif">',
       f'<rect width="{W}" height="{H}" fill="white"/>',
       '<text x="60" y="34" font-size="15" font-weight="bold" fill="#111">'
       'Удельный расход топлива C_e(π*к): Mathcad (штрих) и Turbinis-Cycle (сплошная)</text>',
       '<text x="60" y="55" font-size="12" fill="#555">Точки сравнения обведены кружком; '
       'при одинаковом T*г модели совпадают в пределах 1…2 %</text>']

for xoff, title, f_mc, f_py, marks in PANELS:
    def sx(x, xoff=xoff):
        return xoff + (x - x0) / (x1 - x0) * PW

    def sy(y):
        return TOP + (y1 - y) / (y1 - y0) * PH

    out.append(f'<text x="{xoff + PW/2:.0f}" y="{TOP-12}" font-size="12" font-weight="bold" '
               f'fill="#222" text-anchor="middle">{title}</text>')
    for gx in range(6, 21, 2):
        out.append(f'<line x1="{sx(gx):.1f}" y1="{sy(y0):.1f}" x2="{sx(gx):.1f}" y2="{sy(y1):.1f}" stroke="#ececec"/>')
        out.append(f'<text x="{sx(gx):.1f}" y="{sy(y0)+15:.1f}" font-size="10" fill="#555" '
                   f'text-anchor="middle">{gx}</text>')
    for gy in [0.25, 0.27, 0.29, 0.31, 0.33]:
        out.append(f'<line x1="{sx(x0):.1f}" y1="{sy(gy):.1f}" x2="{sx(x1):.1f}" y2="{sy(gy):.1f}" stroke="#ececec"/>')
        out.append(f'<text x="{sx(x0)-6:.1f}" y="{sy(gy)+3.5:.1f}" font-size="10" fill="#555" '
                   f'text-anchor="end">{gy:.2f}</text>')
    out.append(f'<line x1="{sx(x0)}" y1="{sy(y0)}" x2="{sx(x1)}" y2="{sy(y0)}" stroke="#333"/>')
    out.append(f'<line x1="{sx(x0)}" y1="{sy(y0)}" x2="{sx(x0)}" y2="{sy(y1)}" stroke="#333"/>')
    out.append(f'<text x="{xoff+PW/2:.0f}" y="{sy(y0)+34:.0f}" font-size="11" fill="#222" '
               f'text-anchor="middle">π*к</text>')
    out.append(f'<text x="{xoff-40}" y="{TOP+PH/2:.0f}" font-size="11" fill="#222" text-anchor="middle" '
               f'transform="rotate(-90 {xoff-40} {TOP+PH/2:.0f})">C_e, кг/(кВт·ч)</text>')

    for f, color, dash in ((f_mc, '#c1121f', '6 4'), (f_py, '#1d4e89', None)):
        pts = ' '.join(f"{sx(p):.1f},{sy(f(p)):.1f}" for p in PIS if y0 <= f(p) <= y1)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.2"'
                   + (f' stroke-dasharray="{dash}"' if dash else '') + '/>')
    for px, py, name in marks:
        out.append(f'<circle cx="{sx(px):.1f}" cy="{sy(py):.1f}" r="5" fill="none" '
                   f'stroke="#111" stroke-width="1.6"/>')
        out.append(f'<text x="{sx(px)+8:.1f}" y="{sy(py)-8:.1f}" font-size="10" fill="#111">'
                   f'{name}, π={px:g}</text>')

# легенда
lx, ly = 700, 330
out.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+30}" y2="{ly}" stroke="#1d4e89" stroke-width="2.2"/>')
out.append(f'<text x="{lx+38}" y="{ly+4}" font-size="11" fill="#222">Turbinis-Cycle</text>')
out.append(f'<line x1="{lx}" y1="{ly+20}" x2="{lx+30}" y2="{ly+20}" stroke="#c1121f" stroke-width="2.2" stroke-dasharray="6 4"/>')
out.append(f'<text x="{lx+38}" y="{ly+24}" font-size="11" fill="#222">Mathcad (файл)</text>')
out.append('</svg>')
open('CE_vs_pi.svg', 'w').write('\n'.join(out))
print('ok')
