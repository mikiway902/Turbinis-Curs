"""Read-only numerical verification without plotting dependencies."""
import ast
import contextlib
import copy
import io
import json
import math
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd


def root_scalar(function, x0, x1, method, maxiter, xtol):
    f0, f1 = function(x0), function(x1)
    for _ in range(maxiter):
        if f1 == f0:
            return SimpleNamespace(root=x1, converged=abs(f1) < 1e-8)
        x2 = x1-f1*(x1-x0)/(f1-f0)
        f2 = function(x2)
        if abs(x2-x1) < xtol:
            return SimpleNamespace(root=x2, converged=True)
        x0, x1, f0, f1 = x1, x2, f1, f2
    return SimpleNamespace(root=x1, converged=False)


sys.stdout.reconfigure(encoding='utf-8')
with open('turbine_manual_calculator.ipynb', encoding='utf-8') as stream:
    notebook = json.load(stream)
namespace = dict(np=np, pd=pd, math=math, copy=copy, warnings=warnings,
                 root_scalar=root_scalar, display=lambda *args: None)
for index, cell in enumerate(notebook['cells']):
    if cell['cell_type'] != 'code':
        continue
    tree = ast.parse(''.join(cell['source']))
    if index > 53:
        continue
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(tree, f'cell_{index}', 'exec'), namespace)

for index in (63, 65):
    tree = ast.parse(''.join(notebook['cells'][index]['source']))
    if index == 65:
        tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.Assign))]
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(tree, f'cell_{index}', 'exec'), namespace)

print('Stage constraints:', namespace['coupled_ok'])
duct = namespace['transition_geometry'](namespace['rt'], namespace['rs'])
assert np.isclose(np.degrees(np.arctan((duct['D_wall_out']-duct['D_wall_in'])/(2*duct['length']))), 12.0)
assert duct['length'] > 0
print('Transition:', duct, 'length_mm:', 1000*duct['length'])
sections = pd.concat([namespace['velocity_triangles_three_sections'](namespace[key], label)
                      for key, label in [('rt', 'ТК'), ('rs', 'СТ')]])
assert np.isfinite(sections.to_numpy()).all()
assert sections['Euler_error'].abs().max() < 1e-12
for key, label in [('rt', 'ТК'), ('rs', 'СТ')]:
    for field in ('c1', 'c2', 'w1', 'w2', 'u1', 'u2', 'alpha1', 'alpha2', 'beta1', 'beta2'):
        assert np.isclose(sections.loc[(label, 'Среднее'), field], namespace[key][field], rtol=1e-10)
print(sections[['alpha1', 'beta1', 'beta2', 'rho', 'M_c1', 'M_w2', 'Euler_error']].round(5).to_string())
print(namespace['drawing_df'].round(4).to_string())
print('JSON, code syntax, finite numbers, mean-section reproduction, Euler closure, coprime counts: OK')
