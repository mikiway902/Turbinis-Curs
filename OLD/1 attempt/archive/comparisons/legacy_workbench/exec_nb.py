#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Выполнить код-ячейки блокнота (как Jupyter) и перезаписать outputs.

Запуск:  PYTHONPATH=_pkgs MPLCONFIGDIR=_mpl python3 exec_nb.py <nb.ipynb> [--no-write]
"""
import ast
import base64
import contextlib
import io
import json
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
import matplotlib.pyplot as plt


def run_cell(src, ns, fname):
    """Выполнить код ячейки; вернуть (stdout, value, figures)."""
    tree = ast.parse(src, filename=fname, mode="exec")
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(tree, fname, "exec"), ns)
    val = None
    if last is not None:
        with contextlib.redirect_stdout(buf):
            val = eval(compile(last, fname, "eval"), ns)
    figs = list(plt.get_fignums())
    return buf.getvalue(), val, figs


def main(path, write=True):
    nb = json.load(open(path, encoding="utf-8"))
    ns = {"__name__": "__main__"}
    counts = 0
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        counts += 1
        src = "".join(l for l in cell["source"] if not l.lstrip().startswith(("%", "!")))
        fname = f"{os.path.basename(path)}[ячейка {i}]"
        try:
            text, val, figs = run_cell(src, ns, fname)
        except Exception as e:  # noqa: BLE001
            cell["outputs"] = [{
                "output_type": "error", "ename": type(e).__name__,
                "evalue": str(e), "traceback": [f"{type(e).__name__}: {e}"]}]
            cell["execution_count"] = counts
            print(f"!! ошибка в ячейке {i}: {type(e).__name__}: {e}")
            continue

        outs = []
        for num in figs:
            fig = plt.figure(num)
            b = io.BytesIO()
            fig.savefig(b, format="png", bbox_inches="tight")
            size = fig.get_size_inches() * fig.dpi
            outs.append({"output_type": "display_data",
                         "data": {"image/png": base64.b64encode(b.getvalue()).decode(),
                                  "text/plain": [f"<Figure size {size[0]:.0f}x{size[1]:.0f} with "
                                                 f"{len(fig.axes)} Axes>"]},
                         "metadata": {}})
            plt.close(fig)
        if text:
            outs.append({"output_type": "stream", "name": "stdout", "text": [text]})
        if val is not None:
            data = {}
            if hasattr(val, "to_html"):
                try:
                    data["text/html"] = [val.to_html()]
                except Exception:  # noqa: BLE001
                    pass
            data["text/plain"] = [repr(val)]
            outs.append({"output_type": "execute_result", "data": data,
                         "metadata": {}, "execution_count": counts})
        cell["outputs"] = outs
        cell["execution_count"] = counts
        print(f"ячейка {i}: ok, outputs={len(outs)}")

    if "--show" in sys.argv:
        for k in ("р", "G_в_выб", "C_e_выб", "η_e_выб", "π_к_выб", "T_т", "F_т", "D_т", "n_т_max", "G_в", "N_в_уд"):
            if k in ns:
                v = ns[k]
                try:
                    print(f"  {k} = {float(v):.6g}")
                except Exception:
                    try:
                        print(f"  {k} = {v:+.6g}")
                    except Exception:
                        pass

    if write:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print(f"записано: {path}")


if __name__ == "__main__":
    main(sys.argv[1], write="--no-write" not in sys.argv)
