"""Этап 3: полная сетка (n, D_вт, w2) 31×31×31 для двух конфигураций.

Остальные переменные фиксированы на настройке, которая вывела обе корневые
реактивности в допустимую область (этап 1):

  A: κ1=1.0, w1=0.90, α1_2=20°; остальное — из лучшей точки B87_relaxed
  B: κ1=1.0, w1=0.97, α1_2=20°, α1_3=23°; остальное — из лучшей точки B87_relaxed

Ограничения и ядро не меняются. Сетка покрывает все исходные границы
переменных, а не только окрестность базы.

    python scan3d.py --config A --shard 0 --shards 14
    python scan3d.py --merge
"""
from __future__ import annotations

import argparse
import csv
import glob
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import optimize_extended as oe
import analyze_extended as an
import optimizer_core as oc

RESULTS = oe.RESULTS
CONFIGS = {
    "A": dict(k1=1.0, w1=0.90, a12=20.0),
    "B": dict(k1=1.0, w1=0.97, a12=20.0, a13=23.0),
}
IDX = dict(n=0, D=1, w1=2, w2=3, a11=4, a12=5, a13=6, k1=7, k2=8, k3=9)
FIELDS = ["config", "n", "D_vt", "w2", "status", "hard", "hard_no_strength",
          "ok_count", "lc1_mm", "rho_root_1", "rho_root_2", "a2_1", "a2_2", "a2_3",
          "sigma", "qn", "eta_t", "lD", "u", "N_kW", "l_ok", "rho_ok",
          "a2_ok_B", "a2_ok_A", "sigma_ok", "eta_ok", "feasible",
          "ok_no_strength", "violation_count", "binding", "violations"]


def group(name: str) -> str:
    for g in ("α2_1", "α2_2", "α2_3"):
        if name.startswith(g):
            return g
    if name.startswith("l_c1"):
        return "l_c1"
    if name.startswith("ρ_корня_1"):
        return "ρ1"
    if name.startswith("ρ_корня_2"):
        return "ρ2"
    if name.startswith("σ > ") or name.startswith("q_n > "):
        return "strength"
    if name.startswith("l_t/D_t"):
        return "lD"
    if name.startswith("u >"):
        return "u"
    return "model"


def base_point() -> np.ndarray:
    rows = an.load_csv(RESULTS / "extended_B87_relaxed_q0.995_full10.csv")
    return np.asarray(min(rows, key=lambda r: r["hard"])["x"], dtype=float)


def config_point(cfg: str) -> np.ndarray:
    x = base_point()
    for k, v in CONFIGS[cfg].items():
        x[IDX[k]] = v
    return x


def eval_row(cfg, x, ns, ql):
    r = oe.eval_candidate(x, ns, "B87_relaxed", ql)
    a2 = (r["a2"] + [float("nan")] * 3)[:3]
    rr = (r["rho_root"] + [float("nan")] * 2)[:2]
    groups = {group(v["name"]) for v in r["violations"]}
    nonstrength = [v for v in r["violations"] if group(v["name"]) != "strength"]
    a2_ok_b = all(lo <= v <= hi for v, (lo, hi) in zip(a2, oe.SCENARIOS["B87_relaxed"]["a2"]))
    a2_ok_a = all(lo <= v <= hi for v, (lo, hi) in zip(a2, [(70, 90), (70, 90), (80, 90)]))
    return {
        "config": cfg, "n": float(x[0]), "D_vt": float(x[1]), "w2": float(x[3]),
        "status": r["status"], "hard": r["hard"],
        "hard_no_strength": max((v["amount"] for v in nonstrength), default=0.0),
        "ok_count": 10 - len(groups),
        "lc1_mm": r["lc1"] * 1e3, "rho_root_1": rr[0], "rho_root_2": rr[1],
        "a2_1": a2[0], "a2_2": a2[1], "a2_3": a2[2],
        "sigma": r["sigma"], "qn": r["qn"], "eta_t": r["eta_t"],
        "lD": r["lD"], "u": r["u"], "N_kW": r["N"] / 1e3,
        "l_ok": bool(r["lc1"] >= 0.014),
        "rho_ok": bool(rr[0] >= 0.05 and rr[1] >= 0.05),
        "a2_ok_B": a2_ok_b, "a2_ok_A": a2_ok_a,
        "sigma_ok": bool(r["sigma"] <= 300.0),
        "eta_ok": bool(r["eta_t"] >= 0.90 - 1e-9),
        "feasible": bool(r["feasible"]),
        "ok_no_strength": bool(nonstrength == []),
        "violation_count": r["violation_count"],
        "binding": (max(r["violations"], key=lambda v: v["amount"])["name"]
                    if r["violations"] else "—"),
        "violations": "; ".join(f"{v['name']}:{v['amount']:.3f}" for v in r["violations"]),
    }


def fnum(v, nd=4):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def point_line(r, tag):
    return (f"- **{tag}**: n={r['n']/1e3:.1f} тыс., D_вт={r['D_vt']*1e3:.1f} мм, "
            f"w2={r['w2']:.3f}; hard={r['hard']:.3f}, l_c1={r['lc1_mm']:.2f} мм, "
            f"ρ={fnum(r['rho_root_1'])}/{fnum(r['rho_root_2'])}, "
            f"α2={r['a2_1']:.1f}/{r['a2_2']:.1f}/{r['a2_3']:.1f}, "
            f"σ={r['sigma']:.0f}, q_n={r['qn']:.4f}, η_t*={r['eta_t']:.4f}; "
            f"связывающее: {r['binding']}")


_W: dict = {}


def _init_worker(cfg: str, ql: float) -> None:
    m = oc.load_model()
    _W["cfg"] = cfg
    _W["ql"] = ql
    _W["x0"] = config_point(cfg)
    _W["ns"], _ = oe.build_ns(m, ql, ql ** 2 * oe.SIGMA_LIM)


def _work_n(nvals, steps: int) -> list[dict]:
    dvals = np.linspace(*oe.BOUNDS[1], steps)
    wvals = np.linspace(*oe.BOUNDS[3], steps)
    out = []
    for nv in nvals:
        for dv in dvals:
            for wv in wvals:
                x = _W["x0"].copy()
                x[0], x[1], x[3] = nv, dv, wv
                out.append(eval_row(_W["cfg"], x, _W["ns"], _W["ql"]))
    return out


def run_workers(cfg: str, workers: int, steps: int, ql: float) -> None:
    """Вся сетка конфигурации; детерминированное разбиение по индексу n."""
    nvals = np.linspace(*oe.BOUNDS[0], steps)
    chunks = [nvals[i::workers] for i in range(workers)]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(cfg, ql)) as ex:
        parts = list(ex.map(_work_n, chunks, [steps] * workers))
    rows = [r for part in parts for r in part]
    rows.sort(key=lambda r: (r["n"], r["D_vt"], r["w2"]))
    path = RESULTS / f"scan3d_{cfg}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({f: r.get(f, "") for f in FIELDS})
    print(f"{cfg}: {len(rows)} точек ({workers} процессов) -> {path.name}", flush=True)


def shard(cfg: str, i: int, k: int, steps: int, ql: float) -> None:
    m = oc.load_model()
    x0 = config_point(cfg)
    ns, _ = oe.build_ns(m, ql, ql ** 2 * oe.SIGMA_LIM)
    nvals = np.linspace(*oe.BOUNDS[0], steps)[i::k]
    dvals = np.linspace(*oe.BOUNDS[1], steps)
    wvals = np.linspace(*oe.BOUNDS[3], steps)
    out = []
    for nv in nvals:
        for dv in dvals:
            for wv in wvals:
                x = x0.copy()
                x[0], x[1], x[3] = nv, dv, wv
                out.append(eval_row(cfg, x, ns, ql))
    path = RESULTS / f"scan3d_{cfg}_shard{i:02d}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in out:
            w.writerow({f: r.get(f, "") for f in FIELDS})
    print(f"{cfg} shard {i}: {len(out)} точек -> {path.name}", flush=True)


def load_shards(cfg: str) -> list[dict]:
    rows = []
    paths = sorted(glob.glob(str(RESULTS / f"scan3d_{cfg}_shard*.csv")))
    if not paths and (RESULTS / f"scan3d_{cfg}.csv").exists():
        paths = [str(RESULTS / f"scan3d_{cfg}.csv")]
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                for k in ("n", "D_vt", "w2", "hard", "hard_no_strength", "lc1_mm",
                          "rho_root_1", "rho_root_2", "a2_1", "a2_2", "a2_3",
                          "sigma", "qn", "eta_t", "lD", "u", "N_kW"):
                    r[k] = float(r[k])
                for k in ("l_ok", "rho_ok", "a2_ok_B", "a2_ok_A", "sigma_ok",
                          "eta_ok", "feasible", "ok_no_strength"):
                    r[k] = r[k] == "True"
                r["ok_count"] = int(float(r["ok_count"]))
                r["violation_count"] = int(float(r["violation_count"]))
                rows.append(r)
    return rows


def merge(steps: int) -> None:
    lines = ["# Этап 3: сетка (n, D_вт, w2) — результаты", "",
             f"Сетка {steps}×{steps}×{steps} = {steps**3} точек на конфигурацию; "
             "остальные переменные зафиксированы на настройке этапа 1.", ""]
    summary = []
    for cfg in CONFIGS:
        rows = load_shards(cfg)
        ok = [r for r in rows if r["status"] == "ok"]
        if not ok:
            lines += [f"## Конфигурация {cfg}", "нет данных", ""]
            continue
        with (RESULTS / f"scan3d_{cfg}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({f: r.get(f, "") for f in FIELDS})
        d = CONFIGS[cfg]
        lines += [f"## Конфигурация {cfg} ({', '.join(f'{k}={v}' for k, v in d.items())})", "",
                  f"Точек: {len(rows)} (ok {len(ok)}, ошибок {len(rows)-len(ok)}); "
                  f"hard {min(r['hard'] for r in ok):.3f}…{max(r['hard'] for r in ok):.3f}.", ""]
        best = min(ok, key=lambda r: r["hard"])
        lines.append(point_line(best, "мин. hard"))
        len14 = [r for r in ok if r["l_ok"]]
        if len14:
            lines.append(point_line(max(len14, key=lambda r: r["lc1_mm"]), "макс. l_c1"))
        a2l = [r for r in ok if r["a2_ok_B"] and r["l_ok"]]
        if a2l:
            lines.append(point_line(max(a2l, key=lambda r: r["lc1_mm"]),
                                    "макс. l_c1 при α2 в норме (B)"))
        lr = [r for r in ok if r["l_ok"] and r["rho_ok"]]
        if lr:
            lines.append(point_line(max(lr, key=lambda r: r["a2_2"]),
                                    "макс. α2_2 при l≥14 и ρ≥0.05"))
        arl = [r for r in ok if r["a2_ok_B"] and r["rho_ok"] and r["l_ok"]]
        if arl:
            lines.append(point_line(max(arl, key=lambda r: r["lc1_mm"]),
                                    "макс. l_c1 при α2, ρ, l в норме"))
        target = [r for r in ok if r["feasible"] and r["eta_ok"]]
        if target:
            lines.append(point_line(max(target, key=lambda r: r["lc1_mm"]),
                                    "цель: все ограничения и η_t* ≥ 0.90"))
        ns_ = [r for r in ok if r["ok_no_strength"]]
        if ns_:
            lines.append(point_line(min(ns_, key=lambda r: r["sigma"]),
                                    "мин. σ при всех, кроме прочности"))
        lines += ["",
                  f"Множества: l_c1 ≥ 14 — {sum(r['l_ok'] for r in ok)}; "
                  f"l ≥ 14 и ρ ok — {len(lr)}; "
                  f"l ≥ 14 и α2 ok (B) — {len(a2l)}; "
                  f"l ≥ 14, α2 ok, ρ ok — {len(arl)}; "
                  f"все ограничения — {sum(r['feasible'] for r in ok)}; "
                  f"все кроме прочности — {len(ns_)}; "
                  f"η_t* ≥ 0.90 — {sum(r['eta_ok'] for r in ok)}; "
                  f"все и η_t* ≥ 0.90 — {len(target)}.", ""]
        closest = sorted(ok, key=lambda r: (r["hard"], -r["lc1_mm"]))[:5]
        lines.append("Ближайшие к допустимости:")
        for r in closest:
            lines.append(point_line(r, "кандидат"))
        lines.append("")
        summary.append(dict(
            config=cfg, points=len(rows), ok=len(ok), err=len(rows) - len(ok),
            hard_min=best["hard"],
            l_max=(max((r["lc1_mm"] for r in len14), default=float("nan"))),
            l_max_a2=(max((r["lc1_mm"] for r in a2l), default=float("nan"))),
            a2_2_max_lr=(max((r["a2_2"] for r in lr), default=float("nan"))),
            l_max_all_ok=(max((r["lc1_mm"] for r in arl), default=float("nan"))),
            n_l=len(len14), n_lr=len(lr), n_a2=len(a2l), n_all_ok=len(arl),
            n_feasible=sum(r["feasible"] for r in ok),
            n_nostr=len(ns_), n_target=len(target),
        ))
    if summary:
        with (RESULTS / "scan3d_summary.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
        lines += ["# Сводка", "",
                  "| конфиг | точек | hard min | max l (l≥14) | max l при α2 ok | "
                  "max α2_2 при l≥14+ρ | max l при всех кроме прочности | l≥14 | l+ρ | l+α2 | l+α2+ρ | всё | всё+η |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in summary:
            lines.append(
                f"| {s['config']} | {s['points']} | {s['hard_min']:.3f} | "
                f"{fnum(s['l_max'], 2)} | {fnum(s['l_max_a2'], 2)} | "
                f"{fnum(s['a2_2_max_lr'], 1)} | {fnum(s['l_max_all_ok'], 2)} | "
                f"{s['n_l']} | {s['n_lr']} | {s['n_a2']} | {s['n_all_ok']} | {s['n_feasible']} | "
                f"{s['n_target']} |")
        lines.append("")
    (RESULTS / "REPORT_SCAN3D.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="A")
    ap.add_argument("--workers", type=int, default=0,
                    help=">0: вся конфигурация через ProcessPoolExecutor")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--steps", type=int, default=31)
    ap.add_argument("--q-n", type=float, default=0.995)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merge(args.steps)
    elif args.workers > 0:
        run_workers(args.config, args.workers, args.steps, args.q_n)
    else:
        shard(args.config, args.shard, args.shards, args.steps, args.q_n)


if __name__ == "__main__":
    main()
