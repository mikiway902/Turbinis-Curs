"""Анализ результатов optimize_extended.py.

Считает по сценарию: число оценок, допустимые точки, seed-устойчивость,
максимум l_c1, минимум hard, распределения α2/ρ_корня/σ/q_n/η_t*, частоту
нарушений. Повторно проверяет лучшего кандидата на свежем экземпляре модели
и полноту фазы 2. Сравнивает прогоны full10 (q_n=0.995) и qn1 (q_n=1.0).

    python analyze_extended.py --tags full10 qn1
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import statistics as stx
from collections import Counter
from pathlib import Path

import numpy as np

import optimize_extended as oe
import optimizer_core as oc

RESULTS = Path(__file__).resolve().parent / "results"
FLOAT_COLS = ("score1", "score2", "hard", "hard_max", "n", "D_hub", "w1", "w2",
              "lc1", "sigma", "u", "lD", "qn", "N", "eta_t", "eta_below",
              "cont", "energy", "p2z")
LIST_COLS = ("a2", "rho", "rho_root", "uc0", "uc0_opt")


def _flist(s: str) -> list[float]:
    out = []
    for part in s.strip().strip("[]()").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            out.append(float("nan"))
    return out


def load_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for c in FLOAT_COLS:
                try:
                    r[c] = float(r[c])
                except (TypeError, ValueError):
                    r[c] = float("nan")
            for c in LIST_COLS:
                r[c] = _flist(r.get(c, ""))
            r["feasible"] = str(r["feasible"]).strip() == "True"
            r["eta_ok"] = str(r.get("eta_ok", "")).strip() == "True"
            r["phase"] = int(float(r["phase"]))
            r["seed"] = int(float(r["seed"]))
            r["x"] = [float(v) for v in str(r.get("x", "")).split(";") if v]
            r["violations"] = [v for v in str(r.get("violations", "")).split("; ") if v]
            rows.append(r)
    return rows


def find_results(tag: str) -> dict[str, tuple[Path, float]]:
    """сценарий -> (файл, лимит q_n из имени файла)"""
    out = {}
    for p in sorted(RESULTS.glob(f"extended_*_{tag}.csv")):
        m = re.match(r"extended_(.+)_q([0-9.]+)_" + re.escape(tag) + r"\.csv$", p.name)
        if m:
            out[m.group(1)] = (p, float(m.group(2)))
    return out


def fmt(v: float, nd: int = 2) -> str:
    return "—" if not math.isfinite(v) else f"{v:.{nd}f}"


def rng(vals: list[float], nd: int = 1, scale: float = 1.0) -> str:
    v = [x * scale for x in vals if math.isfinite(x)]
    if not v:
        return "—"
    return f"{min(v):.{nd}f}…{max(v):.{nd}f} (мед {stx.median(v):.{nd}f})"


def col(rows: list[dict], key: str) -> list[float]:
    return [r[key] for r in rows if math.isfinite(r.get(key, float("nan")))]


def flat(rows: list[dict], key: str, i: int) -> list[float]:
    out = []
    for r in rows:
        v = r.get(key) or []
        if len(v) > i and math.isfinite(v[i]):
            out.append(v[i])
    return out


def scenario_metrics(rows: list[dict], q_limit: float) -> dict:
    ph1 = [r for r in rows if r["phase"] == 1]
    ph2 = [r for r in rows if r["phase"] == 2]
    feas = [r for r in rows if r["feasible"]]
    feas_eta = [r for r in feas if r["eta_ok"]]
    seeds = sorted({r["seed"] for r in rows})
    seeds_f = sorted({r["seed"] for r in feas})
    # максимум l_c1: приоритет — допустимые с η_t*≥0.90
    pool = feas_eta or feas
    best_len = max(pool, key=lambda r: (r["lc1"], -r["score2"])) if pool else None
    best_hard = min(rows, key=lambda r: r["hard"]) if rows else None
    infeas = [r for r in rows if not r["feasible"] and r["status"] == "ok"]
    best_inf = min(infeas, key=lambda r: r["hard"]) if infeas else None
    statuses = Counter(r["status"] for r in rows)
    return dict(
        rows=rows, ph1=ph1, ph2=ph2, feas=feas, feas_eta=feas_eta,
        seeds=seeds, seeds_f=seeds_f, best_len=best_len, best_hard=best_hard,
        best_inf=best_inf, statuses=statuses, q_limit=q_limit,
    )


def violation_stats(rows: list[dict], top: int = 10) -> list[tuple[str, int]]:
    c = Counter()
    for r in rows:
        for v in r["violations"]:
            c[v.split(":")[0].strip()] += 1
    return c.most_common(top)


def recheck(best: dict, scenario: str, q_limit: float, m) -> dict:
    """повторная оценка лучшего кандидата на свежем ns (независимость + residual)"""
    ns, _ = oe.build_ns(m, q_limit, q_limit ** 2 * oe.SIGMA_LIM)
    r = oe.eval_candidate(best["x"], ns, scenario, q_limit)
    out = {}
    for k in ("lc1", "sigma", "lD", "qn", "N", "eta_t", "hard", "score1"):
        a, b = best.get(k, float("nan")), r.get(k, float("nan"))
        out[k] = (a, b, abs(a - b) if math.isfinite(a) and math.isfinite(b) else float("nan"))
    a2a, a2b = best["a2"], r["a2"]
    out["a2"] = (a2a, a2b, max((abs(x - y) for x, y in zip(a2a, a2b)), default=float("nan")))
    out["feasible"] = (best["feasible"], r["feasible"])
    out["violations_equal"] = ([v.split(":")[0] for v in best["violations"]]
                               == [v["name"] for v in r["violations"]])
    return out


def report_tag(tag: str, files: dict[str, tuple[Path, float]], m,
               do_recheck: bool) -> tuple[str, list[dict], list[dict]]:
    lines = [f"# Отчёт: {tag}", ""]
    summary: list[dict] = []
    seed_summary: list[dict] = []
    for sc, (path, ql) in files.items():
        rows = load_csv(path)
        met = scenario_metrics(rows, ql)
        lines += [f"## {sc}  (q_n ≤ {ql:g})", ""]
        lines.append(f"- оценок: {len(rows)} (фаза 1 — {len(met['ph1'])}, фаза 2 — {len(met['ph2'])}); "
                     f"seed: {len(met['seeds'])}")
        lines.append(f"- допустимых: {len(met['feas'])}; seed с допустимыми: {len(met['seeds_f'])}"
                     + (f" {met['seeds_f']}" if met["seeds_f"] else ""))
        lines.append(f"- статусы: " + ", ".join(f"{k}={v}" for k, v in met["statuses"].items()))
        if met["best_hard"]:
            b = met["best_hard"]
            lines.append(f"- минимум hard: {b['hard']:.4g} (seed {b['seed']}, фаза {b['phase']}); "
                         f"l_c1={fmt(b['lc1']*1e3)} мм, α2={[round(v,1) for v in b['a2']]}, "
                         f"σ={fmt(b['sigma'],0)} МПа, q_n={fmt(b['qn'],4)}, "
                         f"l/D={fmt(b['lD'],4)}, η_t*={fmt(b['eta_t'],4)}")
            if b["violations"]:
                lines.append("  - нарушения: " + "; ".join(b["violations"]))
        if met["best_len"]:
            b = met["best_len"]
            lines.append(f"- максимум l_c1 среди {'допустимых η_t*≥0.90' if met['feas_eta'] else 'допустимых'}: "
                         f"{fmt(b['lc1']*1e3)} мм (seed {b['seed']}, фаза {b['phase']}); "
                         f"α2={[round(v,1) for v in b['a2']]}, σ={fmt(b['sigma'],0)}, "
                         f"q_n={fmt(b['qn'],4)}, η_t*={fmt(b['eta_t'],4)}, "
                         f"l/D={fmt(b['lD'],4)}, n={fmt(b['n']/1e3,1)} тыс, "
                         f"D_вт={fmt(b['D_hub']*1e3,1)} мм")
        else:
            lines.append("- допустимых точек нет; максимум l_c1 не определён")
        if met["best_inf"]:
            b = met["best_inf"]
            lines.append(f"- лучшая недопустимая: hard={b['hard']:.4g}, l_c1={fmt(b['lc1']*1e3)} мм, "
                         f"l/D={fmt(b['lD'],4)}, q_n={fmt(b['qn'],4)}")
        for name, label in (("a2", "α2"), ("rho_root", "ρ_корня"), ("rho", "ρ_ср")):
            ns_ = 3 if name == "a2" else 4
            ncomp = max((len(r.get(name) or []) for r in met["feas"]), default=0)
            parts = [f"{i+1}: {rng(flat(met['feas'], name, i), ns_)}" for i in range(ncomp)]
            if parts:
                lines.append(f"- допустимые, {label}: " + "; ".join(parts))
        lines.append(f"- допустимые, σ: {rng(col(met['feas'], 'sigma'), 0)} МПа; "
                     f"q_n: {rng(col(met['feas'], 'qn'), 4)}; "
                     f"η_t*: {rng(col(met['feas'], 'eta_t'), 4)}")
        vs = violation_stats(met["rows"])
        if vs:
            lines.append("- частота нарушений: " + ", ".join(f"{n} ({c})" for n, c in vs))
        # по seed: лучший hard и его параметры
        per = {}
        for s_ in met["seeds"]:
            rs = [r for r in met["rows"] if r["seed"] == s_]
            b = min(rs, key=lambda r: r["hard"])
            per[s_] = b
            seed_summary.append(dict(
                tag=tag, scenario=sc, seed=s_, evals=len(rs),
                feasible_evals=sum(r["feasible"] for r in rs),
                phase2_evals=sum(r["phase"] == 2 for r in rs),
                min_hard=b["hard"], lc1_mm=b["lc1"] * 1e3,
                a2=[round(v, 2) for v in b["a2"]], rho_root=[round(v, 4) for v in b["rho_root"]],
                sigma=b["sigma"], qn=b["qn"], lD=b["lD"], eta_t=b["eta_t"],
                n=b["n"], D_hub=b["D_hub"], seed_point_hard=b["hard"]))
        hs = [b["hard"] for b in per.values()]
        lines.append("- min hard по seed: " + ", ".join(f"с{s_}:{b['hard']:.3f}" for s_, b in per.items()))
        lines.append(f"  - разброс min hard: {min(hs):.3f}…{max(hs):.3f}, медиана {stx.median(hs):.3f}")
        # полнота фазы 2
        if met["seeds_f"]:
            ok = bool(met["ph2"]) and any(r["feasible"] for r in met["ph2"])
            lines.append(f"- фаза 2: {'запущена и содержит допустимые' if ok else 'НЕПОЛНА'} "
                         f"({len(met['ph2'])} оценок)")
        cand = met["best_len"] or met["best_hard"]
        if do_recheck and cand:
            rc = recheck(cand, sc, ql, m)
            diffs = {k: v[2] for k, v in rc.items() if isinstance(v, tuple) and len(v) == 3}
            worst = max((d for d in diffs.values() if math.isfinite(d)), default=float("nan"))
            lines.append(f"- повторная проверка на свежем ns: max |Δ| = {worst:.2e}; "
                         f"feasible {rc['feasible'][0]}→{rc['feasible'][1]}; "
                         f"набор нарушений совпал: {rc['violations_equal']}")
        lines.append("")
        summary.append(dict(
            tag=tag, scenario=sc, q_limit=ql, evals=len(rows),
            phase1=len(met["ph1"]), phase2=len(met["ph2"]),
            feasible=len(met["feas"]), seeds_feasible=len(met["seeds_f"]),
            seeds_total=len(met["seeds"]),
            max_lc1_mm=(met["best_len"]["lc1"] * 1e3 if met["best_len"] else float("nan")),
            max_lc1_seed=(met["best_len"]["seed"] if met["best_len"] else ""),
            min_hard=(met["best_hard"]["hard"] if met["best_hard"] else float("nan")),
            min_hard_feasible=(min((r["hard"] for r in met["feas"]), default=float("nan"))),
            sigma_max_feas=(max(col(met["feas"], "sigma"), default=float("nan"))),
            qn_max_feas=(max(col(met["feas"], "qn"), default=float("nan"))),
            eta_min_feas=(min(col(met["feas"], "eta_t"), default=float("nan"))),
            numerical=met["statuses"].get("numerical", 0),
            model_error=met["statuses"].get("model_error", 0),
            binding=(violation_stats(met["rows"], 1)[0][0] if violation_stats(met["rows"], 1) else ""),
        ))
    return "\n".join(lines) + "\n", summary, seed_summary


def comparison(summaries: dict[str, list[dict]]) -> str:
    tags = list(summaries)
    if len(tags) < 2:
        return ""
    lines = ["# Сравнение прогонов", "",
             "| сценарий | " + " | ".join(tags) + " |",
             "|---|" + "---|" * len(tags)]
    by = {t: {r["scenario"]: r for r in summaries[t]} for t in tags}
    for sc in by[tags[0]]:
        cells = []
        for t in tags:
            r = by[t].get(sc)
            if not r:
                cells.append("—")
                continue
            cells.append(f"доп. {r['feasible']} / seed {r['seeds_feasible']}/{r['seeds_total']}, "
                         f"l≤{fmt(r['max_lc1_mm'])} мм, hard≥{r['min_hard']:.3g}, "
                         f"q_n≤{r['q_limit']:g}")
        lines.append(f"| {sc} | " + " | ".join(cells) + " |")
    lines += ["", "Напоминание: отсутствие допустимых не доказывает отсутствие области; "
              "прочность с учётом температуры, ресурс и корневая ρ 3-й ступени — NOT_CHECKED.", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["full10", "qn1"])
    ap.add_argument("--no-recheck", action="store_true")
    args = ap.parse_args()
    m = oc.load_model() if not args.no_recheck else None
    summaries: dict[str, list[dict]] = {}
    for tag in args.tags:
        files = find_results(tag)
        if not files:
            print(f"{tag}: нет CSV в {RESULTS}")
            continue
        text, rows, seeds = report_tag(tag, files, m, not args.no_recheck)
        (RESULTS / f"REPORT_{tag}.md").write_text(text, encoding="utf-8")
        for name, data in ((f"SUMMARY_{tag}.csv", rows), (f"SUMMARY_{tag}_seeds.csv", seeds)):
            with (RESULTS / name).open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
        summaries[tag] = rows
        print(text)
    if len(summaries) > 1:
        text = comparison(summaries)
        (RESULTS / f"REPORT_{'_vs_'.join(summaries)}.md").write_text(text, encoding="utf-8")
        print(text)


if __name__ == "__main__":
    main()
