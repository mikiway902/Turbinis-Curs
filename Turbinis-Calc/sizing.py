"""Предварительное определение диаметров по A, l/D и u/c0.

Последовательность (Михальцев, Моляков, ч. 1, с. 58, 85–90):
  1) A = G·v / c_a;
  2) λ = l/D_ср — выбирается по таблице с. 58;
  3) D_ср = √(A / (π·λ));
  4) l = λ·D_ср;
  5) D_вт = D_ср − l;
  6) D_нар = D_ср + l;
  7) u = π·D_ср·n/60;  (u/c0) сверяется с рис. 18.

Диаметры НЕ являются входными данными: вход — G, v, c_a, λ, n.
Условие D_вт = const обеспечивается подбором λ_i (или D_ср) по ступеням.
"""
from __future__ import annotations

import math

from constraints import rho_from_lD, uc0_opt_from_rho


def size_section(G: float, v: float, c_a: float, lam: float) -> dict:
    """A = Gv/c_a; D = √(A/(πλ)); l = λD; D_вт = D−l; D_нар = D+l."""
    if G <= 0 or v <= 0 or c_a <= 0 or lam <= 0:
        raise ValueError("G, v, c_a, λ должны быть > 0")
    A = G * v / c_a
    D = math.sqrt(A / (math.pi * lam))
    l = lam * D
    return dict(A=A, D_mean=D, l=l, D_hub=D - l, D_tip=D + l)


def size_stage(stage: dict, lam: float, n: float) -> dict:
    """Размеры ступени по её газовому состоянию.

    stage: G_c, v1, c1a (СА); G_л, v2, c2a (РК); H0.
    λ относится к рабочей лопатке (РК, выход); средний диаметр общий для СА/РК.
    """
    A2 = stage["G_л"] * stage["v2"] / stage["c2a"]  # выход РК
    g = size_section(stage["G_л"], stage["v2"], stage["c2a"], lam)
    D = g["D_mean"]
    l_l = g["l"]
    # СА: та же D_ср, длина из неразрывности
    A1 = stage["G_c"] * stage["v1"] / stage["c1a"]
    l_c = A1 / (math.pi * D)
    u = math.pi * D * n / 60.0
    c0 = math.sqrt(2.0 * stage["H0"])
    rho = rho_from_lD(lam)
    uc0 = u / c0
    return dict(
        lam=lam, A1=A1, A2=A2, D_mean=D, l_c=l_c, l_l=l_l,
        D_hub_c=D - l_c, D_hub_l=D - l_l,
        D_tip_c=D + l_c, D_tip_l=D + l_l,
        u=u, c0=c0, rho=rho, uc0=uc0, uc0_opt=uc0_opt_from_rho(rho),
    )


def solve_lam_for_hub(A2: float, D_hub: float) -> float:
    """λ, при котором D_ср − l = D_hub для заданной площади A2.

    D_ср = (D_hub + √(D_hub² + 4A2/π))/2;  l = D_ср − D_hub;  λ = l/D_ср.
    """
    if A2 <= 0 or D_hub <= 0:
        raise ValueError("A2 и D_hub должны быть > 0")
    D = 0.5 * (D_hub + math.sqrt(D_hub * D_hub + 4.0 * A2 / math.pi))
    l = D - D_hub
    return l / D


def solve_hub_for_lam(A2: float, lam: float) -> float:
    """D_вт при заданных A2 и λ."""
    D = math.sqrt(A2 / (math.pi * lam))
    return D - lam * D


def common_hub(stages: list[dict], n: float, target: float | None = None) -> tuple[list[dict], float]:
    """Подбор λ_i так, чтобы D_вт,i совпали.

    target=None — берётся D_вт последней ступени (не вход, а результат).
    Возвращает (список размеров, target).
    """
    if target is None:
        s_last = stages[-1]
        A2_last = s_last["G_л"] * s_last["v2"] / s_last["c2a"]
        lam_last = s_last["l_л"] / s_last["D_л"]
        target = solve_hub_for_lam(A2_last, lam_last)
    out = []
    for s in stages:
        A2 = s["G_л"] * s["v2"] / s["c2a"]
        lam = solve_lam_for_hub(A2, target)
        out.append(size_stage(s, lam, n))
    return out, target
