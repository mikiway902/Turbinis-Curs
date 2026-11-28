# -*- coding: utf-8 -*-
"""
Расчёт параметров цикла турбовинтового двигателя (ТВД).

Методика, формулы и диапазоны значений: Михальцев В.Е., Моляков В.Д.
«Расчёт параметров цикла при проектировании газотурбинных двигателей
и комбинированных установок», МГТУ им. Н.Э. Баумана, 2014.
В комментариях «с. N» — страница этого пособия.

Задание: N = 1000 кВт, Tг* = 1300 К.
От прототипа ТВД-20 взята только схема:
    осевой компрессор, осевая турбина, камера без регенератора,
    двухвальная компоновка со свободной турбиной и выводом мощности на винт.

Точка расчёта — стендовые условия: H = 0, v = 0 (с. 10, с. 23).

ВАРИАНТ 2: политропный КПД турбин в цикле не берётся из диапазона пособия,
а подбирается так, чтобы суммарная работа турбин в цикле совпала с расчётом
по ступеням (увязка КПД, с. 7 и 19). Результаты пишутся в файлы с суффиксом _v2.
"""
import sys
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

KCAL_TO_KJ = 4.187  # с. 57
ALPHA_AIR = math.inf  # чистый воздух в полиномах приложения 6


def mid(a, b):
    """Середина рекомендованного в методичке диапазона."""
    return 0.5 * (a + b)


# ============================== ИСХОДНЫЕ ДАННЫЕ ==============================
@dataclass(frozen=True)
class Params:
    # --- задание ---
    Ne: float = 1000.0              # кВт, эквивалентная мощность (Gв = Ne/Ne_уд, с. 45)
    T_g: float = 1300.0             # К, температура газа перед турбиной

    # --- атмосфера, стенд (с. 10) ---
    T_a: float = 288.0              # К
    p_a: float = 101300.0           # Па

    # --- коэффициенты потерь и КПД (с. 6-9, 25) ---
    sigma_vx: float = mid(0.98, 0.99)       # вход, с. 6
    eta_kp: float = mid(0.89, 0.92)         # политропный КПД ОСЕВОГО компрессора, с. 9
    eta_g: float = mid(0.97, 0.99)          # полнота горения, с. 7
    sigma_g: float = mid(0.96, 0.97)        # камера сгорания, с. 7
    eta_tp: float = mid(0.90, 0.91)         # политропный КПД ОСЕВОЙ турбины, с. 9
    eta_m: float = mid(0.990, 0.995)        # механический, отнесён к Nт, с. 7
    eta_r: float = mid(0.97, 0.99)          # редуктор ТВД, с. 25

    # --- реактивное сопло ТВД (с. 17, 24, 25) ---
    pi_c: float = mid(1.03, 1.05)           # pт/pа, с. 25
    phi_c: float = mid(0.94, 0.96)          # коэффициент скорости сопла ТВД, с. 24
    M_t_flight: float = mid(0.67, 0.77)     # Mт для ТВД в полёте, с. 17
    start_reduction: float = mid(0.15, 0.20)  # снижение на стенде на 15...20 %, с. 17
    beta_v: float = 15.0                    # Н/кВт, тяговый коэффициент винта, с. 24

    # --- утечки и охлаждение (с. 22, 23) ---
    g_ut: float = mid(0.005, 0.010)         # лабиринтные уплотнения, с. 22
    g_ohl: float = 0.0                      # турбины неохлаждаемые (по указанию)
    g_voz: float = 0.0                      # возврата нет, так как нет охлаждения

    # --- топливо: керосин Т-1, ТС-1, Т-2 (табл. 2, с. 21) ---
    Q_n: float = 43000.0                    # кДж/кг
    l0: float = 14.61                       # кг воздуха / кг топлива
    T_0: float = mid(288.0, 290.0)          # К, с. 20; принято Tт ≈ T0

    # --- газовая постоянная: в методичке для воздуха и продуктов сгорания не приведена ---
    R: float = 287.0                        # Дж/(кг·К), нужна для cp = R·k/(k-1)

    # --- расчётная степень повышения давления ---
    # Инженерный выбор между πL и πη по рис. 10, с. 46
    pi_k_design: float = 14.0

    # --- выбор частоты вращения и диаметра (раздел 6, с. 48-51) ---
    K_n: float = 6.8e-6             # F в м², n в об/мин, σ в МПа, с. 48
    sigma_bl_tk: float = 100.0      # МПа, турбина компрессора: наибольшее из 20...100
                                    # для высокотемпературных турбин, с. 49
    sigma_bl_st: float = 200.0      # МПа, свободная турбина: наибольшее из 120...200, с. 49
    l_D_tk: float = 1.0 / 8.0       # снижено для повышения uт при том же n, с. 50
    l_D_st: float = 1.0 / 4.0       # lт/Dт = 1/4...1/3,5 (до 1/3), с. 50
    c_a_inner: float = 250.0        # м/с, за турбиной компрессора: нижняя граница 250...400, с. 16-17


# ============================ ТЕПЛОЁМКОСТИ (прил. 6) ==========================
def cpi(T, alpha):
    """Истинная удельная теплоёмкость, кДж/(кг·К), с. 57."""
    inv = 0.0 if math.isinf(alpha) else 1.0 / alpha
    if T < 750.0:
        third = 0.0 if math.isinf(alpha) else 1.056e-3 / (alpha - 0.2)
        c = (0.0174 * inv + 0.2407) + (0.0093 * inv + 0.0193) * (2.5e-3 * T - 0.875) \
            + (2.0e-3 - third) * (2.5e-5 * T ** 2 - 0.0275 * T + 6.5625)
    else:
        third = 0.0 if math.isinf(alpha) else 0.94e-2 / (alpha ** 2 + 10.0)
        c = (0.0267 * inv + 0.26) + (0.0133 * inv + 0.032) * (1.176e-3 * T - 0.88235) \
            - (0.374e-2 + third) * (5.5556e-6 * T ** 2 - 1.3056e-2 * T + 6.67)
    return c * KCAL_TO_KJ


def cpm(T, alpha):
    """Средняя удельная теплоёмкость, кДж/(кг·К), с. 57."""
    if math.isinf(alpha):
        slope_low, slope_high = 1.2e-5, 2.2e-5
    else:
        slope_low = (2.25 + 1.2 * alpha) / (alpha * 1e5)
        slope_high = (1.25 + 2.2 * alpha) / (alpha * 1e5)
    if T < 700.0:
        c = slope_low * (T - 70.0) + 0.236
    else:
        c = slope_high * (T + 450.0) + 0.218
    return c * KCAL_TO_KJ


def k_from_cp(cp_kj, R):
    """Согласование k и cp: cp = R·k/(k-1), с. 13, 19."""
    cp = cp_kj * 1000.0
    return cp / (cp - R)


# ============================== УЗЛЫ ДВИГАТЕЛЯ ================================
def eta_compressor(pi, k, eta_p):
    """Адиабатный КПД компрессора через политропный, с. 9."""
    m = (k - 1.0) / k
    return (pi ** m - 1.0) / (pi ** (m / eta_p) - 1.0)


def eta_turbine(pi, k, eta_p):
    """Адиабатный КПД турбины через политропный, с. 9."""
    m = (1.0 - k) / k
    return (1.0 - pi ** (m * eta_p)) / (1.0 - pi ** m)


def compressor(pi_k, p):
    """Компрессор: T*к и Lк (с. 12, 25); cp — по средней температуре сжатия."""
    T_v = p.T_a
    T_k = T_v * pi_k ** 0.286
    for _ in range(100):
        cp = cpi(0.5 * (T_v + T_k), ALPHA_AIR)
        k = k_from_cp(cp, p.R)
        eta = eta_compressor(pi_k, k, p.eta_kp)
        T_new = T_v * (1.0 + (pi_k ** ((k - 1.0) / k) - 1.0) / eta)
        if abs(T_new - T_k) < 1e-8:
            T_k = T_new
            break
        T_k = T_new
    L_k = cp * T_v * (pi_k ** ((k - 1.0) / k) - 1.0) / eta  # кДж/кг, с. 25
    return dict(T_k=T_k, L_k=L_k, cp=cp, k=k, eta=eta)


def fuel(T_p, p):
    """Относительный расход топлива g_m и α, с. 20 (T_m ≈ T_0)."""
    def h_air(T):
        return cpm(T, ALPHA_AIR) * T

    def h_clean(T):
        return cpm(T, 1.0) * T

    g_m = (h_air(p.T_g) - h_air(T_p)) / (
        p.Q_n * p.eta_g
        - (1.0 + p.l0) * (h_clean(p.T_g) - h_clean(p.T_0))
        + p.l0 * (h_air(p.T_g) - h_air(p.T_0))
    )
    return g_m, 1.0 / (g_m * p.l0)


def expansion_props(T_in, pi, alpha, R):
    """cp и k расширения: истинная cp при средней температуре адиабатного расширения, с. 16."""
    k = 1.333  # первое приближение, с. 16
    for _ in range(100):
        T_mean = 0.5 * T_in * (1.0 + pi ** ((1.0 - k) / k))
        cp = cpi(T_mean, alpha)
        k_new = k_from_cp(cp, R)
        if abs(k_new - k) < 1e-10:
            return cp, k_new
        k = k_new
    return cp, k


def turbine_total(pi_k, alpha, p):
    """Суммарное расширение в турбинах и параметры сопла на стенде, с. 24-25."""
    M_t = p.M_t_flight * (1.0 - p.start_reduction)  # с. 17
    k = 1.333
    for _ in range(100):
        pi_c_star = p.pi_c * (1.0 + 0.5 * (k - 1.0) * M_t ** 2) ** (k / (k - 1.0))
        pi_t = p.sigma_vx * pi_k * p.sigma_g / pi_c_star
        cp, k_new = expansion_props(p.T_g, pi_t, alpha, p.R)
        if abs(k_new - k) < 1e-10:
            k = k_new
            break
        k = k_new
    eta_t = eta_turbine(pi_t, k, p.eta_tp)
    work_fraction = (1.0 - pi_t ** ((1.0 - k) / k)) * eta_t
    L_t = cp * p.T_g * work_fraction                                # с. 25
    H_c = cp * p.T_g * (1.0 - work_fraction) * (1.0 - pi_c_star ** ((1.0 - k) / k))  # с. 24
    c_c = p.phi_c * math.sqrt(2.0 * H_c * 1000.0)                   # с. 24
    T_t = p.T_g - L_t / cp                                          # с. 19
    return dict(pi_t=pi_t, pi_c_star=pi_c_star, M_t=M_t, cp=cp, k=k, eta=eta_t,
                L_t=L_t, H_c=H_c, c_c=c_c, T_t=T_t)


def compressor_turbine(L_tk, alpha, p):
    """Турбина компрессора: π*т по заданной работе, «Вариант 2», с. 17-19."""
    pi = 2.0
    for _ in range(300):
        cp, k = expansion_props(p.T_g, pi, alpha, p.R)
        eta = eta_turbine(pi, k, p.eta_tp)
        ratio = (1.0 - L_tk / (cp * p.T_g * eta)) ** (k / (k - 1.0))  # с. 19
        new = 1.0 / ratio
        if abs(new - pi) < 1e-9:
            pi = new
            break
        pi = new
    return dict(pi=pi, cp=cp, k=k, eta=eta, T_out=p.T_g - L_tk / cp)


def cycle(pi_k, p):
    """Полный расчёт цикла при заданной π*к."""
    comp = compressor(pi_k, p)
    g_m, alpha = fuel(comp["T_k"], p)                     # без регенератора T*р = T*к, с. 20
    turb = turbine_total(pi_k, alpha, p)

    g_gas = (1.0 + g_m) * (1.0 - p.g_ut - p.g_ohl) + p.g_voz       # с. 22, 25
    N_v_ud = (turb["L_t"] * g_gas * p.eta_m - comp["L_k"]) * p.eta_r  # с. 25
    R_ud = g_gas * turb["c_c"]                                     # с. 24, v = 0
    Ne_ud = N_v_ud + R_ud / p.beta_v                               # с. 23
    Ce = g_m * (1.0 - p.g_ut - p.g_ohl) * 3600.0 / Ne_ud           # с. 43
    eta_e = 3600.0 / (Ce * p.Q_n)                                  # с. 43
    G_v = p.Ne / Ne_ud                                             # с. 45
    return dict(pi_k=pi_k, comp=comp, turb=turb, g_m=g_m, alpha=alpha, g_gas=g_gas,
                N_v_ud=N_v_ud, R_ud=R_ud, Ne_ud=Ne_ud, Ce=Ce, eta_e=eta_e, G_v=G_v)


def split_turbines(r, p):
    """Разделение работы: турбина компрессора приводит компрессор, остаток — свободная турбина."""
    c, t = r["comp"], r["turb"]
    p_g = p.p_a * p.sigma_vx * r["pi_k"] * p.sigma_g
    L_tk = c["L_k"] / (r["g_gas"] * p.eta_m)                       # с. 17
    tk = compressor_turbine(L_tk, r["alpha"], p)
    L_st = t["L_t"] - L_tk
    pi_st = t["pi_t"] / tk["pi"]
    cp_st, k_st = expansion_props(tk["T_out"], pi_st, r["alpha"], p.R)
    eta_st = eta_turbine(pi_st, k_st, p.eta_tp)
    return dict(p_g=p_g, L_tk=L_tk, tk=tk, L_st=L_st, pi_st=pi_st, p_tk=p_g / tk["pi"],
                p_t=p_g / t["pi_t"], cp_st=cp_st, k_st=k_st, eta_st=eta_st)


def section_by_velocity(T_star, p_star, c_a, cp, k, G, R):
    """Статические параметры и площадь сечения при заданной осевой скорости, с. 48."""
    T = T_star - c_a ** 2 / (2.0 * cp * 1000.0)
    ps = p_star * (T / T_star) ** (k / (k - 1.0))
    v = R * T / ps
    return dict(T=T, p=ps, v=v, c=c_a, F=G * v / c_a)


def n_max(sigma, F, K_n):
    """Максимальная частота вращения, об/мин, формула (*), с. 48."""
    return math.sqrt(sigma / (K_n * F))


def geometry(F, l_D, n):
    """Средний диаметр, высота лопатки, окружная скорость, с. 49-50."""
    D = math.sqrt(F / (math.pi * l_D))
    return dict(D=D, l=l_D * D, u=math.pi * D * n / 60.0)


def section6(r, p):
    """Выбор частоты вращения и диаметра турбин двухвальной схемы, раздел 6."""
    t = r["turb"]
    s = split_turbines(r, p)
    G = r["g_gas"] * r["G_v"]

    # Свободная турбина, последняя ступень: pт = pа·πс, cт = Mт·aт (с. 17, 25)
    k = t["k"]
    T_last = t["T_t"] / (1.0 + 0.5 * (k - 1.0) * t["M_t"] ** 2)
    p_last = p.p_a * p.pi_c
    a_last = math.sqrt(k * p.R * T_last)
    c_last = t["M_t"] * a_last
    v_last = p.R * T_last / p_last
    last = dict(T=T_last, p=p_last, v=v_last, c=c_last, F=G * v_last / c_last, a=a_last)
    n_st = n_max(p.sigma_bl_st, last["F"], p.K_n)
    geo_st = geometry(last["F"], p.l_D_st, n_st)

    # Турбина компрессора, выходное сечение
    tk = s["tk"]
    tk_exit = section_by_velocity(tk["T_out"], s["p_tk"], p.c_a_inner, tk["cp"], tk["k"], G, p.R)
    n_tk = n_max(p.sigma_bl_tk, tk_exit["F"], p.K_n)
    geo_tk = geometry(tk_exit["F"], p.l_D_tk, n_tk)

    # Первая ступень двухступенчатой свободной турбины (проверка, с. 51):
    # работа между ступенями разделена поровну
    L1 = 0.5 * s["L_st"]
    T1 = tk["T_out"] - L1 / s["cp_st"]
    p1 = s["p_tk"] * (1.0 - L1 / (s["cp_st"] * tk["T_out"] * s["eta_st"])) ** (s["k_st"] / (s["k_st"] - 1.0))
    first = section_by_velocity(T1, p1, p.c_a_inner, s["cp_st"], s["k_st"], G, p.R)
    sigma_first = p.K_n * n_st ** 2 * first["F"]

    return dict(G=G, last=last, n_st=n_st, geo_st=geo_st, tk_exit=tk_exit, n_tk=n_tk,
                geo_tk=geo_tk, first=first, T1_star=T1, sigma_first=sigma_first, split=s)


def section6_report(r, p):
    d = section6(r, p)
    last, tk_exit, first = d["last"], d["tk_exit"], d["first"]
    K_u = p.K_n * 60.0 ** 2 / math.pi                              # с. 50
    L = []
    L.append("\n=== ЧАСТОТА ВРАЩЕНИЯ И ДИАМЕТР (раздел 6) ===")
    L.append(f"  Kn = {p.K_n:.2e};  Ku = Kn·60²/π = {K_u:.3e};  Gг = {d['G']:.3f} кг/с")
    head = f"  {'':<34}{'Турбина компрессора':>22}{'Свободная турбина':>22}"
    L.append(head)
    L.append("  " + "-" * (len(head) - 2))
    rows = [
        ("Допускаемое напряжение σл, МПа", f"{p.sigma_bl_tk:.0f}", f"{p.sigma_bl_st:.0f}"),
        ("Отношение lт/Dт", f"{p.l_D_tk:.3f}", f"{p.l_D_st:.3f}"),
        ("Статическая температура, К", f"{tk_exit['T']:.1f}", f"{last['T']:.1f}"),
        ("Статическое давление, кПа", f"{tk_exit['p'] / 1000:.2f}", f"{last['p'] / 1000:.2f}"),
        ("Удельный объём vт, м³/кг", f"{tk_exit['v']:.4f}", f"{last['v']:.4f}"),
        ("Осевая скорость cт.а, м/с", f"{tk_exit['c']:.1f}", f"{last['c']:.1f}"),
        ("Площадь Fт, м²", f"{tk_exit['F']:.5f}", f"{last['F']:.5f}"),
        ("Частота nт max, об/мин", f"{d['n_tk']:.0f}", f"{d['n_st']:.0f}"),
        ("Средний диаметр Dт, м", f"{d['geo_tk']['D']:.4f}", f"{d['geo_st']['D']:.4f}"),
        ("Высота лопатки lт, м", f"{d['geo_tk']['l']:.4f}", f"{d['geo_st']['l']:.4f}"),
        ("Окружная скорость uт, м/с", f"{d['geo_tk']['u']:.1f}", f"{d['geo_st']['u']:.1f}"),
    ]
    for name, a, b in rows:
        L.append(f"  {name:<34}{a:>22}{b:>22}")

    L.append("\n  Проверка первой ступени свободной турбины (две ступени, работа поровну):")
    L.append(f"    T* = {d['T1_star']:.1f} К;  Fт = {first['F']:.5f} м²;  "
             f"σл = Kn·n²·F = {d['sigma_first']:.1f} МПа при n = {d['n_st']:.0f} об/мин")

    L.append("\n  Влияние допускаемого напряжения на частоту, об/мин:")
    L.append(f"    {'σл, МПа':>8}{'Турбина компрессора':>24}{'Свободная турбина':>22}")
    for sigma in (100.0, 120.0, 160.0, 200.0):
        L.append(f"    {sigma:8.0f}{n_max(sigma, tk_exit['F'], p.K_n):24.0f}"
                 f"{n_max(sigma, last['F'], p.K_n):22.0f}")

    L.append("\n  Турбина компрессора: увеличение uт снижением lт/Dт при том же n (с. 50):")
    L.append(f"    {'lт/Dт':>8}{'Dт, м':>10}{'lт, мм':>10}{'uт, м/с':>10}")
    for l_D in (1 / 4, 1 / 6, 1 / 8, 1 / 10):
        g = geometry(tk_exit["F"], l_D, d["n_tk"])
        L.append(f"    {l_D:8.3f}{g['D']:10.4f}{g['l'] * 1000:10.1f}{g['u']:10.1f}")
    return "\n".join(L)


# =========================== РАСЧЁТ ПРОТОЧНОЙ ЧАСТИ ===========================
# Методика: Михальцев В.Е., Моляков В.Д. «Теория и проектирование газовой турбины»,
# ч. 1, МГТУ им. Н.Э. Баумана, 2006. Ссылки «ТГ с. N» и «п. N» — на это пособие.
@dataclass(frozen=True)
class StageConstants:
    phi: float = mid(0.95, 0.97)        # коэффициент скорости СА, ТГ с. 34
    psi: float = 0.96                   # РК: числа в пособии нет, принято ψ = φ (ТГ с. 40)
    mu: float = 1.0                     # использование выходной скорости: числа нет, принято 1
    eta_l: float = mid(0.88, 0.92)      # лопаточный КПД транспортных ГТД, ТГ с. 45
    alpha_11: float = 14.0              # угол за СА первой ступени ТК, град: для турбин небольшой
                                        # мощности 11...14° ради длины лопаток (ТГ с. 35); обычно 16...17°
    k_n: float = 7.34e-6                # ТГ с. 56, формула п. 1217: kn=2*pi*rho_м/(3600*(1+ln mu));
                                        # пересчитано под плотность лопаток ТК из ЖС26 (8530 кг/м3
                                        # вместо 7900 по методичке), kn_new=6.8e-6*(8530/7900).
                                        # На СТ (материал ЭИ437Б/ЭИ617/ЭИ698, легче ЖС26) это не влияет:
                                        # её геометрия определяется оптимумом u/c0, а не пределом σл
                                        # (проверено: σ_доп от 200 до 428 МПа даёт одинаковый результат).
    delta_work: float = 0.5e-3          # рабочий радиальный зазор по заданию руководителя (0,5 мм,
                                        # без сотовых вставок); бандаж на РК с лабиринтным уплотнением
                                        # учтён отдельно через bandage_credit (ТГ с. 40-41).
    gap: float = mid(0.20, 0.25)        # осевой зазор δa/b_a, ТГ с. 62
    r1_ratio: float = mid(0.03, 0.04)   # радиус входной кромки r1/b_a, ТГ с. 80
    t_b_min: float = 0.70               # оптимальный относительный шаг, ТГ с. 82
    t_b_max: float = 0.85
    bandage_credit: float = 0.0275       # бандаж на РК с лабиринтным уплотнением (решение руководителя):
                                        # прибавка к η (= вычет из ξз) 2,5...3%, среднее 0,0275, ТГ с. 40-41.
                                        # Формулы в пособии нет, это прямое прочтение качественной оценки
                                        # методички как поправки к ξз. Без лабиринта было бы 0,0175.


@dataclass(frozen=True)
class TurbineConfig:
    name: str
    z: int                  # число ступеней (выбрано при обсуждении, п. 54 без чисел неприменим)
    sigma_bl: float         # МПа
    elong_c: float          # удлинение (l/b_a) сопловых лопаток, ТГ с. 61
    elong_l: float          # удлинение (l/b_a) рабочих лопаток, ТГ с. 61
    cooling: tuple          # вид ступени по таблице ТГ с. 80, по числу ступеней
    gamma_target: float = None  # суммарный угол раскрытия, град (0...20, ТГ с. 56);
                                # None: длина за СА1 определяется по α11


TK_CONFIG = TurbineConfig("Турбина компрессора", 2, 150.0, 1.5, 2.0,
                          ("неохл.", "неохл."))
# Удлинение l/b_a взято по верхней границе диапазона методички (ТГ с. 61: 1,0...1,5 для СА и
# 1,5...2,0 для РК при больших осевых усилиях) — уже лопатка, меньше концевые потери, профильные
# почти не растут; ограничение на жёсткость/крепление лопатки методичка даёт только словесно.
# σ_доп=150 МПа вместо верхней границы методички (100): материал ст.1 — ЖС26, допустимо по
# длительной прочности ~158 МПа на 10000 ч при 915К (запас x1,14, Каблов и др., Авиационные
# материалы и технологии №2, 2015, с.14-25). Ст.2 (ЭИ698, 770°C) при этом выходит на ~150 МПа
# против собственного предела ~158 МПа (запас x1,05) — самое нагруженное место расчёта.
ST_CONFIG = TurbineConfig("Свободная турбина", 2, 200.0, 2.5, 3.0,
                          ("неохл.", "неохл."), gamma_target=mid(0.0, 20.0))
TRAILING_EDGE = {"первая охл.": (0.11, 0.17), "вторая охл.": (0.07, 0.10),
                 "неохл.": (0.055, 0.07)}   # s2/a для СА и РК, ТГ с. 80


RHO_LIMIT = 0.45    # верхняя граница ρ на среднем радиусе, для которой дано оптимальное u/c0 (ТГ с. 58)


def rho_by_lD(l_D):
    """Степень реактивности на среднем радиусе по таблице ТГ с. 58."""
    return float(np.interp(l_D, [1 / 9, 1 / 6, 1 / 4, 1 / 3], [0.2, 0.3, 0.4, 0.5]))


def u_c0_by_rho(rho):
    """Оптимальное u/c0 = 0,55...0,6 при ρ = 0,25...0,45, ТГ с. 58."""
    return float(np.interp(rho, [0.25, 0.45], [0.55, 0.60]))


def gas_props(T, alpha, R):
    """cp, Дж/(кг·К), и k для заданной средней температуры (ТГ с. 15, 24)."""
    cp = cpi(T, alpha) * 1000.0
    return cp, cp / (cp - R)


def turbine_by_work(T0, p0, L, c_a, eta_l, alpha, R):
    """Определение параметров при заданной работе, п. 13-22."""
    H = (L + 0.5 * c_a ** 2) / eta_l
    T_t = T0 * 0.8
    for _ in range(100):
        cp, k = gas_props(0.5 * (T0 + T_t), alpha, R)
        T_new = T0 - H * eta_l / cp
        if abs(T_new - T_t) < 1e-8:
            T_t = T_new
            break
        T_t = T_new
    p_t = p0 * (1.0 - H / (cp * T0)) ** (k / (k - 1.0))
    return dict(H=H, T_t=T_t, p_t=p_t, cp=cp, k=k, c_a=c_a, pi_t=p0 / p_t)


def turbine_by_pressure(T0, p0, p_t, M_t, eta_l, alpha, R):
    """Определение параметров при заданном pт, п. 1-9; cт = Mт·aт (ТГ с. 56)."""
    pi_t = p0 / p_t
    T_t = T0 * pi_t ** -0.25
    for _ in range(100):
        cp, k = gas_props(0.5 * (T0 + T_t), alpha, R)
        H = cp * T0 * (1.0 - pi_t ** ((1.0 - k) / k))
        T_new = T0 - H * eta_l / cp
        if abs(T_new - T_t) < 1e-8:
            T_t = T_new
            break
        T_t = T_new
    c_a = M_t * math.sqrt(k * R * T_t)
    return dict(H=H, T_t=T_t, p_t=p_t, cp=cp, k=k, c_a=c_a, pi_t=pi_t)


def turbine_size(pre, cfg, G, sp, R):
    """n max (п. 11) и средний диаметр по оптимальному u/c0 (ТГ с. 58-59)."""
    v_t = R * pre["T_t"] / pre["p_t"]
    A = G * v_t / pre["c_a"]
    n = math.sqrt(cfg.sigma_bl / (sp.k_n * A))
    c0 = math.sqrt(2.0 * pre["H"] / cfg.z)
    l_D = 0.25
    for _ in range(100):
        rho = rho_by_lD(l_D)
        u = u_c0_by_rho(rho) * c0
        D = 60.0 * u / (math.pi * n)
        l = A / (math.pi * D)
        new = l / D
        if abs(new - l_D) < 1e-9:
            l_D = new
            break
        l_D = new
    return dict(A=A, v_t=v_t, n=n, c0=c0, u=u, D=D, l=l, l_D=l_D, rho=rho, u_c0=u / c0)


def stage_calc(T0, p0, H0, l_c, l_l, D_c, D_l, n, G, alpha, rho, sp, R, p2_fixed=None):
    """Ступень на среднем диаметре, п. 71-129 (постоянный корневой диаметр, u1 ≠ u2)."""
    phi, psi = sp.phi, sp.psi
    cp, k = gas_props(T0 - 0.05 * T0, alpha, R)
    for _ in range(100):
        m = (k - 1.0) / k
        if p2_fixed is not None:
            H0 = cp * T0 * (1.0 - (p2_fixed / p0) ** m)         # п. 140
        T_y = T0 - H0 * phi ** 2 / (2.0 * cp)                   # п. 75
        cp_new, k_new = gas_props(T_y, alpha, R)                # п. 76, замечание 4
        converged = abs(cp_new - cp) < 1e-7
        cp, k = cp_new, k_new
        if converged:
            break
    m = (k - 1.0) / k

    H_c = H0 * (1.0 - rho)                                      # п. 72
    c1 = phi * math.sqrt(2.0 * H_c)                             # п. 74
    T1 = T0 - H_c * phi ** 2 / cp                               # п. 77
    T1s = T0 - H_c / cp                                         # п. 78
    p1 = p0 * (T1s / T0) ** (1.0 / m)                           # п. 79
    A1 = math.pi * D_c * l_c                                    # п. 80
    v1 = R * T1 / p1                                            # п. 81
    c1a = G * v1 / A1                                           # п. 82
    alpha1 = math.asin(min(c1a / c1, 1.0))                      # п. 83
    c1u = c1 * math.cos(alpha1)
    u1 = math.pi * D_c * n / 60.0                               # п. 87
    u2 = math.pi * D_l * n / 60.0                               # п. 88
    w1 = math.sqrt(c1 ** 2 + u1 ** 2 - 2.0 * c1 * u1 * math.cos(alpha1))  # п. 84
    beta1 = math.atan2(c1a, c1u - u1)                           # п. 85

    if p2_fixed is not None:
        H_l = cp * T1 * (1.0 - (p2_fixed / p1) ** m)            # п. 141
    else:
        H_l = rho * H0 * T1 / T1s                               # п. 86
    w2 = psi * math.sqrt(w1 ** 2 + 2.0 * H_l + u2 ** 2 - u1 ** 2)   # п. 90
    T2 = T1 - (w2 ** 2 - w1 ** 2 - u2 ** 2 + u1 ** 2) / (2.0 * cp)  # п. 91
    T2s = T1 - H_l / cp                                         # п. 92
    p2 = p1 * (T2s / T1) ** (1.0 / m)                           # п. 93
    v2 = R * T2 / p2                                            # п. 94
    A2 = math.pi * D_l * l_l                                    # п. 95
    c2a = G * v2 / A2                                           # п. 96
    beta2 = math.asin(min(c2a / w2, 1.0))                       # п. 97
    c2u = w2 * math.cos(beta2) - u2                             # п. 98
    alpha2 = math.atan2(c2a, c2u)                               # п. 99
    c2 = math.hypot(c2a, c2u)                                   # п. 100

    L_u = c1u * u1 + c2u * u2                                   # п. 101
    eta_u = L_u / H0                                            # п. 102
    h_c = (phi ** -2 - 1.0) * c1 ** 2 / 2.0                     # п. 103
    xi_c = h_c / H0
    xi_cs = xi_c * T2s / T1                                     # п. 106
    h_l = (psi ** -2 - 1.0) * w2 ** 2 / 2.0                     # п. 107
    xi_l = h_l / H0
    h_v = c2 ** 2 / 2.0                                         # п. 109
    xi_v = h_v / H0
    eta_u_check = 1.0 - xi_cs - xi_l - xi_v                     # п. 111

    delta = sp.delta_work                                       # п. 113
    h_z = 1.37 * (1.0 + 1.6 * rho) * (1.0 + l_l / D_l) * delta / l_l * L_u   # п. 114
    xi_z = max(h_z / H0 - sp.bandage_credit, 0.0)               # бандаж (см. StageConstants)
    eta_t1 = eta_u - xi_z                                       # п. 117
    eta_l1 = eta_t1 + xi_v                                      # п. 118
    density = 0.5 * (1.0 / v1 + 1.0 / v2)
    N_tv = 1.07 * D_l ** 2 * (u2 / 100.0) ** 3 * density        # п. 119, ε = 1, кВт
    h_tv = N_tv * 1000.0 / G                                    # п. 120
    xi_tv = h_tv / H0
    eta_t = eta_t1 - xi_tv                                      # п. 122
    eta_l = eta_l1 - xi_tv                                      # п. 123
    L_t = H0 * eta_t                                            # п. 124
    T_st = T2 + h_z / cp + h_tv / cp                            # п. 125
    T_st_star = T_st + h_v / cp                                 # п. 126
    p_st_star = p2 * (T_st_star / T_st) ** (1.0 / m)            # п. 127
    H_star = cp * T0 * (1.0 - (p_st_star / p0) ** m)            # п. 128
    eta_star = L_t / H_star                                     # п. 129

    a1 = math.sqrt(k * R * T1)
    a2 = math.sqrt(k * R * T2)
    return dict(T0=T0, p0=p0, H0=H0, rho=rho, rho_real=H_l * T1s / (H0 * T1), l_c=l_c, l_l=l_l,
                D_c=D_c, D_l=D_l, u1=u1, u2=u2, cp=cp, k=k, H_c=H_c, c1=c1, T1=T1, T1s=T1s, p1=p1, v1=v1, c1a=c1a,
                c1u=c1u, alpha1=math.degrees(alpha1), w1=w1, beta1=math.degrees(beta1), H_l=H_l,
                w2=w2, T2=T2, T2s=T2s, p2=p2, v2=v2, c2a=c2a, beta2=math.degrees(beta2), c2u=c2u,
                alpha2=math.degrees(alpha2), c2=c2, L_u=L_u, eta_u=eta_u, eta_u_check=eta_u_check,
                xi_c=xi_cs, xi_l=xi_l, xi_v=xi_v, h_v=h_v, delta=delta, h_z=h_z, xi_z=xi_z,
                eta_t1=eta_t1, eta_l1=eta_l1, N_tv=N_tv, h_tv=h_tv, xi_tv=xi_tv, eta_t=eta_t,
                eta_l=eta_l, L_t=L_t, T_st=T_st, T_st_star=T_st_star, p_st_star=p_st_star,
                H_star=H_star, eta_star=eta_star, M_c1=c1 / a1, M_w2=w2 / a2,
                sigma=None, u_c0=0.5 * (u1 + u2) / math.sqrt(2.0 * H0))


def meridian(l_first, l_last, cfg, sp, previous):
    """Длины по сечениям: линейно по осевой координате, п. 64-68."""
    z = cfg.z
    stations = 2 * z
    lengths = list(previous)
    for _ in range(50):
        x = [0.0]
        geom = []
        for i in range(stations):
            is_rotor = i % 2 == 1
            elong = cfg.elong_l if is_rotor else cfg.elong_c
            b_a = lengths[i] / elong
            geom.append(dict(b_a=b_a, gap=sp.gap * b_a, rotor=is_rotor))
        for i in range(1, stations):
            prev, cur = geom[i - 1], geom[i]
            x.append(x[-1] + prev["gap"] + cur["b_a"])
        new = [l_first + (l_last - l_first) * xi / x[-1] for xi in x]
        done = max(abs(a - b) for a, b in zip(new, lengths)) < 1e-10
        lengths = new
        if done:
            break
    return lengths, geom, x


def annulus_length(A, hub):
    """Длина лопатки при постоянном корневом диаметре: A = π·(Dв + l)·l."""
    return 0.5 * (-hub + math.sqrt(hub ** 2 + 4.0 * A / math.pi))


def run_turbine(cfg, T_in, p_in, G, alpha, size, pre, sp, R, L_target=None, p_t=None, hub_fixed=None,
                n_scale=1.0):
    """Расчёт турбины по ступеням, п. 55-145; корневой диаметр постоянен (как у ТВД-20)."""
    z, n_max, A = cfg.z, size["n"], size["A"]
    n = n_max
    hub = hub_fixed if hub_fixed is not None else size["D"] - size["l"]
    l_last = annulus_length(A, hub)
    lengths = [l_last] * (2 * z)
    rhos = [size["rho"]] * z
    H_first = pre["H"] / z
    for _ in range(800):
        stages = []
        T0, p0 = T_in, p_in
        c_prev, cp_prev, sum_L = None, None, 0.0
        for i in range(z):
            l_c, l_l = lengths[2 * i], lengths[2 * i + 1]
            D_c, D_l = hub + l_c, hub + l_l
            rho = rhos[i]

            def hdrop(H_i):
                if c_prev is None:
                    return H_i
                return H_i * (1.0 + (1.0 - sp.mu ** 2) * c_prev ** 2 / (2.0 * cp_prev * T0)) \
                    + (sp.mu * c_prev) ** 2 / 2.0               # п. 134

            if i < z - 1:
                s = stage_calc(T0, p0, hdrop(H_first), l_c, l_l, D_c, D_l, n, G, alpha, rho, sp, R)
            elif p_t is not None:
                s = stage_calc(T0, p0, None, l_c, l_l, D_c, D_l, n, G, alpha, rho, sp, R, p2_fixed=p_t)
            else:
                need = L_target - sum_L                         # п. 142
                lo, hi = 0.3 * need, 3.0 * need
                for _b in range(100):
                    H0 = 0.5 * (lo + hi)
                    s = stage_calc(T0, p0, H0, l_c, l_l, D_c, D_l, n, G, alpha, rho, sp, R)
                    if s["L_t"] < need:
                        lo = H0
                    else:
                        hi = H0
                    if hi - lo < 1e-5:
                        break
            stages.append(s)
            sum_L += s["L_t"]
            T0 = s["T_st_star"]                                 # п. 130
            m = (s["k"] - 1.0) / s["k"]
            p0 = s["p2"] * (1.0 + (sp.mu * s["c2"]) ** 2 / (2.0 * s["cp"] * s["T_st"])) ** (1.0 / m)  # п. 132
            c_prev, cp_prev = s["c2"], s["cp"]

        first, last = stages[0], stages[-1]
        # Осевой выход из РК: ρ подбирается так, чтобы α2 = 90° (ТГ с. 38, 52, примечание к п. 99)
        new_rhos = [min(max(rho + 0.004 * (s["alpha2"] - 90.0), 0.05), 0.70)
                    for rho, s in zip(rhos, stages)]
        # Равные теплоперепады ступеней H0 (п. 55: формулы в пособии нет)
        new_H_first = H_first + 0.4 * (last["H0"] - first["H0"]) / (z - 1 if z > 1 else 1)
        # Диаметр по оптимальному u/c0 при текущей ρ (ТГ с. 58-59)
        D_eta = 60.0 * u_c0_by_rho(first["rho"]) * math.sqrt(2.0 * first["H0"]) / (math.pi * n_max)
        if hub_fixed is None:
            new_hub, new_n = hub + 0.5 * (D_eta - first["l_l"] - hub), n
        else:
            # Корневой диаметр задан турбиной компрессора: nт = nт max·Dтη/Dт (ТГ с. 60);
            # n_scale < 1 — дополнительное снижение частоты (см. turbine_design)
            new_hub = hub
            new_n = n + 0.5 * (n_scale * min(n_max, n_max * D_eta / first["D_l"]) - n)
        l_last = annulus_length(A, new_hub)

        if cfg.gamma_target is None:
            l_first = G * first["v1"] / (math.pi * first["D_c"] * math.sin(math.radians(sp.alpha_11)) * first["c1"])  # п. 63
            new_lengths, geom, x = meridian(l_first, l_last, cfg, sp, lengths)
        else:
            new_lengths = list(lengths)
            for _m in range(30):
                _l, geom, x = meridian(new_lengths[0], l_last, cfg, sp, new_lengths)
                l_first = l_last - math.tan(math.radians(cfg.gamma_target)) * x[-1]   # γв = 0, γ = γн
                new_lengths, geom, x = meridian(l_first, l_last, cfg, sp, _l)
        change = max(max(abs(a - b) for a, b in zip(new_lengths, lengths)),
                     max(abs(a - b) for a, b in zip(new_rhos, rhos)) * 1e-2,
                     abs(new_H_first - H_first) * 1e-7, abs(new_hub - hub), abs(new_n - n) * 1e-5)
        lengths, rhos, H_first, hub, n = new_lengths, new_rhos, new_H_first, new_hub, new_n
        if change < 2e-8:
            break
    D_last = hub + lengths[-1]
    final_size = dict(size, hub=hub, n=n, n_max=n_max, D=D_last, l=lengths[-1], l_D=lengths[-1] / D_last,
                      rho=stages[0]["rho"], u=stages[-1]["u2"], u_c0=stages[0]["u_c0"],
                      c0=math.sqrt(2.0 * stages[0]["H0"]))
    for s in stages:
        s["sigma"] = sp.k_n * n ** 2 * math.pi * s["D_l"] * s["l_l"]    # формула (63)

    last = stages[-1]
    L_sum = sum(s["L_t"] for s in stages)                        # п. 146
    cp, k = pre["cp"], pre["k"]
    m = (k - 1.0) / k
    H_t = cp * T_in * (1.0 - (last["p2"] / p_in) ** m)           # п. 147
    H_star = cp * T_in * (1.0 - (last["p_st_star"] / p_in) ** m) # п. 150
    totals = dict(L=L_sum, H=H_t, H_star=H_star, eta_t=L_sum / H_t,
                  eta_l=(L_sum + last["h_v"]) / H_t, eta_star=L_sum / H_star,
                  T_out=last["T_st_star"], p_out=last["p_st_star"], p_static=last["p2"],
                  T_static=last["T_st"], c_out=last["c2"])
    return dict(cfg=cfg, stages=stages, lengths=lengths, geom=geom, x=x, totals=totals, size=final_size)


def radial_section(s, relative_r1, relative_r2, R):
    """Закон Γ = const: параметры в сечении, формулы (71)-(82)."""
    k, m = s["k"], (s["k"] - 1.0) / s["k"]
    a1s = math.sqrt(k * R * s["T1s"])
    M1u = s["c1u"] / a1s
    p1 = s["p1"] * (1.0 + 0.5 * (k - 1.0) * M1u ** 2 * (1.0 - relative_r1 ** -2)) ** (k / (k - 1.0))  # (71)
    rho = s["rho_real"] * (p1 ** m - s["p2"] ** m) / (s["p1"] ** m - s["p2"] ** m)        # (74)
    c1u = s["c1u"] / relative_r1                                 # (67)
    alpha1 = math.degrees(math.atan2(s["c1a"], c1u))             # (76)
    u1 = s["u1"] * relative_r1
    beta1 = math.degrees(math.atan2(s["c1a"], c1u - u1))         # (78)
    c2u = s["c2u"] / relative_r2                                 # Γ2u = const
    u2 = s["u2"] * relative_r2
    alpha2 = math.degrees(math.atan2(s["c2a"], c2u))             # (80)
    beta2 = math.degrees(math.atan2(s["c2a"], c2u + u2))         # (82)
    return dict(p1=p1, rho=rho, alpha1=alpha1, beta1=beta1, alpha2=alpha2, beta2=beta2,
                u1=u1, u2=u2, c1=math.hypot(c1u, s["c1a"]))


def flow_deviation(angle, mach):
    """Отставание потока δ, град: кромки тонкие, значения по ТГ с. 78."""
    baseline = 20.0 / 60.0
    if mach >= 0.95:
        return baseline
    if angle < 20.0:
        top, floor = mid(30.0, 50.0) / 60.0, 0.8
    elif angle < 30.0:
        top, floor = mid(1.0, 2.0), 0.6
    else:
        top, floor = mid(2.0, 5.0), 0.6
    return float(np.interp(mach, [floor, 0.95], [top, baseline]))


def blade_rows(result, sp):
    """Геометрия решёток на среднем диаметре, раздел 8."""
    rows = []
    previous = None
    for i, s in enumerate(result["stages"]):
        kind = result["cfg"].cooling[i]
        s_bar = TRAILING_EDGE[kind]
        inlet_sa = 90.0 if i == 0 else result["stages"][i - 1]["alpha2"]
        cooled = kind != "неохл."
        pairs = (("СА", inlet_sa, s["alpha1"], result["geom"][2 * i], s["l_c"], s["M_c1"], 0.45,
                  0.2 if cooled else 0.13, s_bar[0], s["D_c"]),
                 ("РК", s["beta1"], s["beta2"], result["geom"][2 * i + 1], s["l_l"], s["M_w2"], 0.60,
                  0.2 if cooled else 0.11, s_bar[1], s["D_l"]))
        for name, b1, b2, g, l, mach, K, c_max, sb, row_D in pairs:
            diff = b1 - b2
            stagger = 70.0 - 0.127 * diff - 0.0041 * diff ** 2       # ТГ с. 82
            chord = g["b_a"] / math.sin(math.radians(stagger))
            t_b_calc = K * (180.0 / (180.0 - (b1 + b2)) * math.sin(math.radians(b1))
                            / math.sin(math.radians(b2))) ** (1.0 / 3.0) * (1.0 - c_max)
            t_b = min(max(t_b_calc, sp.t_b_min), sp.t_b_max)
            # Число лопаток: t/b остаётся в 0,70...0,85, соседние числа взаимно простые (ТГ с. 82-83).
            # По требованию руководителя: СА — нечётное число, РК — чётное (условие сверх методички).
            target = math.pi * row_D / (t_b * chord)
            lo_n = math.ceil(math.pi * row_D / (sp.t_b_max * chord))
            hi_n = math.floor(math.pi * row_D / (sp.t_b_min * chord))
            parity = 1 if name == "СА" else 0
            in_range = list(range(lo_n, hi_n + 1))
            by_parity = [c for c in in_range if c % 2 == parity]
            options = [c for c in by_parity if previous is None or math.gcd(previous, c) == 1]
            if not options:
                options = by_parity or in_range  # сначала жертвуем взаимной простотой, потом чётностью
            count = min(options, key=lambda c: abs(c - target))
            previous = count
            pitch = math.pi * row_D / count
            dev = flow_deviation(b2, mach)
            throat = pitch * math.sin(math.radians(b2))
            s2 = max(sb * throat, 0.5e-3)                            # не менее 0,5 мм, ТГ с. 80
            rows.append(dict(stage=i + 1, name=name, inlet=b1, exit=b2, b_a=g["b_a"], stagger=stagger,
                             chord=chord, t_b_calc=t_b_calc, t_b=pitch / chord, pitch=pitch, count=count,
                             dev=dev, blade_exit=b2 - dev, throat=throat, s2=s2,
                             r1=sp.r1_ratio * g["b_a"], l=l, mach=mach))
    return rows


def turbine_design(r, p, sp=None):
    """Расчёт обеих турбин: TK по заданной работе, СТ по заданному pт."""
    sp = sp or StageConstants()
    key = (r["pi_k"], replace(p, eta_tp=0.0), sp)        # расчёт по ступеням от η*т.п не зависит
    s = split_turbines(r, p)
    if key in _DESIGN_CACHE:
        return dict(_DESIGN_CACHE[key], L_st_required=s["L_st"], N_v_cycle=r["N_v_ud"] * r["G_v"])
    alpha, R = r["alpha"], p.R
    p_t = p.p_a * p.pi_c
    G_v = r["G_v"]

    # Расход уточняется так, чтобы получить заданную мощность (мощность — контрольный параметр, ТГ с. 5)
    for _ in range(40):
        G = r["g_gas"] * G_v
        tk_pre = turbine_by_work(p.T_g, s["p_g"], s["L_tk"] * 1000.0, p.c_a_inner, sp.eta_l, alpha, R)
        tk_size = turbine_size(tk_pre, TK_CONFIG, G, sp, R)
        tk = run_turbine(TK_CONFIG, p.T_g, s["p_g"], G, alpha, tk_size, tk_pre, sp, R,
                         L_target=s["L_tk"] * 1000.0)

        T_in, p_in = tk["totals"]["T_out"], tk["totals"]["p_out"]
        st_pre = turbine_by_pressure(T_in, p_in, p_t, r["turb"]["M_t"], sp.eta_l, alpha, R)
        st_size = turbine_size(st_pre, ST_CONFIG, G, sp, R)
        # Общий постоянный корневой диаметр обеих турбин, как на разрезе ТВД-20
        def free_turbine(scale):
            return run_turbine(ST_CONFIG, T_in, p_in, G, alpha, st_size, st_pre, sp, R, p_t=p_t,
                               hub_fixed=tk["size"]["hub"], n_scale=scale)

        st = free_turbine(1.0)
        # Если ρ на среднем радиусе выше 0,45, частота снижается до ρ = 0,45 (ТГ с. 58):
        # при меньшей окружной скорости осевой выход получается при меньшей реактивности.
        if max(x["rho"] for x in st["stages"]) > RHO_LIMIT:
            lo, hi = 0.80, 1.0
            for _b in range(16):
                scale = 0.5 * (lo + hi)
                st = free_turbine(scale)
                if max(x["rho"] for x in st["stages"]) > RHO_LIMIT:
                    hi = scale
                else:
                    lo = scale
            st = free_turbine(lo)

        N_v = st["totals"]["L"] / 1000.0 * G * p.eta_m * p.eta_r
        cp, k = st_pre["cp"] / 1000.0, st_pre["k"]
        pi_c_star = st["totals"]["p_out"] / p.p_a
        H_c = cp * st["totals"]["T_out"] * (1.0 - pi_c_star ** ((1.0 - k) / k))   # с. 24 пособия по циклу
        c_c = p.phi_c * math.sqrt(2.0 * H_c * 1000.0)
        thrust = G * c_c
        Ne = N_v + thrust / p.beta_v
        if abs(Ne - p.Ne) < 0.01:
            break
        G_v *= p.Ne / Ne

    out = dict(sp=sp, G=G, G_v=G_v, tk=tk, tk_pre=tk_pre, st=st, st_pre=st_pre,
               thrust=thrust, c_c=c_c, Ne=Ne, N_v=N_v, G_fuel=r["g_m"] * (1.0 - p.g_ut - p.g_ohl) * G_v * 3600.0,
               Ce=r["g_m"] * (1.0 - p.g_ut - p.g_ohl) * G_v * 3600.0 / Ne)
    _DESIGN_CACHE[key] = out
    return dict(out, L_st_required=s["L_st"], N_v_cycle=r["N_v_ud"] * r["G_v"])


_DESIGN_CACHE = {}


def table(title, labels, columns):
    lines = [title]
    for name, values in zip(labels, zip(*columns)):
        lines.append(f"  {name:<40}" + "".join(f"{v:>14}" for v in values))
    return "\n".join(lines)


def design_warnings(result, sp, R):
    notes = []
    cfg, z = result["cfg"], result["cfg"].z
    for i, s in enumerate(result["stages"], 1):
        if s["alpha1"] < 11.0:
            notes.append(f"ступень {i}: α1 = {s['alpha1']:.1f}° ниже 11° (ТГ с. 35)")
        r_c, r_l = s["D_c"] / 2.0, s["D_l"] / 2.0
        root = radial_section(s, (r_c - s["l_c"] / 2.0) / r_c, (r_l - s["l_l"] / 2.0) / r_l, R)
        if root["rho"] < 0.05:
            notes.append(f"ступень {i}: ρ у корня = {root['rho']:.3f} ниже 0,05...0,1 (ТГ с. 58)")
        if root["alpha1"] < 11.0:
            notes.append(f"ступень {i}: α1 у корня = {root['alpha1']:.1f}° ниже 11...12° (ТГ с. 73)")
        low = 80.0 if i == z else 70.0
        alpha2 = round(s["alpha2"], 2)
        if alpha2 < low:
            notes.append(f"ступень {i}: α2 = {alpha2:.1f}° вне {low:.0f}...90° (ТГ с. 38)")
        if alpha2 > 90.0:
            notes.append(f"ступень {i}: α2 = {alpha2:.1f}° > 90°, отрицательная закрутка (ТГ с. 38)")
        if s["u2"] > 400.0:
            notes.append(f"ступень {i}: u2 = {s['u2']:.0f} м/с выше 360...400 м/с (замечание 1); "
                         f"для коротких лопаток допустимо 400...450 м/с (ТГ с. 59)")
        if abs(s["eta_u"] - s["eta_u_check"]) > 0.005:
            notes.append(f"ступень {i}: ηu по п. 102 и п. 111 расходятся на "
                         f"{abs(s['eta_u'] - s['eta_u_check']):.4f}")
    for i in range(1, len(result["x"])):
        dx = result["x"][i] - result["x"][i - 1]
        dl = result["lengths"][i] - result["lengths"][i - 1]
        gamma = math.degrees(math.atan(dl / dx))
        if gamma > 20.0:
            notes.append(f"сечения {i}-{i + 1}: суммарный угол раскрытия γ = {gamma:.1f}° выше 20° (ТГ с. 56)")
    return notes


def turbine_report(r, p):
    d = turbine_design(r, p)
    sp = d["sp"]
    out = ["\n\n=============== РАСЧЁТ ПРОТОЧНОЙ ЧАСТИ ТУРБИН ===============",
           "Заменяет предварительные n, D и Gв из разделов выше: здесь учтены потери по ступеням."]
    out.append(f"φ = {sp.phi:.3f}; ψ = {sp.psi:.3f}; μ = {sp.mu:.2f}; ηл = {sp.eta_l:.3f}; "
               f"α11 ТК = {sp.alpha_11:.1f}°; γ СТ = {ST_CONFIG.gamma_target:.0f}°")
    out.append(f"Постоянный корневой диаметр обеих турбин (как на разрезе ТВД-20): "
               f"Dв = {d['tk']['size']['hub'] * 1000:.1f} мм")
    out.append(f"Расход уточнён по мощности: Gв = {d['G_v']:.3f} кг/с (по циклу {r['G_v']:.3f}); "
               f"G газа = {d['G']:.3f} кг/с, постоянен по ступеням")
    for key, pre_key in (("tk", "tk_pre"), ("st", "st_pre")):
        res, pre = d[key], d[pre_key]
        size = res["size"]
        cfg = res["cfg"]
        out.append(f"\n########## {cfg.name.upper()}: z = {cfg.z} ##########")
        out.append("Предварительный расчёт (п. 1-25, 45-47):")
        out.append(f"  Hт = {pre['H'] / 1000:.2f} кДж/кг; πт = {pre['pi_t']:.3f}; cp = {pre['cp']:.1f}; "
                   f"k = {pre['k']:.4f}; Tт = {pre['T_t']:.1f} К; pт = {pre['p_t'] / 1000:.2f} кПа")
        out.append(f"  cт.а = {pre['c_a']:.1f} м/с; vт = {size['v_t']:.4f} м³/кг; Aт = {size['A']:.5f} м²; "
                   f"σл = {cfg.sigma_bl:.0f} МПа; nт max = {size['n_max']:.0f}; nт = {size['n']:.0f} об/мин")
        out.append(f"  Dв = {size['hub'] * 1000:.1f} мм; c0 ступени = {size['c0']:.1f} м/с; u/c0 ступени 1 = {size['u_c0']:.3f}; "
                   f"на выходе: D = {size['D']:.4f} м; u = {size['u']:.1f} м/с; "
                   f"lт = {size['l'] * 1000:.1f} мм; l/D = 1/{1 / size['l_D']:.1f}; "
                   f"ρ1 по таблице с. 58 = {rho_by_lD(size['l_D']):.3f}")

        stages = res["stages"]
        rows = [
            ("T0*, К", "T0", "{:.1f}"), ("p0*, кПа", "p0", "{:.1f}", 1e-3), ("H0, кДж/кг", "H0", "{:.2f}", 1e-3),
            ("ρ на среднем радиусе", "rho", "{:.3f}"), ("u/c0", "u_c0", "{:.3f}"),
            ("Dс, мм", "D_c", "{:.1f}", 1e3), ("Dл, мм", "D_l", "{:.1f}", 1e3),
            ("u1, м/с", "u1", "{:.1f}"), ("u2, м/с", "u2", "{:.1f}"),
            ("lс, мм", "l_c", "{:.1f}", 1e3), ("lл, мм", "l_l", "{:.1f}", 1e3),
            ("cp, Дж/(кг·К)", "cp", "{:.1f}"), ("k", "k", "{:.4f}"),
            ("c1, м/с", "c1", "{:.1f}"), ("Mc1", "M_c1", "{:.3f}"), ("T1, К", "T1", "{:.1f}"),
            ("p1, кПа", "p1", "{:.1f}", 1e-3), ("v1, м³/кг", "v1", "{:.4f}"),
            ("c1a, м/с", "c1a", "{:.1f}"), ("c1u, м/с", "c1u", "{:.1f}"), ("α1, град", "alpha1", "{:.2f}"),
            ("w1, м/с", "w1", "{:.1f}"), ("β1, град", "beta1", "{:.2f}"),
            ("Hл, кДж/кг", "H_l", "{:.2f}", 1e-3), ("w2, м/с", "w2", "{:.1f}"), ("Mw2", "M_w2", "{:.3f}"),
            ("T2, К", "T2", "{:.1f}"), ("p2, кПа", "p2", "{:.1f}", 1e-3), ("v2, м³/кг", "v2", "{:.4f}"),
            ("c2a, м/с", "c2a", "{:.1f}"), ("β2, град", "beta2", "{:.2f}"), ("c2u, м/с", "c2u", "{:.1f}"),
            ("α2, град", "alpha2", "{:.2f}"), ("c2, м/с", "c2", "{:.1f}"),
            ("Lu, кДж/кг", "L_u", "{:.2f}", 1e-3), ("ηu (п. 102)", "eta_u", "{:.4f}"),
            ("ηu (п. 111)", "eta_u_check", "{:.4f}"), ("ξс'", "xi_c", "{:.4f}"), ("ξл", "xi_l", "{:.4f}"),
            ("ξв", "xi_v", "{:.4f}"), ("δл, мм", "delta", "{:.2f}", 1e3), ("ξз", "xi_z", "{:.4f}"),
            ("ξт.в", "xi_tv", "{:.5f}"), ("ηт (мощностной)", "eta_t", "{:.4f}"), ("ηл", "eta_l", "{:.4f}"),
            ("η*т ступени", "eta_star", "{:.4f}"), ("Lт ступени, кДж/кг", "L_t", "{:.2f}", 1e-3),
            ("T*ст за ступенью, К", "T_st_star", "{:.1f}"), ("p*ст за ступенью, кПа", "p_st_star", "{:.1f}", 1e-3),
            ("σл по (63), МПа", "sigma", "{:.1f}"),
        ]
        labels = [row[0] for row in rows]
        columns = []
        for s in stages:
            col = []
            for row in rows:
                scale = row[3] if len(row) > 3 else 1.0
                col.append(row[2].format(s[row[1]] * scale))
            columns.append(col)
        out.append(table("\nПо ступеням на среднем диаметре (п. 70-129):", labels, columns))
        t = res["totals"]
        out.append("\nПараметры турбины (п. 146-151):")
        out.append(f"  LтΣ = {t['L'] / 1000:.2f} кДж/кг; Hт = {t['H'] / 1000:.2f}; H*т = {t['H_star'] / 1000:.2f} кДж/кг")
        out.append(f"  ηт = {t['eta_t']:.4f}; ηл = {t['eta_l']:.4f}; η*т = {t['eta_star']:.4f}")
        out.append(f"  На выходе: T* = {t['T_out']:.1f} К; p* = {t['p_out'] / 1000:.2f} кПа; "
                   f"T = {t['T_static']:.1f} К; p = {t['p_static'] / 1000:.2f} кПа; c2 = {t['c_out']:.1f} м/с")

        out.append("\nМеридиональное сечение (п. 64-69):")
        names = []
        for i in range(cfg.z):
            names += [f"СА{i + 1}", f"РК{i + 1}"]
        for i, name in enumerate(names):
            gamma = ""
            if i:
                dx = res["x"][i] - res["x"][i - 1]
                dl = res["lengths"][i] - res["lengths"][i - 1]
                gamma = f"; γн к предыдущему = {math.degrees(math.atan(dl / dx)):.1f}°"
            g = res["geom"][i]
            out.append(f"  {name}: l = {res['lengths'][i] * 1000:.1f} мм; b_a = {g['b_a'] * 1000:.1f} мм; "
                       f"δa за венцом = {g['gap'] * 1000:.1f} мм; x = {res['x'][i] * 1000:.1f} мм{gamma}")

        out.append("\nПрофилирование по радиусу, Γ = const (раздел 7):")
        out.append(f"  {'ступень':<8}{'сечение':<10}{'r1, мм':>9}{'p1, кПа':>10}{'ρ':>8}{'α1':>8}{'β1':>8}"
                   f"{'r2, мм':>9}{'α2':>8}{'β2':>8}{'u2, м/с':>9}")
        for i, s in enumerate(stages, 1):
            r_c, r_l = s["D_c"] / 2.0, s["D_l"] / 2.0
            for name, sign in (("корень", -1.0), ("средн.", 0.0), ("перифер.", 1.0)):
                r1 = r_c + sign * s["l_c"] / 2.0
                r2 = r_l + sign * s["l_l"] / 2.0
                q = radial_section(s, r1 / r_c, r2 / r_l, p.R)
                out.append(f"  {i:<8}{name:<10}{r1 * 1000:9.2f}{q['p1'] / 1000:10.1f}{q['rho']:8.3f}{q['alpha1']:8.2f}"
                           f"{q['beta1']:8.2f}{r2 * 1000:9.2f}{q['alpha2']:8.2f}{q['beta2']:8.2f}{q['u2']:9.1f}")

        out.append("\nРешётки на среднем диаметре (раздел 8):")
        out.append(f"  {'венец':<6}{'вход°':>8}{'выход°':>8}{'b_a,мм':>8}{'γуст°':>8}{'b,мм':>8}{'t/b расч':>10}"
                   f"{'t/b':>8}{'t,мм':>8}{'Z':>5}{'δ°':>7}{'угол л.°':>10}{'a,мм':>8}{'s2,мм':>8}{'r1,мм':>8}")
        for row in blade_rows(res, sp):
            label = f"{row['name']}{row['stage']}"
            out.append(f"  {label:<6}{row['inlet']:8.2f}{row['exit']:8.2f}{row['b_a'] * 1e3:8.2f}{row['stagger']:8.2f}"
                       f"{row['chord'] * 1e3:8.2f}{row['t_b_calc']:10.3f}{row['t_b']:8.3f}{row['pitch'] * 1e3:8.2f}"
                       f"{row['count']:5d}{row['dev']:7.2f}{row['blade_exit']:10.2f}{row['throat'] * 1e3:8.2f}"
                       f"{row['s2'] * 1e3:8.2f}{row['r1'] * 1e3:8.2f}")

        notes = design_warnings(res, sp, p.R)
        out.append("\nЗамечания:" if notes else "\nЗамечаний по допустимым пределам нет.")
        out += [f"  {n}" for n in notes]

    out.append("\n########## БАЛАНС МОЩНОСТИ ##########")
    out.append(f"  Работа свободной турбины: по ступеням {d['st']['totals']['L'] / 1000:.2f} кДж/кг; "
               f"по циклу {d['L_st_required']:.2f} кДж/кг")
    out.append(f"  Мощность на винте: {d['N_v']:.1f} кВт (по циклу {d['N_v_cycle']:.1f} кВт)")
    out.append(f"  Скорость истечения из сопла: {d['c_c']:.1f} м/с; тяга сопла: {d['thrust']:.1f} Н")
    out.append(f"  Эквивалентная мощность: {d['Ne']:.1f} кВт (задано {p.Ne:.0f} кВт)")
    out.append(f"  Расход воздуха: {d['G_v']:.3f} кг/с; расход топлива: {d['G_fuel']:.1f} кг/ч; "
               f"Ce = {d['Ce']:.4f} кг/(кВт·ч)")
    return "\n".join(out)


# Материал, оценка температуры металла (неохлаждаемая лопатка) и предел длительной
# прочности на ресурс 10000 ч, МПа (источники: Масленков С.Б. «Жаропрочные стали и
# сплавы», 1983, с.114/118/120 — ЭИ437Б/ЭИ617/ЭИ698; Каблов и др., Авиационные материалы
# и технологии №2, 2015, с.14-25 — ЖС26, экстраполяция по параметру Ларсона-Миллера).
MATERIALS = {
    ("tk", 0): ("ЖС26", 915, 158),
    ("tk", 1): ("ЭИ698 (ХН73МБТЮ)", 770, 158),
    ("st", 0): ("ЭИ617 (ХН70ВМТЮ) / ЭИ437Б", 670, 282),
    ("st", 1): ("ЭИ437Б (ХН77ТЮР)", 580, 450),
}


def materials_report(r, p):
    d = turbine_design(r, p)
    out = ["\n\n=============== МАТЕРИАЛЫ И РЕСУРС ЛОПАТОК (10000 ч) ===============",
           "Температура металла — оценка для неохлаждаемой лопатки, °C; запас — "
           "отношение предела длительной прочности материала к реальному σл расчёта.",
           f"{'Деталь':<22}{'Материал':<28}{'T,°C':>7}{'σл, МПа':>10}{'σ10000':>9}{'Запас':>8}"]
    for key, label in (("tk", "ТК"), ("st", "СТ")):
        for i, st in enumerate(d[key]["stages"]):
            mat, t_c, cap = MATERIALS[(key, i)]
            sigma = st["sigma"]
            out.append(f"{label + f' ст.{i + 1}':<22}{mat:<28}{t_c:>7}{sigma:>10.1f}{cap:>9}{cap / sigma:>7.2f}x")
    return "\n".join(out)


def range_report(r, p):
    """Сводная проверка результатов по диапазонам обеих методичек (Ц — цикл, ТГ — турбина)."""
    rows = []
    for ok, name, v, lo, hi, src in range_rows(r, p):
        lo_s = "—" if lo is None else f"{lo:.4g}"
        hi_s = "—" if hi is None else f"{hi:.4g}"
        rows.append(f"  {'OK ' if ok else 'ВНЕ'} {name:<46}{v:10.4f}   [{lo_s} ... {hi_s}]  {src}")
    return "\n".join(["\n\n=============== ПРОВЕРКА ПО ДИАПАЗОНАМ МЕТОДИЧЕК ===============",
                      "ВНЕ — значение вне типичного диапазона; Ц — пособие по циклу, ТГ — по турбине."] + rows)


def range_rows(r, p, d=None):
    """Строки проверки: (в диапазоне, название, значение, нижняя граница, верхняя, источник)."""
    d = d or turbine_design(r, p)
    sp = d["sp"]
    rows = []

    def chk(name, v, lo, hi, src):
        ok = (lo is None or v >= lo - 1e-6) and (hi is None or v <= hi + 1e-6)
        rows.append((ok, name, v, lo, hi, src))

    ups = p.T_g / p.T_a
    chk("π*к", r["pi_k"], ups, ups ** 2.4, "Ц с. 12")
    chk("η*к адиабатный", r["comp"]["eta"], 0.82, 0.90, "Ц с. 7")
    chk("η*т, принятый в цикле", r["turb"]["eta"], 0.87, 0.93, "Ц с. 7")
    chk("Mт за свободной турбиной", r["turb"]["M_t"], 0.47, 0.62, "ТГ с. 56")
    for key, name in (("tk", "ТК"), ("st", "СТ")):
        res = d[key]
        cfg, t = res["cfg"], res["totals"]
        chk(f"{name}: π*т (двухвальный ТВД)", res["stages"][0]["p0"] / t["p_out"], 2.5, 3.5, "ТГ с. 6")
        chk(f"{name}: η*т по ступеням", t["eta_star"], 0.87, 0.93, "Ц с. 7")
        chk(f"{name}: ηл турбины", t["eta_l"], 0.88, 0.92, "ТГ с. 45")
        chk(f"{name}: осевая скорость на выходе, м/с", res["stages"][-1]["c2a"], 250.0, 350.0, "ТГ с. 56")
        chk(f"{name}: l/D последней ступени", res["size"]["l_D"], None, 1.0 / 3.0, "ТГ п. 45")
        for i, s in enumerate(res["stages"], 1):
            tag = f"{name} ст. {i}: "
            r_c, r_l = s["D_c"] / 2.0, s["D_l"] / 2.0
            root = radial_section(s, (r_c - s["l_c"] / 2.0) / r_c, (r_l - s["l_l"] / 2.0) / r_l, p.R)
            chk(tag + "ηл ступени", s["eta_l"], 0.88, 0.92, "ТГ с. 45")
            chk(tag + "α1, град", s["alpha1"], 11.0, None, "ТГ с. 35 (малая мощность: 11...14)")
            chk(tag + "α1 у корня, град", root["alpha1"], 11.0, None, "ТГ с. 73 (малая мощность: 11...12)")
            chk(tag + "α2, град", round(s["alpha2"], 2), 80.0 if i == cfg.z else 70.0, 90.0, "ТГ с. 38")
            chk(tag + "ρ у корня", root["rho"], 0.05, None, "ТГ с. 58")
            chk(tag + "ρ средняя", s["rho"], 0.25, 0.45, "ТГ с. 58")
            chk(tag + "u/c0", s["u_c0"], 0.55, 0.60, "ТГ с. 58")
            chk(tag + "u2, м/с", s["u2"], None, 400.0, "ТГ зам. 1 (до 450 — с. 59)")
            chk(tag + "σл, МПа", s["sigma"], None, cfg.sigma_bl, "ТГ с. 57")
        for i in range(1, len(res["x"])):
            gamma = math.degrees(math.atan((res["lengths"][i] - res["lengths"][i - 1])
                                           / (res["x"][i] - res["x"][i - 1])))
            chk(f"{name}: угол раскрытия {i}-{i + 1}, град", gamma, 0.0, 20.0, "ТГ с. 56")
        for row in blade_rows(res, sp):
            tag = f"{name} {row['name']}{row['stage']}: "
            chk(tag + "t/b", row["t_b"], sp.t_b_min, sp.t_b_max, "ТГ с. 82")
            chk(tag + "s2, мм", row["s2"] * 1e3, 0.5, 1.6, "ТГ с. 80")
            chk(tag + "угол выхода (для δ), град", row["exit"], 15.0, 40.0, "ТГ с. 78")
    return rows


def book_optimum_estimates(p):
    """Приближённые оценки πL и πη по формулам с. 12 (итерация по π в l̄)."""
    theta = p.T_g / p.T_a

    def state(pi):
        r = cycle(pi, p)
        kv, kg = r["comp"]["k"], r["turb"]["k"]
        cv, cg = r["comp"]["cp"], r["turb"]["cp"]
        l = cg * (1.0 - pi ** ((1.0 - kg) / kg)) / (cv * (1.0 - pi ** ((1.0 - kv) / kv)))
        return l, kv, r["comp"]["eta"], r["turb"]["eta"]

    def pi_L(pi):
        l, kv, eta_s, eta_p = state(pi)
        return (l * theta * eta_s * eta_p) ** (kv / (2.0 * (kv - 1.0)))

    def pi_eta(pi):
        l, kv, eta_s, eta_p = state(pi)
        a = l * theta * eta_p
        root = math.sqrt(a ** 2 - a * (a + 1.0 - theta) * (theta * eta_s + 1.0 - eta_s))
        return ((a - root) / (a - theta + 1.0)) ** (kv / (kv - 1.0))

    out = []
    for f in (pi_L, pi_eta):
        pi = 10.0
        for _ in range(200):
            new = 0.5 * (pi + f(pi))
            if abs(new - pi) < 1e-6:
                break
            pi = new
        out.append(pi)
    return out


# ================================ ВЫВОД ========================================
def range_warnings(r):
    """Проверка адиабатных КПД по диапазонам с. 7."""
    notes = []
    eta_k, eta_t = r["comp"]["eta"], r["turb"]["eta"]
    if not 0.82 <= eta_k <= 0.90:
        notes.append(f"η*к = {eta_k:.4f} вне диапазона 0,82...0,90 (с. 7)")
    if not 0.87 <= eta_t <= 0.93:
        notes.append(f"η*т = {eta_t:.4f} вне диапазона 0,87...0,93 (с. 7)")
    return notes


def series_table(p, values):
    head = (f"{'π*к':>7}{'T*к,К':>8}{'η*к':>8}{'Lк':>8}{'g_m':>9}{'α':>7}{'π*т':>8}{'η*т':>8}"
            f"{'Lт':>8}{'T*т,К':>8}{'cс,м/с':>8}{'Nв.уд':>8}{'Ne.уд':>8}{'Gв,кг/с':>9}"
            f"{'Ce':>8}{'ηe':>8}")
    lines = [head, "-" * len(head)]
    for pi in values:
        r = cycle(pi, p)
        c, t = r["comp"], r["turb"]
        lines.append(
            f"{pi:7.2f}{c['T_k']:8.1f}{c['eta']:8.4f}{c['L_k']:8.1f}{r['g_m']:9.5f}{r['alpha']:7.2f}"
            f"{t['pi_t']:8.3f}{t['eta']:8.4f}{t['L_t']:8.1f}{t['T_t']:8.1f}{t['c_c']:8.1f}"
            f"{r['N_v_ud']:8.1f}{r['Ne_ud']:8.1f}{r['G_v']:9.3f}{r['Ce']:8.4f}{r['eta_e']:8.4f}")
    return "\n".join(lines)


def detail_report(r, p):
    c, t = r["comp"], r["turb"]
    G_v = r["G_v"]
    G_gas = r["g_gas"] * G_v
    G_fuel = r["g_m"] * (1.0 - p.g_ut - p.g_ohl) * G_v * 3600.0   # с. 45

    p_v = p.p_a * p.sigma_vx
    p_k = p_v * r["pi_k"]
    p_t_static = p.p_a * p.pi_c

    s = split_turbines(r, p)
    p_g, p_t, p_tk = s["p_g"], s["p_t"], s["p_tk"]
    L_tk, tk, L_st, pi_st = s["L_tk"], s["tk"], s["L_st"], s["pi_st"]
    cp_st, k_st, eta_st = s["cp_st"], s["k_st"], s["eta_st"]
    L_st_direct = cp_st * tk["T_out"] * (1.0 - pi_st ** ((1.0 - k_st) / k_st)) * eta_st
    mismatch = 100.0 * (L_st_direct - L_st) / L_st

    N_v = r["N_v_ud"] * G_v
    R_thrust = r["R_ud"] * G_v

    rows = [
        ("РАСЧЁТНАЯ ТОЧКА", None, ""),
        ("Степень повышения давления π*к", f"{r['pi_k']:.3f}", ""),
        ("Степень понижения в турбинах π*т", f"{t['pi_t']:.3f}", ""),
        ("Коэффициент избытка воздуха α", f"{r['alpha']:.3f}", ""),
        ("Относительный расход топлива g_m", f"{r['g_m']:.5f}", ""),
        ("Относительный расход газа g_т", f"{r['g_gas']:.4f}", ""),
        ("КОМПРЕССОР", None, ""),
        ("T*в", f"{p.T_a:.1f}", "К"), ("p*в", f"{p_v / 1000:.2f}", "кПа"),
        ("T*к", f"{c['T_k']:.1f}", "К"), ("p*к", f"{p_k / 1000:.1f}", "кПа"),
        ("cp / k", f"{c['cp']:.4f} / {c['k']:.4f}", "кДж/(кг·К) / —"),
        ("η*к", f"{c['eta']:.4f}", ""), ("Lк", f"{c['L_k']:.2f}", "кДж/кг"),
        ("ТУРБИНЫ, СУММАРНО", None, ""),
        ("T*г", f"{p.T_g:.1f}", "К"), ("p*г", f"{p_g / 1000:.1f}", "кПа"),
        ("cp / k", f"{t['cp']:.4f} / {t['k']:.4f}", "кДж/(кг·К) / —"),
        ("η*т", f"{t['eta']:.4f}", ""), ("Lт", f"{t['L_t']:.2f}", "кДж/кг"),
        ("T*т", f"{t['T_t']:.1f}", "К"), ("p*т", f"{p_t / 1000:.2f}", "кПа"),
        ("ТУРБИНА КОМПРЕССОРА", None, ""),
        ("Lт.к", f"{L_tk:.2f}", "кДж/кг"), ("π*т.к", f"{tk['pi']:.3f}", ""),
        ("η*т.к", f"{tk['eta']:.4f}", ""), ("T* на выходе", f"{tk['T_out']:.1f}", "К"),
        ("p* на выходе", f"{p_tk / 1000:.1f}", "кПа"),
        ("СВОБОДНАЯ ТУРБИНА", None, ""),
        ("Lс.т", f"{L_st:.2f}", "кДж/кг"), ("π*с.т", f"{pi_st:.3f}", ""),
        ("η*с.т", f"{eta_st:.4f}", ""), ("T* на входе", f"{tk['T_out']:.1f}", "К"),
        ("T* на выходе", f"{t['T_t']:.1f}", "К"),
        ("Невязка Lс.т (прямой расчёт к разности)", f"{mismatch:+.2f}", "%"),
        ("СОПЛО", None, ""),
        ("Mт (стенд)", f"{t['M_t']:.3f}", ""), ("πс = pт/pа", f"{p.pi_c:.3f}", ""),
        ("pт (статическое)", f"{p_t_static / 1000:.2f}", "кПа"),
        ("π*с", f"{t['pi_c_star']:.4f}", ""), ("H*с", f"{t['H_c']:.2f}", "кДж/кг"),
        ("cс", f"{t['c_c']:.1f}", "м/с"), ("Rуд", f"{r['R_ud']:.1f}", "Н·с/кг"),
        ("ДВИГАТЕЛЬ", None, ""),
        ("Nв.уд", f"{r['N_v_ud']:.2f}", "кВт·с/кг"),
        ("Ne.уд", f"{r['Ne_ud']:.2f}", "кВт·с/кг"),
        ("Расход воздуха Gв", f"{G_v:.3f}", "кг/с"),
        ("Расход газа Gг", f"{G_gas:.3f}", "кг/с"),
        ("Расход топлива", f"{G_fuel:.1f}", "кг/ч"),
        ("Мощность на винте Nв", f"{N_v:.1f}", "кВт"),
        ("Тяга сопла R", f"{R_thrust:.1f}", "Н"),
        ("Эквивалентная мощность Ne", f"{N_v + R_thrust / p.beta_v:.1f}", "кВт"),
        ("Удельный расход Ce", f"{r['Ce']:.4f}", "кг/(кВт·ч)"),
        ("Эффективный КПД ηe", f"{r['eta_e']:.4f}", ""),
    ]
    out = []
    for name, value, unit in rows:
        out.append(f"\n--- {name} ---" if value is None else f"  {name:<42}{value:>20}  {unit}")
    return "\n".join(out)


def make_plot(pis, G, Ce, pi_L, pi_eta, pi_opt, path):
    """График выбора π*к (рис. 10, с. 46): два поля с общей осью вместо двух шкал."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, hue = "#1f2933", "#6b7280", "#e5e7eb", "#2a6fbb"
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True, constrained_layout=True)
    data = ((axes[0], G, "Расход воздуха Gв, кг/с"), (axes[1], Ce, "Удельный расход Ce, кг/(кВт·ч)"))
    for ax, y, label in data:
        ax.plot(pis, y, color=hue, lw=2)
        ax.set_ylabel(label, color=ink)
        ax.grid(True, color=grid, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=muted)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        marks = ((pi_L, "πL", "--", "right", -4), (pi_opt, "π расч", "-", "left", 4),
                 (pi_eta, "πη", "--", "left", 4))
        for x, name, style, align, shift in marks:
            ax.axvline(x, color=muted, lw=1, ls=style)
            ax.annotate(f"{name} = {x:.1f}", (x, 0.97), xycoords=("data", "axes fraction"),
                        xytext=(shift, 0), textcoords="offset points", color=ink, fontsize=9,
                        va="top", ha=align)
    axes[1].set_xlabel("Степень повышения давления π*к", color=ink)
    fig.suptitle("Выбор π*к: Ne = 1000 кВт, T*г = 1300 К (вариант 2, КПД турбин по ступеням)", color=ink)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def calibrated_params():
    """Увязка КПД турбин с расчётом по ступеням. Расчёт по ступеням от η*т.п не зависит:
    из цикла он берёт только p*г, T*г, работу турбины компрессора, α и pт."""
    p_book = Params()
    stages = turbine_design(cycle(p_book.pi_k_design, p_book), p_book)
    L_stages = (stages["tk"]["totals"]["L"] + stages["st"]["totals"]["L"]) / 1000.0
    lo, hi = 0.70, 0.95
    for _ in range(60):
        eta_tp = 0.5 * (lo + hi)
        if cycle(p_book.pi_k_design, replace(p_book, eta_tp=eta_tp))["turb"]["L_t"] < L_stages:
            lo = eta_tp
        else:
            hi = eta_tp
    return replace(p_book, eta_tp=eta_tp), p_book, L_stages


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    here = Path(__file__).resolve().parent
    p, p_book, L_stages = calibrated_params()

    # Диапазон перебора π*к = υ ... υ^2,4 (с. 12)
    upsilon = p.T_g / p.T_a
    pi_min, pi_max = upsilon, upsilon ** 2.4

    fine = np.arange(pi_min, pi_max + 0.01, 0.01)
    results = [cycle(float(x), p) for x in fine]
    Ne_ud = np.array([r["Ne_ud"] for r in results])
    Ce = np.array([r["Ce"] for r in results])
    G = np.array([r["G_v"] for r in results])
    pi_L = float(fine[np.argmax(Ne_ud)])
    pi_eta = float(fine[np.argmin(Ce)])
    pi_design = p.pi_k_design if p.pi_k_design is not None else 0.5 * (pi_L + pi_eta)
    book_L, book_eta = book_optimum_estimates(p)

    grid = [pi_min] + [x for x in (5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 25, 30, 35)
                       if pi_min < x < pi_max] + [pi_max]

    report = [
        "РАСЧЁТ ЦИКЛА ТВД: Ne = {:.0f} кВт, T*г = {:.0f} К, H = 0, v = 0".format(p.Ne, p.T_g),
        "Схема по прототипу ТВД-20: турбина компрессора осевая двухступенчатая неохлаждаемая,",
        "свободная турбина осевая двухступенчатая. Компрессор прототипа осецентробежный;",
        "политропный КПД принят как для осевого (в пособии нет значения для смешанного).",
        "",
        "ВАРИАНТ 3: материалы и ресурс лопаток назначены (ресурс 10000 ч), kn и σ_доп ТК",
        "пересчитаны под реальный материал (ЖС26, ст.1). Радиальный зазор 0,5 мм (без сотовых",
        "вставок, по заданию руководителя), на РК бандаж с лабиринтным уплотнением (поправка к",
        "η ТГ с. 40-41). Число лопаток: СА нечётное, РК чётное (условие руководителя, сверх",
        "методички). Остальное — как в варианте 2.",
        "",
        f"ВАРИАНТ 2: политропный КПД турбин η*т.п = {p.eta_tp:.4f} подобран по расчёту по ступеням",
        f"(суммарная работа турбин {L_stages:.2f} кДж/кг); в пособии диапазон 0,90...0,91, "
        f"в варианте 1 было {p_book.eta_tp:.3f}.",
        "",
        f"υ = T*г/T*а = {upsilon:.3f};  диапазон перебора π*к = {pi_min:.2f} ... {pi_max:.2f} (с. 12)",
        "",
        "Единицы: T, К; Lк, Lт — кДж/кг; Nв.уд, Ne.уд — кВт·с/кг; Ce — кг/(кВт·ч)",
        series_table(p, grid),
        "",
        "ОПТИМУМЫ π*к",
        f"  по расчётному ряду: πL (max Ne.уд, min Gв) = {pi_L:.2f};  πη (min Ce) = {pi_eta:.2f}",
        f"  по приближённым формулам с. 12:  πL = {book_L:.2f};  πη = {book_eta:.2f}",
        f"  принято для подробного расчёта: π*к = {pi_design:.2f}"
        + ("  (задайте pi_k_design в Params)" if p.pi_k_design is None else ""),
    ]
    design = cycle(pi_design, p)
    report.append(detail_report(design, p))
    report.append(section6_report(design, p))
    report.append(turbine_report(design, p))
    report.append(materials_report(design, p))
    report.append(range_report(design, p))
    notes = range_warnings(design)
    if notes:
        report += ["", "ПРЕДУПРЕЖДЕНИЯ:"] + [f"  {n}" for n in notes]
    text = "\n".join(report)

    print(text)
    (here / "cycle_results_v3.txt").write_text(text, encoding="utf-8")
    make_plot(fine, G, Ce, pi_L, pi_eta, pi_design, here / "cycle_pi_choice_v3.png")


if __name__ == "__main__":
    main()
