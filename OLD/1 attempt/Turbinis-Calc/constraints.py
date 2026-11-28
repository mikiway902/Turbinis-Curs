"""Единый источник книжных ограничений и оцифрованных графиков.

Источник: Михальцев В.Е., Моляков В.Д. «Теория и проектирование газовой
турбины», ч. 1 (далее [1]); ч. 2 — для многоступенчатых турбин.

Правила модуля:
* численные диапазоны, которых нет в [1], не выдумываются;
* всё, что является проектным выбором, лежит в DESIGN_ASSUMPTIONS;
* графики 18 и 19 оцифрованы по скану 600 dpi страниц 53–54 (PDF 54–55)
  и хранятся в turbine_prompt_assets/fig18_uc.csv, fig19_eta.csv;
* интерполяция — только внутри фактического диапазона графика,
  экстраполяция запрещена (ValueError);
* рис. 18 — линейная (кривые почти прямые), рис. 19 — PCHIP
  (сравнение с линейной: отличие ≤0.003 для рис. 18 и до 0.045 у пика η_u).

Оцифровка приближённая: rho ±0.01, u/c ±0.02, eta_u ±0.02.
Перед использованием в отчёте сверить с PDF-оригиналом.
"""
from __future__ import annotations

import numpy as np

try:  # PCHIP — только для сравнения; по умолчанию линейная интерполяция
    from scipy.interpolate import PchipInterpolator as _Pchip
except Exception:  # scipy нет — остаётся линейная
    _Pchip = None

# ---------------------------------------------------------------------------
# 1. Книжные ограничения (машиночитаемые метаданные)
# ---------------------------------------------------------------------------
# kind: hard | project | recommendation | info  (для is_feasible/is_strict)
# status: book_requirement | book_recommendation | project_assumption |
#         derived_check | not_checked
BOOK_CONSTRAINTS = [
    dict(key="alpha2_mid", name="α2 промежуточных ступеней", lo=70.0, hi=90.0,
         unit="deg", kind="hard", status="book_requirement",
         source="[1]", page=38, note="с. 38: 70…90°"),
    dict(key="alpha2_last", name="α2 последней ступени", lo=80.0, hi=90.0,
         unit="deg", kind="hard", status="book_requirement",
         source="[1]", page=38, note="с. 38: 80…90°"),
    dict(key="lD_table", name="l/D — табличные точки", lo=1 / 9, hi=1 / 3,
         unit="-", kind="recommendation", status="book_recommendation",
         source="[1]", page=58, note="1/9→0.2; 1/6→0.3; 1/4→0.4; 1/3→0.5"),
    dict(key="lD_last_58", name="l_т/D_т последней ступени (с. 58)", lo=1 / 4, hi=1 / 3,
         unit="-", kind="recommendation", status="book_recommendation",
         source="[1]", page=58, note="1/4…1/3,5, иногда до 1/3"),
    dict(key="lD_last_87", name="l_т/D_т последней ступени (с. 87, п. 45)", lo=1 / 3.5, hi=1 / 3,
         unit="-", kind="recommendation", status="book_recommendation",
         source="[1]", page=87, note="п. 45: ≤ 1/3,5…1/3; разночтение с с. 58"),
    dict(key="rho_root", name="ρ у корня", lo=0.05, hi=0.10,
         unit="-", kind="hard", status="book_requirement",
         source="[1]", page=58, note="с. 58: ρ ≥ 0,05…0,1"),
    dict(key="n_max", name="n < n_max", lo=None, hi=1.0,
         unit="n/n_max", kind="hard", status="book_requirement",
         source="[1]", page="55–57", note="n < n_max; q_n — проектный запас"),
    dict(key="u_normal", name="u_т (обычные лопатки)", lo=300.0, hi=400.0,
         unit="m/s", kind="recommendation", status="book_recommendation",
         source="[1]", page=59, note="300…400 м/с"),
    dict(key="u_short", name="u_т (короткие лопатки)", lo=300.0, hi=450.0,
         unit="m/s", kind="recommendation", status="book_recommendation",
         source="[1]", page=59, note="400…450 м/с при коротких лопатках"),
    dict(key="sigma_normal", name="σ_л (обычные)", lo=120.0, hi=200.0,
         unit="MPa", kind="recommendation", status="book_recommendation",
         source="[1]", page=57, note="120…200 МПа"),
    dict(key="sigma_stressed", name="σ_л (особо напряжённые)", lo=200.0, hi=300.0,
         unit="MPa", kind="recommendation", status="book_recommendation",
         source="[1]", page=57, note="≤ 300 МПа"),
    dict(key="delta_rel", name="δ_r/l", lo=0.007, hi=0.010,
         unit="-", kind="recommendation", status="book_recommendation",
         source="[1]", page=63, note="0,01…0,007; малые турбины δ=0,5…0,7 мм"),
    dict(key="delta_abs", name="δ_r (абсолютный)", lo=0.0005, hi=0.0007,
         unit="m", kind="recommendation", status="book_recommendation",
         source="[1]", page=63, note="0,5…0,7 мм для малых турбин"),
    dict(key="delta_honeycomb", name="δ_r (соты)", lo=None, hi=0.0002,
         unit="m", kind="recommendation", status="book_recommendation",
         source="[1]", page=12, note="после приработки 0,2 мм и меньше"),
    dict(key="phi", name="φ СА", lo=0.95, hi=0.97,
         unit="-", kind="recommendation", status="book_requirement",
         source="[1]", page=34, note="0,95…0,97"),
    dict(key="psi", name="ψ РК", lo=None, hi=None,
         unit="-", kind="not_checked", status="not_checked",
         source="[1]", page=37, note="численного диапазона книга не даёт"),
    dict(key="kappa", name="κ = c2a/c1a", lo=1.0, hi=None,
         unit="-", kind="info", status="book_recommendation",
         source="[1]", page=62, note="κ>1 — частный случай l_л1=l_л2; не универсально"),
    dict(key="gamma", name="угол раскрытия γ", lo=0.0, hi=25.0,
         unit="deg", kind="hard", status="book_requirement",
         source="[1]", page=56, note="0…20°; >25° — отрыв"),
    dict(key="lambda_c1", name="λ_c1 за СА", lo=None, hi=1.05,
         unit="-", kind="hard", status="book_requirement",
         source="[1]", page=78, note="0,9…1,05; критическое истечение допустимо"),
    dict(key="c_t", name="c_т за турбиной (транспорт)", lo=120.0, hi=200.0,
         unit="m/s", kind="recommendation", status="book_recommendation",
         source="[1]", page=56, note="стац. 80…120; транспорт 120…200; авиа 250…350"),
    dict(key="lba_c", name="l/b_a СА", lo=1.0, hi=2.5,
         unit="-", kind="recommendation", status="book_recommendation",
         source="[1]", page=61, note="ТВД 1…1,5; одноступенчатые 2…2,5"),
    dict(key="lba_l", name="l/b_a РК", lo=1.5, hi=3.0,
         unit="-", kind="recommendation", status="book_recommendation",
         source="[1]", page=61, note="ТВД 1,5…2; одноступенчатые 2,5…3"),
]

# ---------------------------------------------------------------------------
# 2. Проектные допущения (НЕ книга)
# ---------------------------------------------------------------------------
DESIGN_ASSUMPTIONS = {
    "pi_k": dict(value=11.0, note="актуальный Turbinis-Cycle; pi_k=6 — ранняя двухвальная схема TVD_pi6"),
    "N_kW": dict(value=110.0, note="150 hp = 110.325 кВт"),
    "sigma_allow_MPa": dict(value=290.0, category="особо напряжённые, ≤300 (нормальные 120…200)",
                            note="290 — верх специальной категории; требуется обоснование материала/температуры"),
    "q_n": dict(value=0.995, note="минимальный положительный запас n/n_max (0.5 %); книга требует только n<n_max"),
    "psi": dict(value=0.96, note="книга не даёт численного диапазона (с. 37)"),
    "seal_type": dict(value="honeycomb", note="δ_min=0.2 мм по с. 12; обычное исполнение — 0.5…0.7 мм"),
    "delta_rel": dict(value=0.005, note="для сот; ниже книжных 0.007…0.01 для обычного зазора"),
    "kappa": dict(value=[1.38, 1.12, 1.15], note="проектный выбор; книга требует κ>1 только для l_л1=l_л2"),
    "lD_last": dict(value=0.250586, note="нижняя граница с. 58; допустимо до 1/3"),
    "u_max": dict(value=400.0, note="обычные лопатки; для коротких — 450 (проектный критерий)"),
    "rho_root_min": dict(value=0.05, note="нижняя граница книги; можно поднять до 0.1"),
    "lba_min_c": dict(value=1.0, note="низ ТВД по с. 61"),
    "lba_min_l": dict(value=1.5, note="низ ТВД по с. 61"),
    "b_a_min_m": dict(value=0.005, note="конструктивный минимум, не книга"),
}

# ---------------------------------------------------------------------------
# 3. Оцифровка рис. 18 и 19
# ---------------------------------------------------------------------------
# рис. 18, [1] с. 53 (PDF 54)
RHO18 = np.array([0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70])
UC0_18 = np.array([0.450, 0.480, 0.510, 0.540, 0.570, 0.600, 0.625, 0.660])
UC1_18 = np.array([0.450, 0.510, 0.590, 0.675, 0.790, 0.900, 1.020, 1.160])
FIG18_ERR = dict(rho=0.01, uc=0.02)

# рис. 19, [1] с. 54 (PDF 55)
UC0_19 = np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
ETA19_RHO0 = np.array([0.52, 0.64, 0.72, 0.72, 0.64, 0.52, np.nan, np.nan, np.nan])
ETA19_RHO03 = np.array([0.52, 0.64, 0.75, 0.81, 0.82, 0.80, 0.75, 0.69, 0.64])
ETA19_RHO05 = np.array([0.55, 0.67, 0.79, 0.84, 0.84, 0.82, 0.78, 0.75, 0.72])
FIG19_ERR = dict(uc0=0.02, eta=0.02)

# Статус оцифровки: рис. 19 — данные пользователя (PCHIP); рис. 18 — сплошная
# предварительная, пунктир упирается в рамку 1.2. До подтверждения uc0/uc1 — NOT_CHECKED.
DIGITIZATION = {
    # рис. 18: сплошная — предварительная; пунктир упирается в верхнюю рамку 1.2
    "fig18_uc0": "preliminary",
    "fig18_uc1": "frame_clipped",
    # рис. 19: данные пользователя, PCHIP; межкривая интерполяция по ρ — допущение
    "fig19": "verified_pchip",
    "source": "оцифровка пользователя 2026-11-08; см. fig18_user.csv, fig19_user.csv",
    "error": {"fig18_rho": 0.01, "fig18_uc": 0.02, "fig19_uc0": 0.02, "fig19_eta": 0.02},
}

# таблица с. 58: l/D ↔ rho
LD_TABLE = np.array([1 / 9, 1 / 6, 1 / 4, 1 / 3])
RHO_TABLE = np.array([0.20, 0.30, 0.40, 0.50])


def _in_range(x: float, lo: float, hi: float, name: str) -> float:
    x = float(x)
    if not np.isfinite(x) or x < lo - 1e-12 or x > hi + 1e-12:
        raise ValueError(
            f"{name}={x:g} вне диапазона [{lo:g}; {hi:g}]. "
            "Экстраполяция за пределы оцифрованного графика запрещена."
        )
    return x


def _interp(x: float, xp, fp, method: str = "linear") -> float:
    if method == "pchip" and _Pchip is not None:
        return float(_Pchip(xp, fp)(x))
    return float(np.interp(x, xp, fp))


def rho_from_lD(lD: float, method: str = "linear") -> float:
    """ρ_ср по таблице [1] с. 58; только внутри 1/9…1/3."""
    lD = _in_range(lD, LD_TABLE[0], LD_TABLE[-1], "l/D")
    return _interp(lD, LD_TABLE, RHO_TABLE, method)


def lD_from_rho(rho: float, method: str = "linear") -> float:
    rho = _in_range(rho, RHO_TABLE[0], RHO_TABLE[-1], "rho")
    return _interp(rho, RHO_TABLE, LD_TABLE, method)


def uc0_opt_from_rho(rho: float, method: str = "linear") -> float:
    """(u/c0)_η по нижней кривой рис. 18."""
    rho = _in_range(rho, RHO18[0], RHO18[-1], "rho (рис. 18)")
    return _interp(rho, RHO18, UC0_18, method)


def uc1_opt_from_rho(rho: float, method: str = "linear") -> float:
    """(u/c1)_η по верхней (штриховой) кривой рис. 18."""
    rho = _in_range(rho, RHO18[0], RHO18[-1], "rho (рис. 18)")
    return _interp(rho, RHO18, UC1_18, method)


def eta_u_from_uc0_rho(uc0: float, rho: float, method: str = "pchip") -> float:
    """η_u по рис. 19; между кривыми ρ=0, 0.3, 0.5 — линейно.

    Для ρ=0 кривая заканчивается при u/c0≈0.7 — дальше экстраполяции нет.
    method="pchip" — только для сравнения (если есть scipy).
    """
    rho = _in_range(rho, 0.0, 0.5, "rho (рис. 19)")
    vals = []
    for r, eta in ((0.0, ETA19_RHO0), (0.3, ETA19_RHO03), (0.5, ETA19_RHO05)):
        m = np.isfinite(eta)
        x, y = UC0_19[m], eta[m]
        uc0 = _in_range(uc0, x[0], x[-1], f"u/c0 (рис. 19, rho={r:g})")
        vals.append(_interp(uc0, x, y, method))
    return _interp(rho, [0.0, 0.3, 0.5], vals, method)


# ---------------------------------------------------------------------------
# 4. Геометрия и проверки одной ступени
# ---------------------------------------------------------------------------
def stage_geometry(D_mean_c: float, D_mean_l: float, l_c: float, l_l: float) -> dict:
    """Единые определения: D_mean=(D_outer+D_hub)/2, l=(D_outer-D_hub)/2."""
    return {
        "D_mean_c": D_mean_c, "D_mean_l": D_mean_l,
        "l_c": l_c, "l_l": l_l,
        "D_hub_c": D_mean_c - l_c, "D_hub_l": D_mean_l - l_l,
        "D_outer_c": D_mean_c + l_c, "D_outer_l": D_mean_l + l_l,
    }


def _chk(key, name, value, limit, ok, kind, status, source, page, note="", amount=0.0):
    if status == "not_checked":
        result = "NOT_CHECKED"
    elif ok:
        result = "PASS"
    elif kind in ("recommendation", "info"):
        result = "WARNING"
    else:
        result = "FAIL"
    return dict(key=key, name=name, value=value, limit=limit, ok=bool(ok),
                kind=kind, status=status, source=source, page=page,
                note=note, amount=float(amount), result=result)


def stage_checks(s: dict, cfg: dict | None = None) -> list[dict]:
    """Проверки одной ступени по книжным/проектным границам.

    Ожидаемые ключи s: D_c, D_l (или D_л), l_c, l_l (или l_л), alpha1,
    alpha2, pi, sigma, u (средняя), rho, rho_root, x_uc0, x_uc1,
    eta_u, delta, phi, psi, phi_ref, psi_ref.
    Отсутствующие величины пропускаются (NOT_CHECKED не выдумывается).
    """
    cfg = cfg or {}
    L = cfg.get("limits", {})
    out = []
    D_c = s.get("D_c"); D_l = s.get("D_l", s.get("D_л"))
    l_c = s.get("l_c"); l_l = s.get("l_l", s.get("l_л"))
    if all(v is not None for v in (D_c, D_l, l_c, l_l)):
        g = stage_geometry(D_c, D_l, l_c, l_l)
        for tag, Dh, Dm, Do in (("СА", g["D_hub_c"], g["D_mean_c"], g["D_outer_c"]),
                                ("РК", g["D_hub_l"], g["D_mean_l"], g["D_outer_l"])):
            out.append(_chk(f"geom_hub_{tag}", f"D_вт {tag} > 0", f"{Dh*1e3:.1f} мм",
                            "> 0", Dh > 0, "hard", "derived_check", "геометрия", "-",
                            "D_вт = D_ср − l"))
            out.append(_chk(f"geom_order_{tag}", f"D_нар > D_ср > D_вт {tag}",
                            f"{Do*1e3:.1f}/{Dm*1e3:.1f}/{Dh*1e3:.1f} мм",
                            "D_нар > D_ср > D_вт", Do > Dm > Dh, "hard", "derived_check",
                            "геометрия", "-", "D_нар = D_ср + l"))

    a1 = s.get("alpha1", s.get("α1"))
    if a1 is not None:
        amin = float(L.get("a1_min", 11.0))
        out.append(_chk("alpha1", "α1 ≥ 11°", f"{a1:.2f}°", f">= {amin:g}°",
                        a1 >= amin - 1e-9, "hard", "book_requirement", "[1]", "35, 73",
                        "малые турбины 11…14°", max(0.0, (amin - a1) / amin)))

    a2 = s.get("alpha2", s.get("α2"))
    if a2 is not None:
        last = bool(s.get("last", s.get("is_last", False)))
        lo, hi = (80.0, 90.0) if last else (70.0, 90.0)
        out.append(_chk("alpha2", f"α2 ∈ [{lo:g}; {hi:g}]°", f"{a2:.2f}°",
                        f"[{lo:g}; {hi:g}]°", lo - 1e-9 <= a2 <= hi + 1e-9,
                        "hard", "book_requirement", "[1]", "38", "с. 38"))

    pi = s.get("pi", s.get("π_ст"))
    if pi is not None:
        lo, hi = float(L.get("pi_min", 1.8)), float(L.get("pi_max", 2.4))
        out.append(_chk("pi_stage", f"π ст. ∈ [{lo:g}; {hi:g}]", f"{pi:.3f}",
                        f"[{lo:g}; {hi:g}]", lo - 1e-9 <= pi <= hi + 1e-9,
                        "hard", "project_assumption", "проект", "-",
                        "диапазон проекта, не книга"))

    sig = s.get("sigma", s.get("σ_р"))
    sig_dop = s.get("sigma_allow", cfg.get("sigma_allow", 290.0))
    if sig is not None:
        out.append(_chk("sigma", "σ_р ≤ σ_л", f"{sig:.1f} МПа", f"<= {sig_dop:.0f} МПа",
                        sig <= sig_dop * (1.0 + 1e-4), "hard", "derived_check", "[1]",
                        "55–57", "1e-4 — сходимость итерации A_т",
                        max(0.0, (sig - sig_dop) / sig_dop)))

    u = s.get("u")
    if u is not None:
        lD = s.get("lD")
        u_max = float(L.get("u_max", 400.0))
        if lD is not None and lD < 1 / 6:
            u_max = float(L.get("u_max_short", 450.0))
        out.append(_chk("u", "u_т ≤ u_max", f"{u:.1f} м/с", f"<= {u_max:.0f}",
                        u <= u_max + 1e-9, "hard", "book_recommendation", "[1]", "59",
                        "400 обычные; 450 короткие", max(0.0, (u - u_max) / u_max)))

    rho_root = s.get("rho_root")
    if rho_root is not None:
        rmin = float(L.get("rho_root_min", 0.05))
        out.append(_chk("rho_root", "ρ корня ≥ rho_min", f"{rho_root:.3f}", f">= {rmin:g}",
                        rho_root >= rmin - 1e-12, "hard", "book_requirement", "[1]", "58",
                        "0,05…0,1", max(0.0, (rmin - rho_root) / rmin)))

    delta = s.get("delta")
    l_l = s.get("l_l", s.get("l_л"))
    if delta is not None and l_l:
        seal = cfg.get("seal_type", "honeycomb")
        if seal == "honeycomb":
            out.append(_chk("delta_abs", "δ ≥ 0.2 мм (соты)", f"{delta*1e3:.2f} мм",
                            ">= 0.2 мм", delta >= 0.0002 - 1e-12, "recommendation",
                            "book_recommendation", "[1]", "12", "соты, 0,2 мм"))
        else:
            out.append(_chk("delta_rel", "δ/l ∈ [0.007; 0.01]", f"{delta/l_l:.4f}",
                            "[0.007; 0.01]", 0.007 - 1e-9 <= delta / l_l <= 0.01 + 1e-9,
                            "recommendation", "book_recommendation", "[1]", "63",
                            "обычное исполнение"))
            out.append(_chk("delta_abs", "δ ∈ [0.5; 0.7] мм", f"{delta*1e3:.2f} мм",
                            "[0.5; 0.7] мм", 0.0005 - 1e-9 <= delta <= 0.0007 + 1e-9,
                            "recommendation", "book_recommendation", "[1]", "63",
                            "малые турбины"))

    # рис. 18: фактические u/c0, u/c1 против оптимума.
    # Сплошная — предварительная; пунктир упирается в рамку 1.2, не экстраполировать.
    rho = s.get("rho")
    verified0 = DIGITIZATION["fig18_uc0"] == "verified"
    verified1 = DIGITIZATION["fig18_uc1"] == "verified"
    note0 = ("рис. 18; 10 % — проектный допуск" if verified0 else
             "рис. 18, сплошная: предварительная аппроксимация; сверять с оригиналом")
    note1 = ("рис. 18; 10 % — проектный допуск" if verified1 else
             "рис. 18, пунктир: упирается в рамку 1.2 при ρ≈0.65; "
             "φ=0.97 даёт 1.345 при ρ=0.7; не экстраполировать")
    if rho is not None and s.get("x_uc0") is not None:
        try:
            opt = uc0_opt_from_rho(rho)
            rel = abs(s["x_uc0"] / opt - 1.0)
            out.append(_chk("uc0_fig18", "(u/c0) vs рис. 18",
                            f"{s['x_uc0']:.3f} (опт {opt:.3f})", "отклонение ≤ 10 %",
                            (rel <= 0.10) if verified0 else True,
                            "recommendation" if verified0 else "info",
                            "book_recommendation" if verified0 else "not_checked",
                            "[1]", "53", note0, rel))
        except ValueError:
            out.append(_chk("uc0_fig18", "(u/c0) vs рис. 18", f"{s['x_uc0']:.3f}",
                            "вне диапазона", False, "info", "not_checked", "[1]", "53",
                            "ρ вне оцифрованного диапазона"))
    if rho is not None and s.get("x_uc1") is not None:
        try:
            opt = uc1_opt_from_rho(rho)
            rel = abs(s["x_uc1"] / opt - 1.0)
            out.append(_chk("uc1_fig18", "(u/c1) vs рис. 18",
                            f"{s['x_uc1']:.3f} (опт {opt:.3f})", "отклонение ≤ 10 %",
                            (rel <= 0.10) if verified1 else True,
                            "recommendation" if verified1 else "info",
                            "book_recommendation" if verified1 else "not_checked",
                            "[1]", "53", note1, rel))
        except ValueError:
            out.append(_chk("uc1_fig18", "(u/c1) vs рис. 18", f"{s['x_uc1']:.3f}",
                            "вне диапазона", False, "info", "not_checked", "[1]", "53",
                            "ρ вне оцифрованного диапазона"))

    # рис. 19: η_u против графика
    if rho is not None and s.get("x_uc0") is not None and s.get("eta_u") is not None:
        try:
            e = eta_u_from_uc0_rho(s["x_uc0"], rho)
            out.append(_chk("eta_u_fig19", "η_u vs рис. 19",
                            f"{s['eta_u']:.3f} (график {e:.3f})", "справочно",
                            True, "info", "book_recommendation", "[1]", "54",
                            "рис. 19 — только η_u, не η*т; PCHIP по кривым; "
                            "межкривая интерполяция по ρ — допущение", abs(s["eta_u"] - e)))
        except ValueError:
            out.append(_chk("eta_u_fig19", "η_u vs рис. 19", f"{s['eta_u']:.3f}",
                            "вне диапазона", False, "info", "not_checked", "[1]", "54",
                            "u/c0 или ρ вне оцифрованного диапазона"))

    # φ/ψ принятые не выше уточнённых по решёткам
    if s.get("phi") is not None and s.get("phi_ref") is not None:
        out.append(_chk("phi_ref", "φ принятое ≤ φ уточнённое",
                        f"{s['phi']:.3f} / {s['phi_ref']:.3f}", "≤",
                        s["phi"] <= s["phi_ref"] + 1e-9, "recommendation",
                        "book_recommendation", "[1]", "34", "уточнение по решёткам"))
    if s.get("psi") is not None and s.get("psi_ref") is not None:
        out.append(_chk("psi_ref", "ψ принятое ≤ ψ уточнённое",
                        f"{s['psi']:.3f} / {s['psi_ref']:.3f}", "≤",
                        s["psi"] <= s["psi_ref"] + 1e-9, "recommendation",
                        "book_recommendation", "[1]", "37", "уточнение по решёткам"))

    return out
