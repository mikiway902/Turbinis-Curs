"""Графики по оцифровке пользователя (рис. 18/19, таблица с. 58).

Данные — в turbine_prompt_assets/fig18_user.csv, fig19_user.csv,
rho_lD_tab58.csv. Скрипт только строит графики; constraints.py не меняет.
Запуск в контейнере (там есть scipy):
    python Turbinis-Calc/plot_user_digitization.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import PchipInterpolator as CubicSpline

ASSETS = Path(__file__).resolve().parent.parent / "turbine_prompt_assets"
OUT = ASSETS

# ---------------------------------------------------------------- fig 18
d18 = np.loadtxt(ASSETS / "fig18_user.csv", delimiter=",", skiprows=3)
rho18, uc0, uc1 = d18[:, 0], d18[:, 1], d18[:, 2]
rho_fine = np.linspace(0.0, 0.7, 300)
uc0_fit = 0.45 + 0.375 * rho18
uc1_fit = uc0_fit / (0.97 * np.sqrt(1.0 - rho18))

fig, ax = plt.subplots(figsize=(8.2, 5.6))
ax.plot(rho18, uc0, "o-", color="tab:red", lw=1.8, ms=6, label="(u/c₀)η — данные пользователя")
ax.plot(rho18, uc1, "s--", color="tab:blue", lw=1.8, ms=6, label="(u/c₁)η — данные пользователя")
ax.plot(rho_fine, 0.45 + 0.375 * rho_fine, ":", color="darkred", lw=1.3,
        label="(u/c₀)η = 0.45 + 0.375ρ")
ax.plot(rho_fine, (0.45 + 0.375 * rho_fine) / (0.97 * np.sqrt(1 - rho_fine)),
        ":", color="navy", lw=1.3, label="(u/c₁)η = (u/c₀)η/(φ√(1−ρ)), φ=0.97")
for x, y, txt in ((0.2, 0.6, "(0.2; 0.6)"), (0.4, 0.6, "(0.4; 0.6)"), (0.4, 0.8, "(0.4; 0.8)")):
    ax.plot(x, y, "k*", ms=13, zorder=5)
    ax.annotate(txt, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=9)
ax.set_xlim(0, 0.75)
ax.set_ylim(0.4, 1.25)
ax.set_xlabel("ρ")
ax.set_ylabel("(u/c)η")
ax.grid(True, alpha=0.35)
ax.legend(fontsize=8, loc="upper left")
ax.set_title("Рис. 18 — данные пользователя (с. 53)")
fig.tight_layout()
fig.savefig(OUT / "fig18_user.png", dpi=180)
fig.savefig(OUT / "fig18_user.pdf")
plt.close(fig)

# ---------------------------------------------------------------- fig 18 overlay
im18 = Image.open(ASSETS / "fig18_u_ratios.png").crop((528, 206, 1356, 1050))
fig, ax = plt.subplots(figsize=(7.2, 5.6))
ax.imshow(im18, extent=[0.0, 0.8, 0.4, 1.2], aspect="auto")
ax.plot(rho18, uc0, "o-", color="red", lw=1.4, ms=4, label="(u/c₀)η — данные")
ax.plot(rho18, uc1, "s--", color="blue", lw=1.4, ms=4, label="(u/c₁)η — данные")
for x, y in ((0.2, 0.6), (0.4, 0.6), (0.4, 0.8)):
    ax.plot(x, y, "k*", ms=11)
ax.set_xlim(0, 0.8)
ax.set_ylim(0.4, 1.2)
ax.set_xlabel("ρ")
ax.set_ylabel("(u/c)η")
ax.legend(fontsize=8, loc="upper left")
ax.set_title("Рис. 18 — накладка на оригинал (рамка 0…0.8 × 0.4…1.2)")
fig.tight_layout()
fig.savefig(OUT / "fig18_user_overlay.png", dpi=180)
fig.savefig(OUT / "fig18_user_overlay.pdf")
plt.close(fig)

# ---------------------------------------------------------------- fig 19
long = np.loadtxt(ASSETS / "fig19_user.csv", delimiter=",", skiprows=3)
series = {}
for r in np.unique(long[:, 0]):
    m = long[:, 0] == r
    series[float(r)] = (long[m, 1], long[m, 2])

fig, ax = plt.subplots(figsize=(8.2, 5.8))
colors = {0.0: "tab:blue", 0.3: "tab:orange", 0.5: "tab:green"}
for r, (x, y) in series.items():
    order = np.argsort(x)
    x, y = x[order], y[order]
    ax.plot(x, y, "o", color=colors[r], ms=6, label=f"ρ = {r:g} — точки")
    if len(x) >= 4:
        spl = CubicSpline(x, y)
        xx = np.linspace(x.min(), x.max(), 300)
        ax.plot(xx, spl(xx), "-", color=colors[r], lw=1.2, alpha=0.8)
arrow = dict(arrowstyle="-", lw=0.6, color="gray")
ax.plot(0.465, 0.800, "k*", ms=12)
ax.annotate("ρ=0: max 0.800\nпри 0.465", (0.465, 0.800), xytext=(0.245, 0.845),
            textcoords="data", fontsize=8, arrowprops=arrow)
ax.plot(0.6, 0.7, "k*", ms=12)
ax.annotate("узел (0.6; 0.7)", (0.6, 0.7), xytext=(0.50, 0.655),
            textcoords="data", fontsize=8, arrowprops=arrow)
ax.plot(0.56, 0.848, "k*", ms=12)
ax.annotate("ρ=0.3: max 0.848\nпри 0.56", (0.56, 0.848), xytext=(0.255, 0.885),
            textcoords="data", fontsize=8, arrowprops=arrow)
ax.plot(0.63, 0.836, "k*", ms=12)
ax.annotate("пересечение ρ=0.3 и ρ=0.5\n(0.63; 0.836)", (0.63, 0.836),
            xytext=(0.70, 0.885), textcoords="data", fontsize=8, arrowprops=arrow)
ax.set_xlim(0.2, 1.0)
ax.set_ylim(0.48, 0.90)
ax.set_xlabel("u/c₀")
ax.set_ylabel("η_u")
ax.grid(True, alpha=0.35)
ax.legend(fontsize=8, loc="lower left")
ax.set_title("Рис. 19 — данные пользователя (с. 54); линии — PCHIP")
fig.tight_layout()
fig.savefig(OUT / "fig19_user.png", dpi=180)
fig.savefig(OUT / "fig19_user.pdf")
plt.close(fig)

# ---------------------------------------------------------------- fig 19 overlay
im19 = Image.open(ASSETS / "fig19_eta_u.png").crop((414, 138, 1872, 874))
fig, ax = plt.subplots(figsize=(7.4, 5.8))
ax.imshow(im19, extent=[0.2, 1.0, 0.5, 0.9], aspect="auto")
for r, (x, y) in series.items():
    order = np.argsort(x)
    x, y = x[order], y[order]
    ax.plot(x, y, "o-", color=colors[r], ms=4, lw=1.1, label=f"ρ = {r:g}")
    if len(x) >= 4:
        spl = CubicSpline(x, y)
        xx = np.linspace(x.min(), x.max(), 300)
        ax.plot(xx, spl(xx), "-", color=colors[r], lw=1.0, alpha=0.7)
ax.set_xlim(0.2, 1.0)
ax.set_ylim(0.5, 0.9)
ax.set_xlabel("u/c₀")
ax.set_ylabel("η_u")
ax.legend(fontsize=8, loc="lower left")
ax.set_title("Рис. 19 — накладка на оригинал (рамка 0.2…1.0 × 0.5…0.9)")
fig.tight_layout()
fig.savefig(OUT / "fig19_user_overlay.png", dpi=180)
fig.savefig(OUT / "fig19_user_overlay.pdf")
plt.close(fig)

# ---------------------------------------------------------------- table p.58
tab = np.loadtxt(ASSETS / "rho_lD_tab58.csv", delimiter=",", skiprows=2)
lD, rho = tab[:, 0], tab[:, 1]
fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.plot(lD, rho, "o-", color="tab:purple", ms=7)
for x, y, s in zip(lD, rho, ["1/9", "1/6", "1/4", "1/3"]):
    ax.annotate(f"{s}\nρ={y:g}", (x, y), textcoords="offset points", xytext=(6, -14), fontsize=8)
ax.set_xlabel("l_т/D_т")
ax.set_ylabel("ρ_ср")
ax.grid(True, alpha=0.35)
ax.set_title("Таблица с. 58: ρ_ср от l_т/D_т")
fig.tight_layout()
fig.savefig(OUT / "rho_lD_tab58.png", dpi=180)
fig.savefig(OUT / "rho_lD_tab58.pdf")
plt.close(fig)

print("Сохранено в", OUT)
for f in ("fig18_user.png", "fig18_user_overlay.png", "fig19_user.png",
          "fig19_user_overlay.png", "rho_lD_tab58.png"):
    print(" ", f)
