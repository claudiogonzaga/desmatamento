"""
Helpers de visualização para o ABM de Desmatamento / ACP.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import pandas as pd

from model import (
    FOREST, PASTURE, SOY, DEGRADED_PASTURE, REGENERATING,
    APP, LEGAL_RESERVE, WATER, ROAD, EMBARGOED, STATE_NAMES,
)

# ── Cores (RGB 0-1) ─────────────────────────────────────────────────
COLORS = {
    FOREST:           np.array([20, 80, 35]) / 255,
    PASTURE:          np.array([210, 190, 100]) / 255,
    SOY:              np.array([230, 210, 50]) / 255,
    DEGRADED_PASTURE: np.array([140, 110, 50]) / 255,
    REGENERATING:     np.array([50, 120, 60]) / 255,
    APP:              np.array([30, 150, 80]) / 255,
    LEGAL_RESERVE:    np.array([40, 100, 50]) / 255,
    WATER:            np.array([50, 110, 170]) / 255,
    ROAD:             np.array([160, 150, 130]) / 255,
    EMBARGOED:        np.array([180, 40, 40]) / 255,
}

# Estilo global
DARK_BG = "#121212"
PANEL_BG = "#1a1a1a"
TEXT_COLOR = "#e0e0e0"
GRID_ALPHA = 0.15


def _apply_style(fig, ax):
    """Aplica estilo escuro."""
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.grid(True, alpha=GRID_ALPHA, linestyle=":", color="#555555")


def plot_grid_map(grid_array, boundaries=None, figsize=(8, 8)):
    """Painel 1: mapa do município (grid colorido)."""
    N = grid_array.shape[0]
    rgb = np.zeros((N, N, 3))
    for state, color in COLORS.items():
        mask = grid_array == state
        rgb[mask] = color

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    _apply_style(fig, ax)
    ax.imshow(rgb, origin="upper", interpolation="nearest")

    # Limites das propriedades
    if boundaries:
        for b in boundaries:
            edgecolor = "#cc3333" if b["embargoed"] else "#555555"
            linewidth = 1.5 if b["embargoed"] else 0.5
            rect = mpatches.Rectangle(
                (b["x0"] - 0.5, b["y0"] - 0.5),
                b["w"], b["h"],
                linewidth=linewidth,
                edgecolor=edgecolor,
                facecolor="none",
                linestyle="-" if not b["embargoed"] else "--",
            )
            ax.add_patch(rect)

    # Legenda
    legend_items = []
    for state in [FOREST, PASTURE, SOY, DEGRADED_PASTURE, REGENERATING,
                  APP, LEGAL_RESERVE, WATER, ROAD, EMBARGOED]:
        patch = mpatches.Patch(color=COLORS[state], label=STATE_NAMES[state])
        legend_items.append(patch)

    legend = ax.legend(
        handles=legend_items, loc="upper right",
        fontsize=7, framealpha=0.8,
        facecolor=PANEL_BG, edgecolor="#444",
        labelcolor=TEXT_COLOR,
    )

    ax.set_title("Mapa do Município", fontfamily="serif", fontsize=14, pad=10)
    ax.set_xlabel("x (ha)", fontfamily="serif")
    ax.set_ylabel("y (ha)", fontfamily="serif")
    fig.tight_layout()
    return fig


def plot_trajectory(history, figsize=(10, 5)):
    """Painel 2: trajetória temporal — uso do solo + dissuasão."""
    years = [h["year"] for h in history]
    pct_forest = [h["pct_forest"] for h in history]
    pct_pasture = [h["pct_pasture"] for h in history]
    pct_soy = [h["pct_soy"] for h in history]
    pct_degraded = [h["pct_degraded"] for h in history]
    VD_total = [h["VD_total"] for h in history]
    VE = [h["VE"] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 2])
    _apply_style(fig, ax1)
    _apply_style(fig, ax2)

    # Uso do solo
    ax1.plot(years, pct_forest, color="#148523", linewidth=2, label="Floresta (%)")
    ax1.plot(years, pct_pasture, color="#d2be64", linewidth=2, label="Pastagem (%)")
    ax1.plot(years, pct_soy, color="#e6d232", linewidth=1.5, label="Soja (%)")
    ax1.plot(years, pct_degraded, color="#8c6e32", linewidth=1.5,
             linestyle="--", label="Degradada (%)")
    ax1.set_ylabel("Cobertura (%)", fontfamily="serif")
    ax1.set_title("Trajetória de Uso do Solo", fontfamily="serif", fontsize=13, pad=8)
    ax1.legend(fontsize=8, facecolor=PANEL_BG, edgecolor="#444",
               labelcolor=TEXT_COLOR, loc="center right")
    ax1.set_xlim(years[0], years[-1])

    # Dissuasão vs Vantagem
    ax2.fill_between(years, VE, alpha=0.25, color="#cc5555", label="VE (R$/ha)")
    ax2.plot(years, VE, color="#cc5555", linewidth=1.5)
    ax2.plot(years, VD_total, color="#55aacc", linewidth=2, linestyle="--",
             label="VD total (R$/ha)")
    ax2.set_ylabel("R$/ha", fontfamily="serif")
    ax2.set_xlabel("Ano", fontfamily="serif")
    ax2.set_title("Dissuasão (VD) vs. Vantagem Econômica (VE)",
                  fontfamily="serif", fontsize=13, pad=8)
    ax2.legend(fontsize=8, facecolor=PANEL_BG, edgecolor="#444",
               labelcolor=TEXT_COLOR)
    ax2.set_xlim(years[0], years[-1])

    fig.tight_layout()
    return fig


def plot_metrics_bar(snapshot, figsize=(10, 4)):
    """Painel 3: barras de uso do solo + indicadores numéricos do ano."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, width_ratios=[3, 2])
    _apply_style(fig, ax1)
    _apply_style(fig, ax2)

    # Barras
    categories = ["Floresta", "Pastagem", "Soja", "Degradada", "Embargada", "Regen."]
    values = [
        snapshot["pct_forest"],
        snapshot["pct_pasture"],
        snapshot["pct_soy"],
        snapshot["pct_degraded"],
        snapshot["pct_embargoed"],
        snapshot.get("pct_regenerating", 0),
    ]
    bar_colors = [
        COLORS[FOREST],
        COLORS[PASTURE],
        COLORS[SOY],
        COLORS[DEGRADED_PASTURE],
        COLORS[EMBARGOED],
        COLORS[REGENERATING],
    ]

    bars = ax1.barh(categories, values, color=bar_colors, edgecolor="#333")
    ax1.set_xlabel("% da área total", fontfamily="serif")
    ax1.set_title(f"Uso do Solo — Ano {snapshot['year']}",
                  fontfamily="serif", fontsize=13, pad=8)
    ax1.invert_yaxis()

    for bar, val in zip(bars, values):
        if val > 1:
            ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     f"{val:.1f}%", va="center", color=TEXT_COLOR, fontsize=9)

    # Indicadores numéricos
    ax2.axis("off")
    VD_admin = snapshot.get("VD_admin", 0)
    VD_acp = snapshot.get("VD_acp", 0)
    VD_total = snapshot.get("VD_total", 0)
    VE = snapshot.get("VE", 0)
    C = snapshot.get("C", 0)

    lines = [
        ("VD admin", f"R$ {VD_admin:,.2f}/ha"),
        ("VD ACP", f"R$ {VD_acp:,.2f}/ha"),
        ("VD total", f"R$ {VD_total:,.2f}/ha"),
        ("VE", f"R$ {VE:,.2f}/ha"),
        ("C (comportamento)", f"R$ {C:,.2f}/ha"),
    ]

    y_pos = 0.9
    for label, value in lines:
        ax2.text(0.05, y_pos, f"{label}:", fontsize=10, fontfamily="serif",
                 color="#aaaaaa", transform=ax2.transAxes, va="center")
        ax2.text(0.95, y_pos, value, fontsize=10, fontfamily="serif",
                 color=TEXT_COLOR, transform=ax2.transAxes, va="center", ha="right")
        y_pos -= 0.15

    # Semáforo
    semaforo_color = "#cc3333" if C > 0 else "#33aa55"
    semaforo_text = "DESMATAR COMPENSA" if C > 0 else "DESMATAR NÃO COMPENSA"
    ax2.text(0.5, 0.05, semaforo_text, fontsize=11, fontweight="bold",
             fontfamily="serif", color=semaforo_color,
             transform=ax2.transAxes, ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=DARK_BG,
                       edgecolor=semaforo_color, linewidth=2))

    # Métricas de fiscalização
    ax2.text(0.5, 0.92, f"Ano {snapshot['year']}", fontsize=12,
             fontweight="bold", fontfamily="serif", color=TEXT_COLOR,
             transform=ax2.transAxes, ha="center", va="center")

    fig.tight_layout()
    return fig


def plot_comparison(history_a, history_b, label_a="Cenário A", label_b="Cenário B",
                    figsize=(12, 8)):
    """Gráfico comparativo de 2 cenários lado a lado."""
    years_a = [h["year"] for h in history_a]
    years_b = [h["year"] for h in history_b]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for ax_row in axes:
        for ax in ax_row:
            _apply_style(fig, ax)

    # 1. Floresta remanescente
    ax = axes[0, 0]
    ax.plot(years_a, [h["pct_forest"] for h in history_a],
            color="#148523", linewidth=2, label=label_a)
    ax.plot(years_b, [h["pct_forest"] for h in history_b],
            color="#55cc77", linewidth=2, linestyle="--", label=label_b)
    ax.set_title("Floresta Remanescente (%)", fontfamily="serif", fontsize=11)
    ax.set_ylabel("%", fontfamily="serif")
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor="#444", labelcolor=TEXT_COLOR)

    # 2. Desmatamento anual
    ax = axes[0, 1]
    ax.plot(years_a, [h["annual_converted"] for h in history_a],
            color="#cc5555", linewidth=2, label=label_a)
    ax.plot(years_b, [h["annual_converted"] for h in history_b],
            color="#ff8888", linewidth=2, linestyle="--", label=label_b)
    ax.set_title("Desmatamento Anual (ha)", fontfamily="serif", fontsize=11)
    ax.set_ylabel("ha", fontfamily="serif")
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor="#444", labelcolor=TEXT_COLOR)

    # 3. VD total
    ax = axes[1, 0]
    ax.plot(years_a, [h["VD_total"] for h in history_a],
            color="#55aacc", linewidth=2, label=label_a)
    ax.plot(years_b, [h["VD_total"] for h in history_b],
            color="#aaddee", linewidth=2, linestyle="--", label=label_b)
    ax.set_title("Dissuasão Total — VD (R$/ha)", fontfamily="serif", fontsize=11)
    ax.set_ylabel("R$/ha", fontfamily="serif")
    ax.set_xlabel("Ano", fontfamily="serif")
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor="#444", labelcolor=TEXT_COLOR)

    # 4. ACPs acumuladas
    ax = axes[1, 1]
    acps_a = np.cumsum([h["annual_acps"] for h in history_a])
    acps_b = np.cumsum([h["annual_acps"] for h in history_b])
    ax.plot(years_a, acps_a, color="#ddaa33", linewidth=2, label=label_a)
    ax.plot(years_b, acps_b, color="#ffdd77", linewidth=2, linestyle="--", label=label_b)
    ax.set_title("ACPs Acumuladas", fontfamily="serif", fontsize=11)
    ax.set_ylabel("Nº ACPs", fontfamily="serif")
    ax.set_xlabel("Ano", fontfamily="serif")
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor="#444", labelcolor=TEXT_COLOR)

    fig.tight_layout()
    return fig
