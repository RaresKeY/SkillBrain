"""Generate a Seaborn chart for Agent R performance evolution over 100 days.

Usage:
    .venv/bin/python assignments/hw_adv/a_10_fleet_game/generate_agent_r_evolution_plot.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def build_dataset(days: int = 100, seed: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    day = np.arange(1, days + 1)

    # Simulated Agent R learning curves with diminishing improvements.
    deliveries = 48 + 22 * (1 - np.exp(-day / 25.0)) + rng.normal(0, 1.2, size=days)
    on_time = 72 + 24 * (1 - np.exp(-day / 28.0)) + rng.normal(0, 1.5, size=days)
    route_score = 58 + 30 * (1 - np.exp(-day / 22.0)) + rng.normal(0, 1.3, size=days)
    fuel_eff = 61 + 18 * (1 - np.exp(-day / 32.0)) + rng.normal(0, 1.1, size=days)

    data = pd.DataFrame(
        {
            "Day": day,
            "Deliveries/day": deliveries.clip(0, 100),
            "On-time rate (%)": on_time.clip(0, 100),
            "Route score": route_score.clip(0, 100),
            "Fuel efficiency": fuel_eff.clip(0, 100),
        }
    )
    return data


def main() -> None:
    root = Path(__file__).parent
    out_dir = root / "export"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "agent_r_100_day_evolution.png"

    df = build_dataset(days=100, seed=10)
    plot_df = df.melt(id_vars="Day", var_name="Metric", value_name="Value")

    # Match the dark visual direction used in seaborn visuals in README.
    sns.set_theme(
        style="darkgrid",
        context="talk",
        rc={
            "figure.facecolor": "#1f1f1f",
            "axes.facecolor": "#222222",
            "axes.edgecolor": "#808080",
            "grid.color": "#3f3f3f",
            "text.color": "#d6d6d6",
            "axes.labelcolor": "#d6d6d6",
            "xtick.color": "#c8c8c8",
            "ytick.color": "#c8c8c8",
        },
    )

    palette = {
        "Deliveries/day": "#67d5ff",
        "On-time rate (%)": "#f4a6c8",
        "Route score": "#8bd17c",
        "Fuel efficiency": "#ffd166",
    }

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(
        data=plot_df,
        x="Day",
        y="Value",
        hue="Metric",
        palette=palette,
        linewidth=2.8,
        ax=ax,
    )

    ax.set_title("Fleet Game Agent R Evolution (100 Days)", pad=14, fontsize=20, fontweight="bold")
    ax.set_xlabel("Day")
    ax.set_ylabel("Normalized Performance Index")
    ax.set_xlim(1, 100)
    ax.set_ylim(40, 102)
    ax.legend(title="Metric", loc="lower right", frameon=True)
    for spine in ax.spines.values():
        spine.set_alpha(0.35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
