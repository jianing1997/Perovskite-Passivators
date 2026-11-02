from __future__ import annotations

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from _shared import calculate_structural_diversity, configure_style, load_run, save_png, validate_png


def main() -> None:
    configure_style(22)
    smiles = load_run("1e-5_5")["sample_points"]
    diversity = calculate_structural_diversity(smiles)
    rounds = np.arange(1, len(diversity) + 1)
    colors = cm.viridis(np.linspace(0, 1, len(diversity)))
    fig, ax = plt.subplots(figsize=(10.39, 8.51))
    bars = ax.bar(rounds, diversity, color=colors, width=0.36)
    for bar, value in zip(bars, diversity):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.3f}", ha="center")
    ax.set_xlabel("Round", fontsize=27)
    ax.set_ylabel("Structural diversity", fontsize=27)
    ax.set_xticks(rounds)
    ax.set_ylim(0.75, 0.888)
    fig.tight_layout()
    path = save_png(fig, "Figure S9.png")
    validate_png(path, 950, 750)


if __name__ == "__main__":
    main()
