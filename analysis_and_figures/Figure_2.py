from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from _shared import configure_style, load_run, save_png, validate_png


def main() -> None:
    configure_style(12)
    data = load_run("1e-5_5")["sample_properties"]
    pooled = np.vstack(data)
    colors = mpl.colormaps["viridis"].resampled(len(data))
    rounds = np.arange(1, len(data) + 1)
    mean_dn = np.array([values[:, 0].mean() for values in data])
    mean_dm = np.array([values[:, 1].mean() for values in data])
    joint_score = mean_dn * mean_dm

    fig = plt.figure(figsize=(12.68, 8.9))
    grid = fig.add_gridspec(2, 6, height_ratios=[1.05, 0.95], hspace=0.29, wspace=0.72)
    selected = [0, 4, 9]
    axes_top = [fig.add_subplot(grid[0, 0:2]), fig.add_subplot(grid[0, 2:4]), fig.add_subplot(grid[0, 4:6])]
    for ax, index in zip(axes_top, selected):
        ax.scatter(pooled[:, 0], pooled[:, 1], s=12, color="lightgray", alpha=0.52, edgecolors="none")
        ax.scatter(data[index, :, 0], data[index, :, 1], s=15, color=colors(index), alpha=0.9, edgecolors="none")
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 25)
        ax.set_xlabel("DN (kcal/mol)")
        ax.set_ylabel("DM (Debye)")
        ax.text(0.03, 0.91, f"Round {index + 1}", transform=ax.transAxes, fontsize=13)

    ax_b = fig.add_subplot(grid[1, 0:2])
    ax_c = fig.add_subplot(grid[1, 2:4])
    ax_d = fig.add_subplot(grid[1, 4:6])
    selected_rounds = rounds[selected]
    for ax, values, ylabel, ylimit in [
        (ax_b, mean_dn[selected], "DN (kcal/mol)", (0, 47)),
        (ax_c, mean_dm[selected], "DM (Debye)", (0, 11.8)),
    ]:
        bars = ax.bar(selected_rounds, values, width=0.62, color=[colors(i) for i in selected])
        ax.plot(selected_rounds, values, color="black", marker="o", linewidth=1.4, markersize=4)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + ylimit[1] * 0.012, f"{value:.1f}", ha="center", fontsize=8)
        ax.set_xlim(0.2, 10.8)
        ax.set_ylim(*ylimit)
        ax.set_xticks(selected_rounds)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)

    bars = ax_d.bar(rounds, joint_score, width=0.38, color=[colors(i) for i in range(10)])
    ax_d.plot(rounds, joint_score, color="black", marker="o", linewidth=1.4, markersize=4)
    for bar, value in zip(bars, joint_score):
        ax_d.text(bar.get_x() + bar.get_width() / 2, value + 5, f"{value:.1f}", ha="center", fontsize=7)
    ax_d.set_ylim(0, 530)
    ax_d.set_xticks(rounds)
    ax_d.set_xlabel("Round")
    ax_d.set_ylabel("Joint DN-DM monitoring score")

    fig.text(0.012, 0.955, "a", fontsize=25, fontweight="bold")
    fig.text(0.012, 0.48, "b", fontsize=25, fontweight="bold")
    fig.text(0.35, 0.48, "c", fontsize=25, fontweight="bold")
    fig.text(0.684, 0.48, "d", fontsize=25, fontweight="bold")
    path = save_png(fig, "Figure 2.png")
    validate_png(path, 1100, 750)


if __name__ == "__main__":
    main()
