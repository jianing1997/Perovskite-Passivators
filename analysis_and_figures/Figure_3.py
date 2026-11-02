from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from _shared import FIGURE_DATA_DIR, configure_style, save_png, validate_png


REQUIRED_COLUMNS = {"DN", "DM", "SAS"}


def load_table(filename: str) -> pd.DataFrame:
    path = FIGURE_DATA_DIR / filename
    table = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return table


def main() -> None:
    configure_style(13)
    before = load_table("Before_SAS.csv")
    selected = load_table("optimized_SAS.csv")
    without_color = "#E5A000"
    with_color = "#7A0177"

    # Key quantitative panel 3a. Molecular structure insets are intentionally
    # excluded because they belong to manuscript assembly, not data plotting.
    fig_a, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(
        before["DN"],
        before["DM"],
        s=22,
        color=without_color,
        alpha=0.62,
        edgecolors="none",
        label="Without SAS constraint",
    )
    ax.scatter(
        selected["DN"],
        selected["DM"],
        s=28,
        color=with_color,
        alpha=0.82,
        edgecolors="none",
        label="With SAS constraint",
    )
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 25)
    ax.set_xlabel("DN (kcal/mol)")
    ax.set_ylabel("DM (Debye)")
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    fig_a.tight_layout()
    path_a = save_png(fig_a, "Figure 3a.png")
    validate_png(path_a, 900, 750)

    # Key quantitative panel 3b.
    violin_data = pd.concat(
        [
            before[["SAS"]].assign(Group="Without SAS constraint"),
            selected[["SAS"]].assign(Group="With SAS constraint"),
        ],
        ignore_index=True,
    )
    fig_b, ax = plt.subplots(figsize=(4.1, 4.6))
    sns.violinplot(
        data=violin_data,
        x="Group",
        y="SAS",
        hue="Group",
        palette=[without_color, with_color],
        inner="box",
        cut=0,
        linewidth=1.2,
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("SAS (a.u.)")
    ax.set_xticks([0, 1], ["Without SAS", "With SAS"])
    fig_b.tight_layout()
    path_b = save_png(fig_b, "Figure 3b.png")
    validate_png(path_b, 700, 750)


if __name__ == "__main__":
    main()
