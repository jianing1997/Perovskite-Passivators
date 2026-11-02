from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from _shared import PROPERTY_DIR, configure_style, save_png, validate_png


def main() -> None:
    configure_style(18)
    dn = pd.read_csv(PROPERTY_DIR / "DN.csv")["DN"]
    dm = pd.read_csv(PROPERTY_DIR / "DM.csv")["DM"]
    fig, axes = plt.subplots(1, 2, figsize=(12.13, 4.75))
    sns.histplot(dn, bins=30, kde=True, color="#EB8763", edgecolor="black", alpha=0.58, ax=axes[0])
    sns.histplot(dm, bins=30, kde=True, color="#EB8763", edgecolor="black", alpha=0.58, ax=axes[1])
    axes[0].set_xlabel("Donor Number (kcal/mol)")
    axes[1].set_xlabel("Dipole Moment (Debye)")
    for ax in axes:
        ax.set_ylabel("Counts")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
    axes[0].set_xlim(-1, 62)
    axes[1].set_xlim(-0.2, 8.7)
    fig.text(0.005, 0.95, "a", fontsize=25, fontweight="bold")
    fig.text(0.515, 0.95, "b", fontsize=25, fontweight="bold")
    fig.tight_layout(w_pad=2.0)
    path = save_png(fig, "Figure S1.png")
    validate_png(path, 1100, 430)


if __name__ == "__main__":
    main()
