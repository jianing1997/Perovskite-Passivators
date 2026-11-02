from __future__ import annotations

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from _shared import MODEL_DATA_DIR, configure_style, save_png, validate_png


def main() -> None:
    configure_style(16)
    table = pd.read_csv(MODEL_DATA_DIR / "df_final_DN.csv")
    values = MinMaxScaler().fit_transform(table.iloc[:, 1:])
    profile = pd.Series(values.var(axis=0), index=table.columns[1:]).sort_values(ascending=False)
    colors = cm.plasma(np.linspace(0, 1, len(profile)))
    fig, ax = plt.subplots(figsize=(12.13, 8.92))
    ax.bar(np.arange(len(profile)), profile.values, color=colors, width=0.82)
    ax.set_xlim(-1.5, len(profile) - 0.5)
    ax.set_ylim(0, 0.16)
    ax.set_yticks(np.arange(0, 0.17, 0.04))
    ax.set_xlabel("Descriptor")
    ax.set_ylabel("Standard deviation")
    ax.set_xticks([])
    fig.tight_layout()
    path = save_png(fig, "Figure S2.png")
    validate_png(path, 1100, 800)


if __name__ == "__main__":
    main()
